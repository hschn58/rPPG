#  ──────────────────────────────────────────────────────────────────────────────
#  dynamic_roi_optical_flow.py   •   Henry Schnieders – June 2025
#  -----------------------------------------------------------------------------
#  Pipeline:
#     1.  Load full-frame RGB stack from <video_name>_wholeframe.npy
#     2.  Detect face landmarks with MediaPipe FaceMesh (YuNet fallback)
#     3.  Build adaptive ROI masks (forehead + cheeks) per frame
#     4.  Visibility + yaw filtering with hysteresis
#     5.  Colour-homogeneous pixel clustering inside each ROI
#     6.  LK (CUDA if available) forward–backward optical flow in 6-s windows
#     7.  Save validated pixel trajectories to <video_name>_tracks.npz
#
#  Dependencies:  opencv-python-headless  mediapipe  numpy  (opencv-contrib-python-cuda: optional)
#  -----------------------------------------------------------------------------

# 0 ── imports & CLI ────────────────────────────────────────────────────────────
import argparse, collections, os, time, sys, math, json
from pathlib import Path

import cv2, numpy as np
import mediapipe as mp


def get_params_from_cli_or_prompt():
    """
    • First try to grab --foo and --bar from the command line.
    • If the user didn't provide them, fall back to asking interactively.
    """
# 1 ── argument parser ─────────────────────────────────────────────────────────
    CLI = argparse.ArgumentParser(description="Dynamic-ROI optical-flow tracker")
    CLI.add_argument("--npy",  default = None, help = "*_wholeframe.npy produced by capture script")
    CLI.add_argument("--out",  default = None, help = "output .npz (default: <npy> -> _tracks.npz)")
    CLI.add_argument("--fps",  type=int, default=30)
    args = CLI.parse_args()

    while True:
        if args.npy is None:
            args.npy = '/Users/henryschnieders/Documents/Research/My_Data/mornin_6_16_wholeface.npy' #input('Enter whole-frame .npy file: ')

            if not args.npy.endswith(".npy"):
                continue

            else:
                break


    while True:
        if args.out is None:
            args.out = '/Users/henryschnieders/Documents/Research/My_Data/mornin_6_16_trajectories.npz' #input('Enter output path for .npz trajectories: ')

            if not args.out.endswith(".npz"):
                continue

            else:
                break

    return args.npy, args.out, args.fps



npy_file, out_file, fps = get_params_from_cli_or_prompt()

in_npy  = Path(npy_file).expanduser()
if not in_npy.exists():
    sys.exit(f"❌  file not found: {in_npy}")

out_npz = Path(out_file) if out_file else in_npy.with_name(in_npy.stem.replace("_wholeframe", "_tracks") + ".npz")

# 2 ── Face detectors ──────────────────────────────────────────────────────────
mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                         max_num_faces=1,
                                         refine_landmarks=True,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)

# YuNet fallback (only called when FaceMesh fails > N frames)
yunet_path = "/Users/henryschnieders/Documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
yunet_det  = cv2.FaceDetectorYN_create(yunet_path, "", (0, 0))

# 3 ── ROI landmark groups (indices follow MediaPipe 468-pt mesh) ──────────────
ROI_IDXS = dict(
    forehead=[10, 67, 297, 336, 338, 151, 299, 9],
    l_cheek=[234, 93, 132, 58, 172, 136, 150],
    r_cheek=[454, 323, 361, 288, 397, 365, 379])
ROI_KEYS = list(ROI_IDXS)

# 4 ── visibility / hysteresis parameters ──────────────────────────────────────
AREA_MIN      = 400                 # px² minimal ROI area
YAW_THRESH    = 35                   # deg – hide far cheek past this
DUR_ON, DUR_OFF = 3, 2              # frames hysteresis

# 5 ── optical-flow parameters ─────────────────────────────────────────────────
lk_win  = (21, 21)
lk_lvl  = 3
lk_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

has_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
if has_cuda:
    lk_gpu = cv2.cuda.SparsePyrLKOpticalFlow_create(winSize=lk_win, maxLevel=lk_lvl, iters=30, useInitialFlow=False)

# 6 ── helper: head-pose yaw estimation via PnP ────────────────────────────────
_IMG_IDXS = [1, 33, 61, 291, 199, 199]
_OBJ_PTS  = np.float32([[0,0,0], [-30,32,-30], [30,32,-30],
                        [-75,-20,-50], [0,-77,-5], [0,-77,-5]])

def yaw_from_landmarks(lm_xy, w, h):
    cam = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], np.float32)
    ok, rvec, _ = cv2.solvePnP(_OBJ_PTS, lm_xy[_IMG_IDXS].astype(np.float32), cam, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0
    rot,_ = cv2.Rodrigues(rvec)
    yaw  = math.degrees(math.asin(np.clip(rot[1,0], -1, 1)))
    return yaw

# 7 ── helper: adaptive ROIs & seed grids ─────────────────────────────────────-

def build_rois(frame_rgb, lm_xy, grid_pitch=2, grid_cap=81):
    h, w, _ = frame_rgb.shape
    masks, seeds = [], []
    for idxs in ROI_IDXS.values():
        poly = lm_xy[idxs].astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, poly, 1)
        masks.append(mask.astype(bool))
        # grid seeds inside mask
        ys, xs = np.where(mask)
        if xs.size == 0:
            seeds.append(np.empty((0,2), np.float32)); continue
        xs_grid = np.arange(xs.min(), xs.max()+1, grid_pitch)
        ys_grid = np.arange(ys.min(), ys.max()+1, grid_pitch)
        xv, yv  = np.meshgrid(xs_grid, ys_grid)
        pts     = np.column_stack([xv.ravel(), yv.ravel()]).astype(np.float32)
        inside  = mask[pts[:,1].astype(int), pts[:,0].astype(int)]
        seeds.append(pts[inside][:grid_cap])
    return masks, seeds

# 8 ── helper: colour-homogeneous pixel filter ─────────────────────────────────

def homogeneous_pixels(frame_rgb, mask, eps=6.0, min_keep=0.1):
    """Return boolean mask subset inside *mask* whose Lab dist ≤ eps from main cluster."""
    
    sel_px = frame_rgb[mask]   # shape (N,3)
    if sel_px.size == 0:
        return mask  # nothing better
    
    lab = cv2.cvtColor(sel_px.reshape(-1,1,3), cv2.COLOR_RGB2Lab).reshape(-1,3)
    # k-means 3 clusters, pick largest
    _, labels, ctrs = cv2.kmeans(lab.astype(np.float32), 3, None,
                                 (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                 3, cv2.KMEANS_PP_CENTERS)
    main = ctrs[np.bincount(labels.ravel()).argmax()]
    good = np.linalg.norm(lab-main, axis=1) < eps

            # fallback: if we kept < min_keep of pixels, disable the filter

    if good.mean() < min_keep:
        return mask
    
    mask_out = np.zeros_like(mask)
    mask_out[mask] = good

    return mask_out

# 9 ── main processing ─────────────────────────────────────────────────────────
print("📂  loading frames …", end="", flush=True)
frames = np.load(in_npy, mmap_mode="r", allow_pickle=True)                # (T,H,W,3) uint8 RGB
frames = np.ascontiguousarray(frames, np.uint8)  # ensure contiguous memory
T, H, W, _ = frames.shape
print(f"  {T} frames, {W}×{H}")

hyst = {k:0 for k in ROI_KEYS}                          # hysteresis counters
tracks = []                                             # output list of dicts
fail_count = 0                                          # FaceMesh failure counter

WINDOW_SEC = 6;  STEP_SEC = 3
WIN = int(WINDOW_SEC * fps)
STEP = int(STEP_SEC * fps)

def add_track(region, x0, y0, rgb):
    tracks.append(dict(region=region, x0=float(x0), y0=float(y0), rgb=rgb))

for t0 in range(0, T-WIN+1, STEP):
    t1 = t0 + WIN
    window = frames[t0:t1]                              # view, not copy

    # --- FaceMesh on first frame of window -----------------------------------
    lm_ok = False
    for retry in range(3):                              # try current frame + 2 back-offs
        frm = window[retry]
        res = mp_mesh.process(frm)
        if res.multi_face_landmarks:
            lm_ok = True
            break
    if not lm_ok:
        fail_count += 1
        if fail_count >= 5:
            # fallback YuNet on RGB→BGR
            yunet_det.setInputSize((W, H))
            _, boxes = yunet_det.detect(frm[..., ::-1])
            if boxes is not None and len(boxes):
                fail_count = 0   # reset – FaceMesh will likely pick up next iter
        continue
    fail_count = 0

    lm = res.multi_face_landmarks[0].landmark
    lm_xy = np.array([(p.x*W, p.y*H) for p in lm], np.float32)
    yaw = yaw_from_landmarks(lm_xy, W, H)

    roi_masks, roi_seeds = build_rois(frm, lm_xy)

    # --- visibility + hysteresis --------------------------------------------
    roi_ok = []
    for k, m in zip(ROI_KEYS, roi_masks):
        area = int(m.sum())
        ok = area >= AREA_MIN
        if k == "l_cheek" and yaw < -YAW_THRESH:
            ok = False
        if k == "r_cheek" and yaw >  YAW_THRESH:
            ok = False
        # hysteresis counter update
        if ok:
            hyst[k] = min(hyst[k]+1, DUR_ON)
        else:
            hyst[k] = max(hyst[k]-1, -DUR_OFF)
        roi_ok.append(hyst[k] > 0)

    # --- diagnostic --------------------------------------------------------------
    if True:
        print(f"\n[{t0:5d}]  ROI diagnostic "
            f"(yaw={yaw:+.1f}°, win {t0}→{t0+WIN-1})")
        for rid, k in enumerate(ROI_KEYS):
            area = int(roi_masks[rid].sum())
            seeds_raw = len(roi_seeds[rid])
            print(f"  {k:<9} area={area:5d}px²  "
                f"hyst={hyst[k]:+2d}  "
                f"roi_ok={roi_ok[rid]}  "
                f"raw_seeds={seeds_raw}")

    # --- per-ROI colour mask & seed thinning ---------------------------------
    pick_seeds = {}
    for rid, (k, m, seeds) in enumerate(zip(ROI_KEYS, roi_masks, roi_seeds)):
        if not roi_ok[rid] or seeds.size == 0:
            continue
        m_skin = homogeneous_pixels(frm, m)
        # choose subset of seeds that fall on m_skin
        m_skin = cv2.dilate(m_skin.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool)

        good = m_skin[seeds[:,1].astype(int), seeds[:,0].astype(int)].astype(bool)
        sel  = seeds[good]

        if sel.size == 0:
            print(f"ROI {k}: all {len(seeds)} seeds filtered out")
            continue

        pick_seeds[rid] = sel

    print(f"[{t0:5d}] seeds this window:",
        {rid: len(v) for rid, v in pick_seeds.items()})

    if not pick_seeds:
        continue


    

    # --- optical flow across window -----------------------------------------
    prev = window[0][..., ::-1]                         # BGR for OpenCV
    pts0 = np.vstack(list(pick_seeds.values())).astype(np.float32)
    region_id = np.concatenate([[rid]*len(p) for rid,p in pick_seeds.items()])

    # GPU path
    if has_cuda:
        gpu_prev = cv2.cuda_GpuMat(); gpu_prev.upload(prev[:,:,0])  # upload gray once
        gpu_prev = cv2.cuda_GpuMat()
        gpu_prev.upload(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY))      # correct gray upload

        # we need P1-level greyscale for LK – convert inside loop for simplicity
    pts_traj = np.empty((len(pts0), WIN, 2), np.float32)
    pts_traj[:,0] = pts0

    good_mask = np.ones(len(pts0), bool)                # keep status

    for fidx in range(1, WIN):
        nxt = window[fidx][..., ::-1]   # BGR
        if has_cuda:
            gpu_nxt = cv2.cuda_GpuMat(); gpu_nxt.upload(cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY))
            p1, st = lk_gpu.calc(gpu_prev, gpu_nxt, pts_traj[good_mask,fidx-1])
            gpu_prev = gpu_nxt
            p1 = p1.download()
        else:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
                                                 cv2.cvtColor(nxt,  cv2.COLOR_BGR2GRAY),
                                                 pts_traj[good_mask,fidx-1], None,
                                                 winSize=lk_win, maxLevel=lk_lvl, criteria=lk_crit)
            prev = nxt
        st = st.ravel() > 0
        # update trajectories & status
        good_idx = np.where(good_mask)[0]
        pts_traj[good_idx[~st], :] = np.nan            # lost – mark NaN
        good_mask[good_idx[~st]] = False
        pts_traj[good_idx[st], fidx] = p1[st]

    # --- save surviving tracks where >½ frames are valid --------------------
    for i, ok in enumerate(good_mask):
        if not ok: continue
        traj = pts_traj[i]
        valid = np.isfinite(traj[:,0])
        if valid.sum() < WIN//2:           # too short
            continue
        rgb_series = window[:, traj[0,1].astype(int), traj[0,0].astype(int)]  # (T,3)

        xy_int = np.round(traj[valid]).astype(int)          # (F_valid, 2)
        rgb_series = window[valid, xy_int[:,1], xy_int[:,0]]


        add_track(int(region_id[i]), traj[0,0], traj[0,1], rgb_series.tolist())

    if t0 == 180:  # or any window you want to watch
        print(f"\n[{t0:5d}]  ROI diagnostic")
        for rid, k in enumerate(ROI_KEYS):
            area = int(roi_masks[rid].sum())
            print(f"  {k:<9} area={area:6d}  hyst={hyst[k]:+2d}  "
                f"ok={roi_ok[rid]}  seeds={len(roi_seeds[rid])}")



print(f"[{t0:5d}] seeds this window:",
    {k: len(v) for k, v in pick_seeds.items()})

print(f"💾  writing {len(tracks)} tracks → {out_npz}")
np.savez_compressed(out_npz, tracks=tracks, fps=fps)
print("✅  done")
