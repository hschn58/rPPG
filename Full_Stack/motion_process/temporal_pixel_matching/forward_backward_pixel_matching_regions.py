import cv2
import numpy as np
import time
from pathlib import Path
import params  # custom module for facial landmark detection
import matplotlib.pyplot as plt

import argparse

def get_params_from_cli_or_prompt():
    """
    • First try to grab --foo and --bar from the command line.
    • If the user didn't provide them, fall back to asking interactively.
    """
    p = argparse.ArgumentParser(description="Process pixel trajectories and compute region signals from wholeface .npy file.")
    p.add_argument("--data_path", type = str, help = 'pixel trajectories file path' )               # default is None
    args = p.parse_args()

    if args.data_path is None:
        args.data_path = input("Enter the path to the .npy wholeface file:")

    return args.data_path


# ------------------------------------------------------------------
# tunables
# ------------------------------------------------------------------

GRID_SIDE        = 20        # 9×9 grid around each seed
GRID_SPACING_PX  = 2        # pixel pitch inside the grid
FB_THRESH_PX_MIN    = 0.00001
FB_THRESH_PX_MAX    = 0.5   # forward–backward error threshold (in pixels)

# ------------------------------------------------------------------
# helper: dense grid centred on (x, y)
# ------------------------------------------------------------------
def make_grid(x, y, side=GRID_SIDE, pitch=GRID_SPACING_PX):
    half = (side // 2) * pitch
    xs   = np.arange(x - half, x + half + 1, pitch)
    ys   = np.arange(y - half, y + half + 1, pitch)
    xv, yv = np.meshgrid(xs, ys)
    return np.column_stack((xv.ravel(), yv.ravel())).astype(np.float32)

# ------------------------------------------------------------------
# helper: landmark → (x, y)
# each params.* function returns [[row], [col]]
# ------------------------------------------------------------------
def to_xy(coord):
    return int(coord[1][0]), int(coord[0][0])

def get_region_seeds(img, bbox):
    x_c,  y_c  = to_xy(params.chin            (img, bbox, DELTA = 0, region = 'false'))
    x_mlc,y_mlc= to_xy(params.midleft_cheek   (img, bbox, DELTA = 0, region = 'false'))
    x_mrc,y_mrc= to_xy(params.midright_cheek  (img, bbox, DELTA = 0, region = 'false'))
    x_lc, y_lc = to_xy(params.left_cheek      (img, bbox, DELTA = 0, region = 'false'))
    x_rc, y_rc = to_xy(params.right_cheek     (img, bbox, DELTA = 0, region = 'false'))

    return [
        make_grid(x_c,  y_c),   # region 0
        make_grid(x_mlc,y_mlc), # region 1
        make_grid(x_mrc,y_mrc), # region 2
        make_grid(x_lc, y_lc),  # region 3
        make_grid(x_rc, y_rc)   # region 4
    ]

# ------------------------------------------------------------------
# main routine
# ------------------------------------------------------------------

def fb_trace_pixels(frames_rgb, lk_params, save_npz=True, source_npy_path=None):
    """
        Args
            frames_rgb : list / ndarray of BGR images (H,W,3) uint8

        Returns
            trajectories_db : list[dict] – full-length FB-validated pixel tracks
                dict keys: 'region' (0–4), 'x0', 'y0', 'rgb' (list[(B,G,R)])
    """
    frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_rgb]
    n_frames    = len(frames_rgb)
    H, W        = frames_gray[0].shape

    # initial grids
    seeds       = get_region_seeds(frames_rgb[0], [0, 0, W, H])
    pts_all     = np.vstack(seeds).astype(np.float32)          # (M,2)
    reg_index   = np.concatenate([[i]*len(p) for i, p in enumerate(seeds)])
    valid       = np.ones(len(pts_all), dtype=bool)

    all_fb_err  = []                  # ← NEW: collect per-frame FB errors

    # DB for RGB evolution (pre-filled with None)
    trajectories = [
        dict(region=int(r),
             x0=int(pt[0]), y0=int(pt[1]),
             rgb=[None]*n_frames)
        for r, pt in zip(reg_index, pts_all)
    ]

    # fill frame 0 RGB
    for idx, tr in enumerate(trajectories):
        x, y = tr['x0'], tr['y0']
        if 0 <= x < W and 0 <= y < H:
            trajectories[idx]['rgb'][0] = tuple(int(c) for c in frames_rgb[0][y, x])

    # time loop
    for t in range(n_frames - 1):
        prev, nxt = frames_gray[t], frames_gray[t + 1]

        p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(prev, nxt, pts_all[valid], None, **lk_params)
        p0r, st_bwd, _ = cv2.calcOpticalFlowPyrLK(nxt, prev, p1,             None, **lk_params)

        p1  = p1.reshape(-1, 2)
        p0r = p0r.reshape(-1, 2)

        fb_err = np.linalg.norm(pts_all[valid] - p0r, axis=1)
        all_fb_err.extend(fb_err)     # ← NEW

        good_local = ((st_fwd.flatten() == 1) & (st_bwd.flatten() == 1) &
                      (fb_err < FB_THRESH_PX_MAX) & (fb_err > FB_THRESH_PX_MIN)) 

        # global indices corresponding to 'good_local'
        idx_valid = np.flatnonzero(valid)
        idx_good  = idx_valid[good_local]

        # update positions
        pts_all[idx_good] = p1[good_local]
        valid[:] = False
        valid[idx_good] = True

        # write RGB for all still-valid tracks at frame t+1
        for g_idx, (x_f, y_f) in zip(idx_good, p1[good_local]):
            x_i, y_i = int(round(x_f)), int(round(y_f))
            if 0 <= x_i < W and 0 <= y_i < H:
                trajectories[g_idx]['rgb'][t + 1] = tuple(int(c) for c in frames_rgb[t + 1][y_i, x_i])

    # keep only full-length tracks (no None left)
    trajectories_db = [
        tr for tr in trajectories
        if all(pixel is not None for pixel in tr['rgb'])
    ]

     # ── NEW: show distribution of forward–backward errors ──────────────
    if all_fb_err:                                  # non-empty safeguard
        errors = np.asarray(all_fb_err)
        plt.figure(figsize=(8, 4))
        plt.hist(errors, bins=2000, edgecolor='k')
        plt.xlabel("Forward–backward error  |p₀ – p₀ʳ|  [pixels]")
        plt.ylabel("Number of pixel instances")
        plt.title("Pixel tracking error across all frames")
        plt.grid(True, ls="--", alpha=0.4)
        plt.xlim(0, 0.05)
        plt.xscale('symlog')  # log scale for better visibility
        plt.tight_layout()
        plt.show()


    # after you’ve built trajectories_db:
    if save_npz and trajectories_db:
        if source_npy_path:
            base = Path(source_npy_path).stem
        else:
            base = "unknown_source"
        fname = (
            f"/Users/henryschnieders/Documents/Research/My_Data/"
            f"pixel_trajectories_{base}.npz"
        )
        np.savez_compressed(fname, trajectories=trajectories_db)
        print(f"• saved {len(trajectories_db)} complete tracks → {Path(fname).resolve()}")

    return trajectories_db

# Usage:

if __name__ == "__main__":

    data_path = get_params_from_cli_or_prompt()

    lk_params = dict(
        winSize  =(21, 21),
        maxLevel =3,
        criteria =(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    data = np.load(data_path, allow_pickle=True)

    trajectories_db = fb_trace_pixels(
        data,
        lk_params=lk_params,
        save_npz=True,
        source_npy_path=data_path
    )
    print(f"{len(trajectories_db)} pixels tracked flawlessly through the video.")

    # inspect the first trajectory
    traj0 = trajectories_db[0]
    print(traj0['region'], traj0['x0'], traj0['y0'])
    print('RGB timeline length:', len(traj0['rgb']))