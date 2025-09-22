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

def pick_seed_pixels(img, *, n_seeds=800, colour_thresh=8.0):
    """
    Return ≤ n_seeds (x, y) locations whose BGR values sit inside a tight
    Euclidean ball around the frame-average colour.
    """
    H, W, _ = img.shape

    # build coordinate   [(x0,y0), …] and colour   [(B,G,R), …] arrays
    xs, ys  = np.meshgrid(np.arange(W), np.arange(H))
    coords  = np.column_stack((xs.ravel(), ys.ravel())).astype(np.float32)
    colours = img.reshape(-1, 3).astype(np.float32)

    # centre-of-mass colour & distances to it
    centre  = colours.mean(axis=0)
    dists   = np.linalg.norm(colours - centre, axis=1)

    # keep only those *inside* the given radius; fall back to the closest set
    inball  = np.where(dists < colour_thresh)[0]
    if len(inball) < n_seeds:                       # not enough? relax rule
        inball = np.argsort(dists)[:n_seeds]

    choose  = np.random.choice(inball,
                               size=min(n_seeds, len(inball)),
                               replace=False)
    return coords[choose]                           # (N, 2) float32


def plot_seed_pixels(frame_bgr, seeds_xy, *, save_path=None, dot_size=6):
    """
    Display (or save) the first video frame with the chosen seed pixels over-plotted.

    Parameters
    ----------
    frame_bgr : ndarray (H, W, 3)  uint8
        The frame in BGR colour order (as OpenCV returns).
    seeds_xy  : ndarray (N, 2)  float32
        Output of `pick_seed_pixels` – (x, y) coordinates.
    save_path : str or Path or None
        If given, write the annotated image to this file instead of (or in
        addition to) showing it on screen.
    dot_size  : int
        Marker size for scatter plot.
    """
    # Convert to RGB for correct colours in Matplotlib
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(frame_rgb)
    plt.scatter(seeds_xy[:, 0], seeds_xy[:, 1],
                s=dot_size, c='lime', edgecolors='k', linewidths=0.3)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Seed-pixel preview saved → {save_path.resolve()}")

    plt.show()



# ------------------------------------------------------------------
# main routine
# ------------------------------------------------------------------

def fb_trace_pixels(frames_rgb, lk_params,
                    *, save_npz=True, source_npy_path=None):

    frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_rgb]
    n_frames    = len(frames_rgb)
    H, W        = frames_gray[0].shape

    # seed selection
    pts_all     = pick_seed_pixels(frames_rgb[0], n_seeds=800,
                                   colour_thresh=8.0)        # (M, 2)
    valid       = np.ones(len(pts_all), dtype=bool)

    trajectories = [                                   # no 'region' key
        dict(x0=int(pt[0]), y0=int(pt[1]), rgb=[None]*n_frames)
        for pt in pts_all
    ]

    # initial RGB
    for idx, tr in enumerate(trajectories):
        x, y = tr['x0'], tr['y0']
        trajectories[idx]['rgb'][0] = tuple(int(c)
                                            for c in frames_rgb[0][y, x])

    all_fb_err = []

    # ── time loop ─────────────────────────────────────────────────────────
    for t in range(n_frames - 1):
        prev, nxt = frames_gray[t], frames_gray[t + 1]

        p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(prev, nxt,
                                                pts_all[valid], None,
                                                **lk_params)
        p0r, st_bwd, _ = cv2.calcOpticalFlowPyrLK(nxt, prev, p1, None,
                                                 **lk_params)

        p1, p0r = p1.reshape(-1, 2), p0r.reshape(-1, 2)
        fb_err  = np.linalg.norm(pts_all[valid] - p0r, axis=1)
        all_fb_err.extend(fb_err)

        good    = ((st_fwd.flatten() == 1) & (st_bwd.flatten() == 1) &
                   (fb_err < FB_THRESH_PX_MAX) &
                   (fb_err > FB_THRESH_PX_MIN))

        idx_valid = np.flatnonzero(valid)
        idx_good  = idx_valid[good]

        pts_all[idx_good] = p1[good]
        valid[:]          = False
        valid[idx_good]   = True

        for g_idx, (x_f, y_f) in zip(idx_good, p1[good]):
            xi, yi = int(round(x_f)), int(round(y_f))
            if 0 <= xi < W and 0 <= yi < H:
                trajectories[g_idx]['rgb'][t + 1] = tuple(int(c)
                                                          for c in
                                                          frames_rgb[t + 1]
                                                                   [yi, xi])

    # keep only complete tracks
    trajectories_db = [tr for tr in trajectories
                       if all(p is not None for p in tr['rgb'])]

    # ── unchanged: histogram + saving logic ───────────────────────────────
    if all_fb_err:  # error histogram
        errors = np.asarray(all_fb_err)
        plt.figure(figsize=(8, 4))
        plt.hist(errors, bins=2000, edgecolor='k')
        plt.xlabel("Forward–backward error  |p₀ – p₀ʳ|  [pixels]")
        plt.ylabel("Number of pixel instances")
        plt.title("Pixel tracking error across all frames")
        plt.grid(True, ls="--", alpha=0.4)
        #plt.xlim(0, 0.05)
        plt.xscale('symlog')
        plt.tight_layout()
        plt.show()


    if save_npz and trajectories_db:
        base = Path(source_npy_path).stem if source_npy_path else "unknown_source"
        fname = (f"/Users/henryschnieders/Documents/Research/My_Data/"
                 f"pixel_trajectories_{base}.npz")
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

        # after loading / grabbing the first frame …
    first_frame = data[0]               # (H, W, 3) uint8
    seeds       = pick_seed_pixels(first_frame,
                                n_seeds=800,
                                colour_thresh=8.0)

    plot_seed_pixels(first_frame, seeds,
                    save_path="seed_preview.png")  # omit save_path to just display

    # # inspect the first trajectory
    # traj0 = trajectories_db[0]
    # print(traj0['region'], traj0['x0'], traj0['y0'])
    # print('RGB timeline length:', len(traj0['rgb']))