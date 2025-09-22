import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import os 

import argparse 

import subprocess



# --------------------- configurable ---------------------
phi_tol      = 0.99         # cosine threshold
fps          = 30            # frames per second
n_regions    = 5             # chin, mid-L, mid-R, L, R
# --------------------------------------------------------
def complex_mag(x):
    return np.abs(x)

def signal_process_alter(pix_intensity, fps, num = 5):
    sampling_rate = fps
    lowcut_heart, highcut_heart = 0.5, 4.0

    N = len(pix_intensity)
    freqs = np.fft.fftfreq(N, d=1.0 / sampling_rate)
    coefs = np.fft.fft(pix_intensity)

    keep = (np.abs(freqs) >= lowcut_heart) & (np.abs(freqs) <= highcut_heart)
    coefs, freqs = coefs[keep], freqs[keep]

    idx = np.argsort(complex_mag(coefs))[-num:]
    coefs, freqs = coefs[idx], freqs[idx]

    bpm = np.average(np.abs(freqs), weights=complex_mag(coefs)) * 60
    return bpm, freqs

# --------------------------------------
def build_gray(rgb_arr):
    """
    Return per-frame Euclidean norm ‖(R,G,B)‖₂.
    """
    rgb_arr = rgb_arr.astype(np.float32)
    return np.linalg.norm(rgb_arr, axis=1).astype(np.float32)

def process_tracks_motion_sigma(db,
                                n_regions,
                                std_thresh=5,      # “X σ” threshold
                                window_frac=0.10,  # sliding-window size for z-score normalisation
                                eps=1e-12):
    """
    Build region-level grey / RGB / motion signals after discarding, frame-by-frame,
    any track whose motion magnitude deviates more than `std_thresh` standard
    deviations from the region mean for that frame.

    Parameters
    ----------
    db : list of dict
        Your per-pixel track database (each item must have keys 'rgb', 'region', 'x0', 'y0').
    n_regions : int
        Number of distinct facial regions in the DB.
    std_thresh : float, optional
        Keep sensors with |mag − μ| ≤ std_thresh · σ.  Default = 3.
    window_frac : float, optional
        Fraction of track length used for centred sliding-window z-scoring.  Default = 0.10.
    eps : float, optional
        Small value to avoid division by zero.
    """
    # ---------- pass 1: per-track normalisation + motion magnitude ----------
    n_frames = len(db[0]['rgb'])
    window_size = max(1, int(len(db[0]['rgb']) * window_frac))
    half        = window_size // 2

    region_tracks, id_map = [[] for _ in range(n_regions)], [{} for _ in range(n_regions)]

    for tr in db:
        rgb_raw  = np.asarray(tr['rgb'], np.float32)              # (T,3)
        rgb_norm = np.empty_like(rgb_raw)
        T        = len(rgb_raw)

        # --- sliding-window z-score ---
        for t in range(T):
            start = max(0, min(t - half, T - window_size))
            window  = rgb_raw[start:start + window_size]
            mean    = window.mean(axis=0, keepdims=True)
            std     = window.std (axis=0, keepdims=True) + eps
            rgb_norm[t] = rgb_raw[t]

        # --- motion magnitude (between consecutive frames) ---
        motion      = np.diff(rgb_norm, axis=0)                   # (T-1, 3)
        motion_mag  = np.linalg.norm(motion, axis=1)              # (T-1,)

        track = dict(
            rgb        = rgb_norm,
            gray       = np.linalg.norm(rgb_norm, axis=1),        # (T,)
            motion_mag = motion_mag,                              # (T-1,)
            id         = f"x{tr['x0']}_y{tr['y0']}"
        )
        reg = tr['region']
        id_map[reg][track['id']] = len(region_tracks[reg])
        region_tracks[reg].append(track)

    # ---------- pass 2: region-level σ-filter & aggregation ----------
    region_signals, motion_signals, RGB_region_signals = [], [], []
    for reg in range(n_regions):
        P = len(region_tracks[reg])
        if P == 0:
            region_signals.append(np.full(n_frames, np.nan))
            motion_signals.append(np.full(n_frames, np.nan))
            RGB_region_signals.append(np.full((n_frames,3), np.nan))
            continue

        # matrices
        G      = np.stack([trk['gray']       for trk in region_tracks[reg]])   # P×T
        G_RGB  = np.stack([trk['rgb']        for trk in region_tracks[reg]])   # P×T×3
        M      = np.stack([trk['motion_mag'] for trk in region_tracks[reg]])   # P×(T-1)

        # --- σ-filter: keep sensors inside ± std_thresh σ for each (frame-1) ---
        mu    = M.mean(axis=0, keepdims=True)
        sigma = M.std (axis=0, keepdims=True) + eps
        MS    = (np.abs(M - mu) <= std_thresh * sigma).astype(np.uint8)   # P×(T-1)
        MS    = MS[:, 1:]                # align to frames 1…T-2  ⇒  P×(T-2)

        # ---------- aggregate with mask ----------
        sig      = np.zeros(n_frames, np.float32)
        RGB_sig  = np.zeros((n_frames, 3), np.float32)
        sig[0]   = np.median(G[:, 0])
        sig[-1]  = np.median(G[:, -1])
        RGB_sig[0]  = np.median(G_RGB[:, 0],  axis=0)
        RGB_sig[-1] = np.median(G_RGB[:, -1], axis=0)

        for t in range(1, n_frames - 1):            # frames 1 … T-2
            good           = MS[:, t - 1].astype(bool)
            sig[t]         = np.median(G[good, t])               if good.any() else np.nan
            RGB_sig[t]     = np.median(G_RGB[good, t], axis=0)   if good.any() else np.nan

        mot = np.zeros(n_frames, np.float32)
        mot[:] = np.nan          # undefined at frames 0 & T-1
        for t in range(1, n_frames - 1):
            good     = MS[:, t - 1].astype(bool)
            mot[t]   = np.median(M[good, t]) if good.any() else np.nan

        region_signals   .append(sig)
        motion_signals   .append(mot)
        RGB_region_signals.append(RGB_sig)

    # quick diagnostic: sensors kept per (region-last) frame
    good_per_frame = MS.sum(axis=0)          # shape (T-2,)
    plt.figure(figsize=(10, 4))
    plt.plot(good_per_frame)
    plt.title(f'# sensors kept (±{std_thresh}σ)')

    return region_signals, motion_signals, id_map, RGB_region_signals



def norm_rgb_regions(rgb_regions, window_frac=0.10, eps=1e-8):
    """
    Sliding-window z-score per channel (R,G,B) for each region.
    `rgb_regions` is a list/ndarray of shape (N_regions, T, 3).

    Returns a *new* ndarray with the same shape, normalised.
    """
    normed = []
    for arr in rgb_regions:                      # arr: (T, 3)
        T   = arr.shape[0]
        win = max(1, int(round(T * window_frac)))
        half = win // 2
        out  = np.empty_like(arr, dtype=np.float32)

        for t in range(T):
            # centred window, clamped to [0, T-win]
            start = max(0, min(t - half, T - win))
            window = arr[start:start + win]
            mu  = window.mean(axis=0, keepdims=True)
            sd  = window.std (axis=0, keepdims=True) + eps
            out[t] = (arr[t] ) / mu
        normed.append(out)

    return np.stack(normed)                      # (N_regions, T, 3)


def get_params_from_cli_or_prompt():
    """
    • First try to grab --foo and --bar from the command line.
    • If the user didn't provide them, fall back to asking interactively.
    """
    p = argparse.ArgumentParser(description="Process pixel trajectories and compute region signals.")
    p.add_argument("--data_path", type = str, help = 'pixel trajectories file path' )               # default is None
    args = p.parse_args()

    if args.data_path is None:
        args.data_path = input("Enter the path to the pixel trajectories file:")

    return args.data_path



# -------------------- driver demo --------------------

if __name__ == '__main__':

    pix_trajectories = get_params_from_cli_or_prompt()
    db = np.load(pix_trajectories, allow_pickle=True)['trajectories'].tolist()
    region_signals, motion_signals, id_map, RGB_region_signals = process_tracks_motion_sigma(db, n_regions= 5)


    # --- remove first and last frame from all signals because no mask for those frames -----------------
    cut = 25

    region_signals = [sig[cut:-cut] for sig in region_signals]
    motion_signals = [mot[cut:-cut] for mot in motion_signals]

    #     # --- NEW:  window-normalise each region’s RGB trace ----------------------
    # RGB_region_signals = norm_rgb_regions(RGB_region_signals,  # ← just added
    #                                         window_frac=0.10)    # (10 % window)
    

    base = os.path.basename(pix_trajectories).split('.')[0].replace('pixel_trajectories_', '')  # e.g. "testagain_6_10_wholeface"

    out_file  = f"/Users/henryschnieders/Documents/Research/My_Data/rgb_per_region_{base}.npy"
    np.save(out_file, RGB_region_signals, allow_pickle=True)




    # ---------- Raw grayscale per region ----------
    raw_signals = []
    T = len(db[0]['rgb'])
    for reg in range(n_regions):
        tracks = [
            build_gray(np.asarray(tr['rgb'], np.float32))
            for tr in db if tr['region'] == reg
        ]
        raw_signals.append(np.mean(tracks, axis=0) if tracks else np.full(T, np.nan))




    T_trim = len(raw_signals[0])                      # after your `cut`
    win    = max(1, int(0.05 * (T_trim + 2*cut)))        # same 10 % rule
    sigma  = win / 6.0                                   # so ±3σ ~ window
    x      = np.arange(win) - win // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum().astype(np.float32)    # normalise to 1.0

    gw_signals = []
    for sig in raw_signals:
        ok       = ~np.isnan(sig)
        filled   = np.where(ok, sig, 0.0)
        w_sum    = np.convolve(filled, kernel, mode='same')      # weighted sum
        w_norm   = np.convolve(ok.astype(np.float32), kernel, mode='same')  # sum of weights that hit valid samples
        gw_mean  = np.where(w_norm > 0, w_sum / w_norm, np.nan)
        gw_signals.append(gw_mean)
        

    ma_signals = gw_signals  # use the Gaussian-weighted signals as moving average

    # ------------------ plot ------------------------------------------


    ma_freq_est = []
    for s in ma_signals:
        ok = ~np.isnan(s)
        bpm, _ = signal_process_alter(s[ok], fps) if ok.any() else (np.nan, [])
        ma_freq_est.append(bpm)




    fig = plt.figure(figsize=(12, 10))
    for r, (orig, ma) in enumerate(zip(raw_signals, ma_signals)):
        plt.subplot(5, 1, r + 1)
        plt.plot(ma,    lw=1.2, label=f"Region {r+1}  (HR ≈ {ma_freq_est[r]:.1f} BPM)", color='k')
        plt.plot(orig, lw=0.5, label=f"Region {r+1} raw gray", color='C0', alpha=0.7)
        plt.ylabel("Gray")
        plt.legend(); plt.grid(True)
        if r == 4:
            plt.xlabel("Frame (after trim)")
    plt.suptitle(f"Moving-average intensity of raw signal per region (window = {win} frames)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    st.pyplot(fig)  # Display in Streamlit app
    plt.close(fig)  # Close the figure to free memory

    plt.savefig('/Users/henryschnieders/Desktop/movavg_intensity_per_region.png', dpi=300)

    # --- estimate BPM: heart rate from polished signal -----------------
    hr_est = []
    for s in region_signals:
        ok = ~np.isnan(s)
        bpm, _ = signal_process_alter(s[ok], fps) if ok.any() else (np.nan, [])
        hr_est.append(bpm)


    #------ estimate frequency: motion data -----------

    freq_est = []
    for s in motion_signals:
        ok = ~np.isnan(s)
        bpm, _ = signal_process_alter(s[ok], fps, num = 5) if ok.any() else (np.nan, [])
        freq_est.append(bpm)




    # # ---------- Normal pipeline ----------
    # region_signals, motion_signals, id_map = process_tracks_with_phi(db, phi_tol=0.95)

    # ---------- Plot 0: raw grayscale intensity ----------
    fig = plt.figure(figsize=(12, 10))
    for r, sig in enumerate(raw_signals):
        plt.subplot(5, 1, r + 1)
        plt.plot(sig, label=f"Region {r + 1} raw gray")
        plt.ylabel("Gray")
        plt.legend()
        plt.grid(True)
        if r == 4:
            plt.xlabel("Frame")
    plt.suptitle("Raw grayscale (Euclidean‐norm) intensity per region")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    st.pyplot(fig)  # Display in Streamlit app
    plt.close(fig)  # Close the figure to free memory

    plt.savefig('/Users/henryschnieders/Desktop/raw_grayscale_per_region.png', dpi=300)

    # --------- Plot 1: motion ----------
    fig = plt.figure(figsize=(12,10))
    for r, mot in enumerate(motion_signals):
        plt.subplot(5,1,r+1)

        plt.plot(mot, label=f"Region {r+1} motion (est. freq ≈ {freq_est[r]:.1f} BPM)")
        plt.ylabel("|ΔI|")
        plt.legend(); plt.grid(True)
        if r==4: plt.xlabel("Frame")
    plt.suptitle("Motion filtered out of raw signal per region"); plt.tight_layout(rect=[0,0,1,0.96])

    st.pyplot(fig)  # Display in Streamlit app
    plt.close(fig)  # Close the figure to free memory

    plt.savefig('/Users/henryschnieders/Desktop/motion_per_region.png', dpi=300)

    # --------- Plot 2: intensity -------
    fig = plt.figure(figsize=(12,10))
    for r, sig in enumerate(region_signals):
        plt.subplot(5,1,r+1)
        plt.plot(sig, label=f"Region {r+1}  (HR ≈ {hr_est[r]:.1f} BPM)")
        plt.ylabel("Gray")
        plt.legend(); plt.grid(True)
        if r==4: plt.xlabel("Frame")
    plt.suptitle("Filtered signal per region"); plt.tight_layout(rect=[0,0,1,0.96])

    st.pyplot(fig)  # Display in Streamlit app
    plt.close(fig)  # Close the figure to free memory

    plt.savefig('/Users/henryschnieders/Desktop/avg_intensity_per_region.png', dpi=300)

    # lookup example
    who = "x250_y180"
    for r in range(n_regions):
        if who in id_map[r]:
            print(f"{who} is track {id_map[r][who]} in region {r+1}")


    # dirname = os.path.dirname(__file__)
    # out_path = os.path.join(dirname, 'occulsion_processing_figs.png')


    # st.image(out_path, caption="moving-average intensity")

    who = "x250_y180"
    for r in range(n_regions):
        if who in id_map[r]:
            print(f"{who} is track {id_map[r][who]} in region {r+1}")


    # dirname = os.path.dirname(__file__)
    # out_path = os.path.join(dirname, 'occulsion_processing_figs.png')


    # st.image(out_path, caption="moving-average intensity")


    st.markdown('<div id="done"></div>', unsafe_allow_html=True)

    subprocess.run([
        "/opt/homebrew/opt/python@3.12/bin/python3.12",
        "/Users/henryschnieders/Documents/Research/My_work_parts/motion_process/view_frame_brightness.py",
        "/Users/henryschnieders/Documents/Research/My_Data/testagain_6_10_wholeface.npy"
    ])