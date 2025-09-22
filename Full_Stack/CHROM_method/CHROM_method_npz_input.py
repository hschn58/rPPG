# ------------------------------------------------------------------
#  extra block – CHROM pulse estimate per region
# ------------------------------------------------------------------
from scipy.signal import butter, filtfilt, welch
import numpy as np
import matplotlib.pyplot as plt

#cli inputs
import argparse, os

#merge pdfs
from matplotlib.backends.backend_pdf import PdfPages


def get_params_from_cli_or_prompt():
    """
    • First try to grab --foo and --bar from the command line.
    • If the user didn't provide them, fall back to asking interactively.
    """
    p = argparse.ArgumentParser(description="get data file for pixel trajectories")
    p.add_argument("--data_path", type = str, help = 'pixel trajectories file path (.npz)' )               # default is None
    args = p.parse_args()

    if args.data_path is None:
        args.data_path = input("Enter the path to the .npz trajectories file:")

    return args.data_path

# # --- organise by region -------------------------------------------------
# region_tracks = [[] for _ in range(n_regions)]        # list of signals

def chrom_bpm(R, G, B, fs, f_lo=0.7, f_hi=4.0):
    """Return (bpm, snr, Xf) where snr is power-ratio around the pulse peak."""
    win = max(1, int(round(1.6 * fs)))
    k   = np.ones(win) / win

    def detrend(v):
        ok = ~np.isnan(v)
        filled = np.where(ok, v, 0.0)
        local  = np.convolve(filled, k, 'same')
        counts = np.convolve(ok.astype(float), k, 'same')
        return np.where(counts > 0, v - local / counts, 0.0)

    r,g,b = map(detrend, (R,G,B))

    Sx = 3*r - 2*g
    Sy = 1.5*r + g - 1.5*b
    alpha = np.nanstd(Sx) / (np.nanstd(Sy) + 1e-8)
    X = Sx - alpha * Sy

    b, a = butter(3, [f_lo/(fs/2), f_hi/(fs/2)], btype='band')
    Xf   = filtfilt(b, a, X)

    f, P = welch(Xf, fs=fs, nperseg=min(512, len(Xf)))
    band = (f >= f_lo) & (f <= f_hi)
    if not np.any(band):
        return np.nan, 0.0, Xf            # no valid band → NaNs

    f_peak = f[band][np.argmax(P[band])]
    bpm    = f_peak * 60.0

    mask_peak = (f >= f_peak-0.2) & (f <= f_peak+0.2)
    snr = P[mask_peak].sum() / (P[band].sum() - P[mask_peak].sum() + 1e-9)

    return bpm, snr, Xf

# ---------- (re)build per-region RGB signals with the φ-mask ----------
# RGB_signals = []   # list of (R,G,B) tuples per region
# for reg in range(n_regions):
#     P = len(region_tracks[reg])
#     if P == 0:
#         RGB_signals.append((np.full_like(region_signals[0], np.nan),)*3)
#         continue

#     # G  : P×T  already exists  (greyscale), but we need R,G,B per track
#     Rm = []; Gm = []; Bm = []
#     MS = np.stack([trk['mask'] for trk in region_tracks[reg]])   # P×(T-2)
#     for trk in region_tracks[reg]:
#         rgb = trk['rgb'].T    # shape 3×T
#         Rm.append(rgb[0]); Gm.append(rgb[1]); Bm.append(rgb[2])
#     Rm, Gm, Bm = map(np.vstack, (Rm, Gm, Bm))   # each: P×T

#     # average with same NaN/φ logic used for greyscale
#     def masked_mean(M):
#         sig = np.zeros(T, np.float32)
#         sig[:]=np.nan
#         sig[0]  = np.nanmean(M[:,0])
#         sig[-1] = np.nanmean(M[:,-1])
#         for t in range(1, T-1):
#             good = MS[:,t-1].astype(bool)
#             sig[t] = np.nanmean(M[good,t]) if good.any() else np.nan
#         return sig[cut:-cut]      # apply same trimming

#     RGB_signals.append(tuple(map(masked_mean, (Rm, Gm, Bm))))

# ---------- rebuild RGB_signals directly ---------------------------

if __name__ == "__main__":

    cut = 25
    fps = 30
    n_regions = 5
    RGB_signals = []
    
    data_path = get_params_from_cli_or_prompt()
    
    trajectories_db = np.load(data_path, allow_pickle=True)['trajectories'].tolist()

    n_frames  = len(trajectories_db[0]['rgb'])            # all tracks full-length
            

    for reg in range(n_regions):
        tracks = [tr for tr in trajectories_db if tr['region']==reg]
        if not tracks:
            RGB_signals.append((np.full(n_frames-2*cut, np.nan),)*3)
            continue

        # stack: P×T×3
        cube = np.stack([tr['rgb'] for tr in tracks])       # P×T×3
        # simple mean across tracks (no φ-mask here)
        R = cube[...,0].mean(axis=0)[cut:-cut]
        G = cube[...,1].mean(axis=0)[cut:-cut]
        B = cube[...,2].mean(axis=0)[cut:-cut]
        RGB_signals.append((R, G, B))

    chrom_bpm_est, chrom_snr_est, chrom_traces = [], [], []

    for R, G, B in RGB_signals:
        bpm, snr, trace = chrom_bpm(R, G, B, fps)
        chrom_bpm_est.append(bpm)
        chrom_snr_est.append(snr)
        chrom_traces.append(trace)

    fig1 = plt.figure(figsize=(12,10))

    fig1.suptitle("CHROM no motion process", fontsize=18, y=0.95)

    for r, (tr, bpm, snr) in enumerate(zip(chrom_traces, chrom_bpm_est, chrom_snr_est)):
        plt.subplot(5,1,r+1)
        plt.plot(tr, lw=0.9, label = f"Region {r+1}  (HR≈{bpm:.1f} BPM,  SNR={snr:.1f})")
        plt.ylabel("CHROM\nsignal")
        plt.grid(True)
        if r == 4:
            plt.xlabel("Frame")
        plt.legend()

    plt.savefig('/Users/henryschnieders/Desktop/chrom_signal_per_region2.png', dpi=300)


    figs = [fig1]

    base = os.path.basename(data_path).split('.')[0].replace('rgb_per_region_','') + '_CHROM_nomotionprocess.pdf'

    pdf_path = f'/Users/henryschnieders/Documents/Research/My_Data/{base}'

    with PdfPages(pdf_path) as pdf:
        for f in figs:
            pdf.savefig(f)
            plt.close(f)
    print(pdf_path)   # so the driver can capture it

