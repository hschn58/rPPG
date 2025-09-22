#!/usr/bin/env python3
# ---------------------------------------------------------------
#  POS pulse extraction on per-region RGB time-series
# ---------------------------------------------------------------
import numpy as np, matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, detrend
from matplotlib.backends.backend_pdf import PdfPages
import argparse, os




# ---------------- POS helper -----------------------------------

def get_params_from_cli_or_prompt():
    """
    • First try to grab --foo and --bar from the command line.
    • If the user didn't provide them, fall back to asking interactively.
    """
    p = argparse.ArgumentParser(description="get data file for pixel trajectories")
    p.add_argument("--data_path", type = str, default = None, help = 'pixel trajectories file path (.npz)' )               # default is None
    args = p.parse_args()

    if args.data_path is None:
        args.data_path = input("Enter the path to the .npz trajectories file:")

    return args.data_path


def pos_trace(R, G, B, fs, win_len=1.6, overlap=0.5):
    """
    Return full-length POS pulse trace, BPM and SNR.
    """
    L = int(round(win_len * fs))
    H = int(round(L * overlap))
    N = len(R)

    # pre-allocate reconstruction
    pulse = np.zeros(N);  wsum = np.zeros(N)

    for start in range(0, N-L+1, H):
        stop = start + L
        # normalise RGB in this window
        r = R[start:stop] / np.nanmean(R[start:stop])
        g = G[start:stop] / np.nanmean(G[start:stop])
        b = B[start:stop] / np.nanmean(B[start:stop])

        # two orthogonal chrominance signals
        X =  3*r - 2*g
        Y =  1.5*r + g - 1.5*b
        alpha = np.nanstd(X) / (np.nanstd(Y) + 1e-8)
        S = X - alpha * Y        # POS pulse in this window

        # Hann window for overlap-add
        w = np.hanning(L)
        pulse[start:stop] += S * w
        wsum[start:stop]  += w

    pulse /= (wsum + 1e-12)       # normalise overlap

    # band-pass for HR/SNR
    b, a = butter(3, [BAND[0]/(fs/2), BAND[1]/(fs/2)], btype='band')
    bp   = filtfilt(b, a, detrend(pulse))

    f,P = welch(bp, fs=fs, nperseg=min(512, len(bp)))
    mask = (f>=BAND[0])&(f<=BAND[1])
    if not np.any(mask):
        return pulse, np.nan, 0.0
    fpk  = f[mask][np.argmax(P[mask])]
    bpm  = fpk * 60
    snr  = P[(f>=fpk-0.2)&(f<=fpk+0.2)].sum() / (P[mask].sum()+1e-9)
    return bp, bpm, snr

# ---------------- run POS on every region ----------------------
if __name__ == "__main__":


    # ---------------- user paths -----------------------------------
    NPZ_FILE = get_params_from_cli_or_prompt()  # path to pixel trajectories file (.npz)

    CUT_FRAMES   = 25      # discard frames at start & end
    FPS          = 30
    REGIONS      = 5
    BAND         = (0.7, 4.0)          # 42–240 BPM

    # ---------------- load & rebuild RGB per region ----------------
    traj = np.load(NPZ_FILE, allow_pickle=True)['trajectories'].tolist()
    TOT = len(traj[0]['rgb'])

    RGB = []
    for reg in range(REGIONS):
        tracks = [tr for tr in traj if tr['region'] == reg]
        if not tracks:
            RGB.append((np.full(TOT-2*CUT_FRAMES, np.nan),)*3)
            continue
        cube = np.stack([tr['rgb'] for tr in tracks])       # P×T×3
        R = cube[...,0].mean(0)[CUT_FRAMES:-CUT_FRAMES]
        G = cube[...,1].mean(0)[CUT_FRAMES:-CUT_FRAMES]
        B = cube[...,2].mean(0)[CUT_FRAMES:-CUT_FRAMES]
        RGB.append((R, G, B))

    pos_tr , pos_bpm , pos_snr = [], [], []
    for R,G,B in RGB:
        if np.all(np.isnan(R)):
            pos_tr.append(np.full_like(R, np.nan))
            pos_bpm.append(np.nan); pos_snr.append(0.0); continue
        tr,bpm,snr = pos_trace(R,G,B, fs=FPS)
        pos_tr.append(tr); pos_bpm.append(bpm); pos_snr.append(snr)

    # ---------------- plot -----------------------------------------
    fig1 = plt.figure(figsize=(12,10))

    fig1.suptitle("POS pulse extraction per region (no motion process)", fontsize=16)

    for r,(tr,bpm,snr) in enumerate(zip(pos_tr,pos_bpm,pos_snr)):
        plt.subplot(5,1,r+1)
        plt.plot(tr, lw=1, label = f"Region {r+1}: HR≈{bpm:.1f} BPM  SNR={snr:.1f}")
        plt.ylabel("POS\nsignal")
        plt.grid(True)
        plt.legend()
        if r==4: plt.xlabel("Frame")



    figs = [fig1]

    base = os.path.basename(NPZ_FILE).split('.')[0].replace('pixel_trajectories_','') + '_POS_nomotionprocess.pdf'

    pdf_path = f'/Users/henryschnieders/Documents/Research/My_Data/{base}'

    with PdfPages(pdf_path) as pdf:
        for f in figs:
            pdf.savefig(f)
            plt.close(f)
    print(pdf_path)   # so the driver can capture it