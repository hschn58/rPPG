# ------------------------------------------------------------------
#  extra block – CHROM pulse estimate per region
# ------------------------------------------------------------------
from scipy.signal import butter, filtfilt, welch
import numpy as np
import matplotlib.pyplot as plt
import argparse, os
from matplotlib.backends.backend_pdf import PdfPages



# # --- organise by region -------------------------------------------------
# region_tracks = [[] for _ in range(n_regions)]        # list of signals
def get_params_from_cli_or_prompt():
    """
    • First try to grab --foo and --bar from the command line.
    • If the user didn't provide them, fall back to asking interactively.
    """
    p = argparse.ArgumentParser(description="Signal process motion-processed pixel trajectories.")
    p.add_argument("--motion_processed_paths", type = str, help = 'motion-processed pixel trajectories file path' )               # default is None
    args = p.parse_args()

    if args.motion_processed_paths is None:
        args.motion_processsed_paths = input("Enter the path to the pixel trajectories file:")

    return args.motion_processed_paths

def chrom_bpm(R, G, B, fps, f_lo=0.7, f_hi=4.0, snr_bw=0.1):
    """
    CHROM pulse estimation with SNR.
    Returns (bpm_estimate, snr_dB, pulse_trace).
      • snr_dB uses the power ratio within ±snr_bw Hz of the peak (and its 2× harmonic)
        versus the remaining band-pass power, in dB.
    """
    # ---------- detrend (1.6-s moving mean) ----------
    win = max(1, int(round(1.6 * fps)))
    kernel = np.ones(win) / win
    def detrend(x):
        ok = ~np.isnan(x)
        local  = np.convolve(np.where(ok, x, 0.0), kernel, 'same')
        counts = np.convolve(ok.astype(float),        kernel, 'same')
        return np.where(counts > 0, x - local / counts, 0.0)
    r, g, b = map(detrend, (R, G, B))

    # ---------- chrominance projection ----------
    Sx, Sy = 3*r - 2*g, 1.5*r + g - 1.5*b
    alpha  = np.nanstd(Sx) / (np.nanstd(Sy) + 1e-8)
    X      = Sx - alpha*Sy

    # ---------- 0.7–4 Hz band-pass ----------
    b_f, a_f = butter(3, [f_lo/(fps/2), f_hi/(fps/2)], btype='band')
    Xf       = filtfilt(b_f, a_f, X)

    # ---------- Welch spectrum & peak ----------
    f, Pxx = welch(Xf, fs=fps, nperseg=min(512, len(Xf)))
    band    = (f >= f_lo) & (f <= f_hi)
    if not band.any():
        return np.nan, np.nan, Xf
    f_band, P_band = f[band], Pxx[band]
    f_peak = f_band[np.argmax(P_band)]
    bpm    = f_peak * 60.0

    # ---------- simple SNR (±snr_bw Hz around 1× & 2× peaks) ----------
    sig_mask = (np.abs(f_band - f_peak)     <= snr_bw) | \
               (np.abs(f_band - 2.0*f_peak) <= snr_bw)
    P_sig  = P_band[sig_mask].sum()
    P_noise = P_band[~sig_mask].sum() + 1e-12
    snr_db = 10.0 * np.log10(P_sig / P_noise)

    return bpm, snr_db, Xf

def build_gray(rgb_arr):
    """
    Return per-frame Euclidean norm ‖(R,G,B)‖₂.

    Parameters
    ----------
    rgb_arr : ndarray shape (T, 3)  uint8 / float
        One RGB sample per frame.

    Returns
    -------
    gray_like : ndarray shape (T,)  float32
        Intensity proxy √(R²+G²+B²) for every frame.
    """
    rgb_arr = rgb_arr.astype(np.float32)
    return np.linalg.norm(rgb_arr, axis=1).astype(np.float32)


if __name__ == "__main__":



    cut = 25
    fps = 30
    n_regions = 5

    #load motion-processed trajectories

    data_path = get_params_from_cli_or_prompt() #rgb_per_region_{base}.npy

    RGB_arr = np.load(data_path, allow_pickle=True)
    RGB_arr = RGB_arr[:, cut:-cut, :]
    n_regions , T_trim, _ = RGB_arr.shape

    # ------------------------------------------------------------------
    # 2.  Run CHROM per region (now capturing SNR)
    # ------------------------------------------------------------------
    chrom_bpm_est, chrom_snr_est, chrom_traces = [], [], []

    for reg in range(n_regions):
        R, G, B = RGB_arr[reg].T          # (T_trim,) each
        bpm, snr_db, tr = chrom_bpm(R, G, B, fps)
        chrom_bpm_est.append(bpm)
        chrom_snr_est.append(snr_db)
        chrom_traces.append(tr)
    

    # ------------------ plot -----------------------------------------
    fig1 = plt.figure(figsize=(12, 10))

    fig1.suptitle("CHROM with motion-processed signal", fontsize=18, y=0.95)

    for r, (tr, bpm, snr_db) in enumerate(zip(chrom_traces,
                                            chrom_bpm_est,
                                            chrom_snr_est)):
        plt.subplot(n_regions, 1, r + 1)
        plt.plot(tr, lw=0.9,
                label=f"Region {r+1}: HR ≈ {bpm:5.1f} BPM | SNR ≈ {snr_db:4.1f} dB")
        plt.ylabel("CHROM"); plt.grid(True)
        if r == n_regions - 1:
            plt.xlabel("Frame")
        plt.legend()

   

    fig2 = plt.figure(figsize=(12, 10))

    fig2.suptitle("Reconstructed signal from CHROM input", fontsize=18, y=0.95)
    
    for r in range(n_regions):
        plt.subplot(n_regions, 1, r + 1)
        plt.plot(build_gray(RGB_arr[r]), label=f'Region {r + 1} gray signal')

        plt.ylabel('Intensity')
        plt.grid(True)
        if r == n_regions - 1:
            plt.xlabel('Frame')
        plt.legend()


    figs = [fig1, fig2]

    base = os.path.basename(data_path).split('.')[0].replace('rgb_per_region_','') + '_CHROM_wmotionprocess.pdf'

    pdf_path = f'/Users/henryschnieders/Documents/Research/My_Data/{base}'

    with PdfPages(pdf_path) as pdf:
        for f in figs:
            pdf.savefig(f)
            plt.close(f)
    print(pdf_path)   # so the driver can capture it

    
