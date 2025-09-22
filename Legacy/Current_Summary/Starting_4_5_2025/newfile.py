import numpy as np
import matplotlib.pyplot as plt



# Load the .npy file
# Replace 'your_file.npy' with the actual path to your .npy file
data = np.load('/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Starting_4_5_2025/Data/test415_545_regions.npy', allow_pickle=True)


# ---------------------------
# Helper Functions
# ---------------------------
def complex_mag(x):
    """Return the magnitude of complex value(s)."""
    return np.abs(x)

def signal_process_alter(pix_intensity, fps):
    """
    Process the pixel intensity signal via FFT bandpass filtering between 0.5 Hz and 3.0 Hz
    to estimate the heart rate.
    
    Parameters:
      pix_intensity (1D array): Averaged pixel intensity signal.
      fps (float): Frames per second.
      
    Returns:
      average (float): Estimated heart rate (BPM).
      top5freqs (1D array): Top 5 frequency components (Hz).
    """
    sampling_rate = fps
    lowcut_heart = 0.5
    highcut_heart = 4.0

    N = len(pix_intensity)  # Signal length
    T = 1.0 / sampling_rate  # Sampling interval
    coefs = np.fft.fft(pix_intensity)  # FFT of the signal
    freqs = np.fft.fftfreq(N, T)       # Frequency bins

    # Select frequencies within the heart rate band.
    indices = np.where((np.abs(freqs) >= lowcut_heart) & (np.abs(freqs) <= highcut_heart))
    coefsinrange = coefs[indices]
    freqsinrange = freqs[indices]

    # Select top 5 frequency components based on FFT coefficient magnitude.
    max_indicies = np.argsort(complex_mag(coefsinrange))[-5:]
    coefsinrange = coefsinrange[max_indicies]
    freqsinrange = freqsinrange[max_indicies]

    # Compute a weighted average frequency (Hz) then convert to BPM.
    FFT_heart_rate = np.average(np.abs(freqsinrange), weights=complex_mag(coefsinrange))
    average = FFT_heart_rate * 60

    top5indices = np.argsort(complex_mag(coefsinrange))[-5:]
    top5freqs = freqsinrange[top5indices]

    return average, top5freqs

def motion_process_5regions(data, window_size=10, phi_tol=0.8):
    """
    Process motion for face regions when the input data is organized as:
         data[frame][region]
    For example, if there are T frames and 5 regions per frame, then data is a list of T elements,
    each being a list of 5 images (each image shape: (H, W, 3)).
    
    The function performs temporal normalization using a sliding window, computes frame-to-frame
    differences, and then evaluates a phi value between consecutive difference frames. A phi mask
    (bitmask) is created for each region (for valid frames only) such that a pixel is marked good (1)
    if the phi value is ≥ phi_tol; otherwise 0.
    
    Returns:
      motion_data: A list of length T, where each element is a list of 5 arrays (one per region)
                   of shape (H, W, 3) representing the motion-filtered images.
      phi_bitmask: A list of length T, where each element is a list of 5 phi bitmask arrays (or None)
                   for each region. (A phi mask is computed for frames 1 to T-2; frames 0 and T-1 are None.)
      phi_tol:     The phi tolerance value used.
    """
    T = len(data)
    num_regions = len(data[0])
    
    # ------------------------------------------------
    # Reorganize data per region: each region as a time series.
    # ------------------------------------------------
    regions = []
    for r in range(num_regions):
        # For each region, collect the same region across all frames.
        region_frames = [data[t][r] for t in range(T)]
        regions.append(np.stack(region_frames, axis=0))  # shape (T, H, W, 3)
    
    motion_data_regions = []  # to store motion per region (shape: (T, H, W, 3))
    region_masks_all = []     # to store phi masks for each region (list length T-2 per region)
    
    for region in regions:
        R = region.copy().astype(np.float32)
        # -------------------------------
        # Temporal normalization using a sliding window.
        # -------------------------------
        for t in range(T):
            start = max(0, t - window_size + 1)
            end = t + 1  # window covers frames start...t
            window = R[start:end]
            mean_val = window.mean(axis=(0, 1, 2), keepdims=True)
            std_val = window.std(axis=(0, 1, 2), keepdims=True) + 1e-8
            R[t] = (R[t] - mean_val) / std_val
        
        # -------------------------------
        # Compute motion differences (frame-to-frame differences)
        # -------------------------------
        motion = np.zeros_like(R)
        for t in range(T - 1):
            motion[t] = R[t+1] - R[t]
        # The last frame remains zeros.
        
        # -------------------------------
        # Compute phi masks based on consecutive motion difference frames.
        # The computed phi for a pixel in frame t+1 comes from motion[t] and motion[t+1].
        # We have valid phi values for frames 1 to T-2.
        # -------------------------------
        masks = []
        for t in range(T - 2):
            C_t = motion[t]
            C_t1 = motion[t+1]
            norm_C_t = np.linalg.norm(C_t, axis=-1, keepdims=True) + 1e-8
            norm_C_t1 = np.linalg.norm(C_t1, axis=-1, keepdims=True) + 1e-8
            C_t_normalized = C_t / norm_C_t
            C_t1_normalized = C_t1 / norm_C_t1
            phi = np.sum(C_t_normalized * C_t1_normalized, axis=-1)  # shape: (H, W)
            mask = (phi <= phi_tol).astype(np.uint8)
            masks.append(mask)
            
            # Apply the bitmask to the motion difference in frame t+1.
            expanded_mask = np.expand_dims(mask, axis=-1)
            motion[t+1] = motion[t+1] * expanded_mask
        
        motion_data_regions.append(motion)
        region_masks_all.append(masks)
    
    # ------------------------------------------------
    # Convert per-region outputs back to frame-first indexing.
    # ------------------------------------------------
    # motion_data: list with length T; each element is a list of 5 regions' motion data.
    motion_data = []
    for t in range(T):
        frame_motion = []
        for r in range(num_regions):
            frame_motion.append(motion_data_regions[r][t])
        motion_data.append(frame_motion)
    
    # phi_bitmask: create a list of length T where each element is a list of 5 phi masks.
    # Valid phi masks are available for frames 1 to T-2. For frames 0 and T-1, store None.
    phi_bitmask = []
    for t in range(T):
        frame_masks = []
        for r in range(num_regions):
            if 1 <= t < T - 1:
                # The mask for frame t is stored at index t-1 in the region's mask list.
                frame_masks.append(region_masks_all[r][t - 1])
            else:
                frame_masks.append(None)
        phi_bitmask.append(frame_masks)
    
    return motion_data, phi_bitmask, phi_tol

# ---------------------------
# Driver Code
# ---------------------------
def main():
    # --- Simulation parameters ---
    fps = 30            # frames per second
    T = len(data)      # total number of frames in each video region
    num_regions = 5     # e.g., 5 face regions per frame


    # --- Compute motion data and phi bitmask ---
    motion_data, phi_bitmask, used_phi_tol = motion_process_5regions(data, window_size=10, phi_tol=0.9)
    
    # --- Build the average pixel intensity signal per face region ---
    # For each frame with a valid phi mask (frames 1 to T-2), compute the average pixel intensity
    # over the "good" pixels (those with mask value 1) for each region.
    region_signals = [[] for _ in range(num_regions)]
    
    for t in range(T):
        for r in range(num_regions):
            mask = phi_bitmask[t][r]
            if mask is not None:
                # Get the corresponding original frame for face region r.
                frame = data[t][r]
                # Convert to grayscale by averaging the three channels.
                gray_frame = np.mean(frame, axis=-1)
                # Select only the "good" pixels (mask == 1).
                good_pixels = gray_frame[mask == 1]
                if good_pixels.size > 0:
                    avg_intensity = np.mean(good_pixels)
                else:
                    avg_intensity = 0
                region_signals[r].append(avg_intensity)
            else:
                # For frames without a valid phi mask, store NaN.
                region_signals[r].append(np.nan)
    
    # Convert each region's signal to a NumPy array.
    region_signals = [np.array(signal) for signal in region_signals]
    
    # --- Process the intensity signals to compute heart rate per region ---
    region_heart_rates = []
    for r in range(num_regions):
        # Remove NaN values from the signal.
        valid_signal = region_signals[r][~np.isnan(region_signals[r])]
        if len(valid_signal) > 0:
            hr, top5freqs = signal_process_alter(valid_signal, fps)
        else:
            hr, top5freqs = 0, np.array([])
        region_heart_rates.append(hr)
        print(f"Region {r+1}: Estimated Heart Rate = {hr:.2f} BPM, Top Frequencies (Hz) = {top5freqs}")
    
    # --- Plot the heart rate (average intensity) signal for each region ---
    plt.figure(figsize=(12, 10))
    for r in range(num_regions):
        plt.subplot(num_regions, 1, r+1)
        plt.plot(region_signals[r], label=f"Region {r+1} (HR: {region_heart_rates[r]:.1f} BPM)")
        plt.ylabel("Avg Intensity")
        plt.legend(loc="upper right")
        if r == num_regions - 1:
            plt.xlabel("Frame Index")
    plt.suptitle("Heart Rate Signal (Average Pixel Intensity) per Face Region")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    main()
