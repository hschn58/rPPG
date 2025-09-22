import numpy as np
import scipy.signal as signal
import pywt
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.signal as signal
import pywt

def denoise_heartrate(roi_mean):
    try:
        B1 = roi_mean
        len_b1 = len(B1)
        end_ind = len_b1 - 10
        A1 = B1[10:end_ind]  # Trimming edges as per original logic
        fs = 51  # Sampling frequency
        found = False
        denoised_data = np.zeros(end_ind - 10)
        peak = None
        peak_locs = None

        # Define ranges for parameter tuning
        winwithd_range = np.arange(0.01, 0.32, 0.1)
        max_butter_range = np.arange(3.1, 2.5, -0.1)
        mindis_range = np.arange(0.3, 0.9, 0.1)
        min_width_range = np.arange(0, 11, 2)

        # Parameter tuning to find the best configuration
        for th in range(5, 31, 5):
            if found:
                break
            for winwithd in winwithd_range:
                if found:
                    break
                for max_butter in max_butter_range:
                    if found:
                        break
                    for mindis in mindis_range:
                        if found:
                            break
                        for min_width in min_width_range:
                            if found:
                                break

                            # Apply bandpass filter
                            b, a = signal.butter(5, [1, max_butter], btype='bandpass', fs=fs)
                            filtered_ppg = signal.filtfilt(b, a, A1)

                            # Wavelet denoising
                            c = pywt.wavedec(filtered_ppg, 'db4', level=5)
                            thr = np.median(np.abs(c[-1])) / 0.6745 * np.sqrt(2 * np.log(len(filtered_ppg)))
                            c_denoised = [pywt.threshold(ci, thr, mode='soft') for ci in c]
                            wavelet_denoised = pywt.waverec(c_denoised, 'db4')

                            # Smoothing with a moving average
                            window_size = round(winwithd * fs)
                            denoised_ppg = np.convolve(wavelet_denoised, np.ones(window_size) / window_size, mode='same')

                            # Peak detection
                            peaks, _ = signal.find_peaks(denoised_ppg, distance=int(round(mindis * fs)))
                            peak_widths = signal.peak_widths(denoised_ppg, peaks)[0]
                            peak_threshold = -0.1 * np.max(denoised_ppg[peaks])
                            valid_peak_idx = (denoised_ppg[peaks] > peak_threshold) & (peak_widths >= min_width) & (
                                    peak_widths <= 100)
                            valid_peaks = denoised_ppg[peaks][valid_peak_idx]
                            valid_peak_locs = peaks[valid_peak_idx]

                            # Variance checks
                            best_variance = np.inf
                            if len(valid_peak_locs) > 2:
                                for j1 in range(len(valid_peak_locs) - 2):
                                    group1 = [valid_peak_locs[j1]]
                                    group2 = [valid_peak_locs[j1 + 1]]
                                    group3 = [valid_peak_locs[j1 + 2]]

                                    variance_group1 = np.var(group1)
                                    variance_group2 = np.var(group2)
                                    variance_group3 = np.var(group3)
                                    mean_group1 = np.mean(group1)
                                    mean_group2 = np.mean(group2)
                                    mean_group3 = np.mean(group3)
                                    gap_2 = mean_group3 - mean_group2
                                    gap_1 = mean_group2 - mean_group1
                                    total_variance = variance_group1 + variance_group2 + variance_group3

                                    if (variance_group1 < th and variance_group2 < th and
                                            variance_group3 < th and total_variance < best_variance and
                                            abs(gap_2 - gap_1) < 7):
                                        best_variance = total_variance
                                        best_group1 = group1
                                        best_group2 = group2
                                        best_group3 = group3
                                        best_params = [winwithd, max_butter, mindis, min_width]
                                        best_th = th
                                        found = True

        if found:
            heartrate = 60 * fs / ((np.mean(best_group3) - np.mean(best_group1)) / 2)
            if heartrate > 145:
                heartrate = heartrate / 2
        else:
            heartrate = 0
    except Exception as e:
        heartrate = 0
        print(e)

    return heartrate

def process_video(processed_npy):
    """Process video and generate heatmap of variance for denoised signals."""
    video_array = np.load(processed_npy)
    print(video_array[0].shape)

    frame_count, width, height, _ = video_array.shape
    

    # Initialize a 2D array to hold variances for each pixel
    variance_map = np.load(output_npy)
    heartrate = np.zeros((width, height))
    heartrate_list = []
    # Process each pixel across all frames
    for y in range(height):
        for x in range(width):
 
            # Extract the pixel values over time
            pixel_values = video_array[:, x, y, :].mean(axis=1)  # 平均 RGB 值
            if variance_map[x, y] > 0.3 and variance_map[x, y] < 1.5:
                heartrate[x, y] = denoise_heartrate(pixel_values)
                if heartrate[x, y] > 100:
                    heartrate[x, y] = 0
                else:
                    heartrate_list.append(heartrate[x, y])
            else:
                heartrate[x, y] = 0

    heartrate_array = np.array(heartrate_list)

    plt.figure(figsize=(10, 6))
    plt.hist(heartrate_array, bins=30, color='blue', alpha=0.7, edgecolor='black')
    plt.title('Heart Rate Distribution')
    plt.xlabel('Heart Rate (bpm)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    plt.show()
    np.save(heartrate_npy, heartrate)
    np.savetxt(heartrate_distribution, heartrate_array, fmt='%.2f', header='Heart Rate Values (bpm)', comments='')
    print(variance_map.shape)
    # Plot the heatmap of average variances
    plt.imshow(heartrate, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Variance Heatmap of Denoised Signals")
    plt.axis('off')  # Hide the axis
    plt.clim()
    plt.show()


video_file = '/Users/henryschnieders/Documents/Research/My_work/Data/100_light'+'_padding_color_mp4.npy'

output_npy = '/Users/henryschnieders/downloads/test.npy'

heartrate_npy = '/Users/henryschnieders/downloads/heartrate_test.npy'

heartrate_distribution = '/Users/henryschnieders/downloads/heartrate_dis.txt'

process_video(video_file)

