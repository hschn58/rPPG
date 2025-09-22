import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os
import subprocess
import time


mp4_run='/Users/henryschnieders/Documents/Research/My_work/Face_detection_methods/corresponding_dataset_conversion/temmatch_driver_explicitframes_color_mp4.py'
model_path ='/Users/henryschnieders/documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'


base_dir =  os.path.dirname(os.path.realpath(__file__))

mp4_folder = os.path.join(base_dir+'/Videos', 'mp4')

# Process all AVI and MP4 videos



# Face detection helper function
def find_face(frame, vid_name):
    """
    Detect the face in the given frame using the provided face detector.
    """

    if vid_name=='0_light':
        score_threshold=0.001
    else:
        score_threshold=0.1
    
    detector = cv2.FaceDetectorYN_create(model_path, "", (frame.shape[1], frame.shape[0]), score_threshold=score_threshold)

    # Normalize and convert frame to a 3-channel image if needed
    frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)  # Normalize to 0-255 range
    frame_uint8 = frame_normalized.astype(np.uint8)  # Convert to uint8
    frame_rgb=frame_uint8

    _, faces = detector.detect(frame_rgb)

    if faces is not None:
        return faces[0][:4].astype(int), frame_rgb
    return None, None



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

def denoise(signal_data):
        """Denoise a 1D signal using wavelet transform and filtering."""
        if len(signal_data) < 10:  # 确保信号长度足够
            return signal_data

        fs = 30  # 采样频率
        b, a = signal.butter(5, [1, 2], btype='bandpass', fs=fs)
        filtered_signal = signal.filtfilt(b, a, signal_data)

        # Wavelet decomposition
        coeffs = pywt.wavedec(filtered_signal, 'db4', level=5)
        thr = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(filtered_signal)))
        coeffs_denoised = [pywt.threshold(c, thr, mode='soft') for c in coeffs]
        denoised_signal = pywt.waverec(coeffs_denoised, 'db4')

        return denoised_signal[:len(signal_data)]  # 确保输出长度一致

def process_video_bpm(processed_npy,vid,x,y,w,h):
    """Process video and generate heatmap of variance for denoised signals."""

    vid=vid
    # Initialize a 4D array to hold video frames
    video_array = np.load(processed_npy)

    frame_count, width, height, _ = video_array.shape

    width = min([frame.shape[0] for frame in video_array])
    height = min([frame.shape[1] for frame in video_array])   #ive been using width for first index and height for second index, but this is the opposite of what is done in the original code
    
    x_0=x
    y_0=y

    print(f'width: {width}, height: {height}')
    print(f'x: {x}, y: {y}, w: {w}, h: {h}')

    # Initialize a 2D array to hold variances for each pixel
    variance_map = np.zeros((width, height))
    heartrate_mark = np.zeros((width, height))
    # Process each pixel across all frames
    for y in range(height):
        for x in range(width):
            
            # Extract the pixel values over time
            #(frame number, x, y, RGB)
            pixel_values = video_array[:, x, y, :].mean(axis=1)  # 平均 RGB 值
            denoised_pixel_values = denoise(pixel_values)
            variance_map[x, y] = np.var(denoised_pixel_values)

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
    plt.hist(heartrate_array, bins=30, color='blue', edgecolor='black', zorder=2)
    plt.title('Heart Rate Distribution')
    plt.xlabel('Heart Rate (bpm)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    plt.savefig(base_dir+'/out/'+vid+'_heartrate_distribution.png', dpi=600)
    
    np.save(base_dir+'/out/'+vid+'heartrate.npy', heartrate)
    np.savetxt(base_dir+'/out/'+vid+'_heartrate_distribution.txt', heartrate_array, fmt='%.2f', header='Heart Rate Values (bpm)', comments='')
    
    # Plot the heatmap of average variances

    good_region=heartrate[x_0:(x_0+w), y_0:(y_0+h)]

    nonzero=good_region[good_region!=0]
    ave=np.average(nonzero)
    
    plt.imshow(heartrate, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Variance Heatmap of Denoised Signals")
    plt.axis('off')  # Hide the axis
    plt.clim()
    plt.savefig(base_dir+'/out/'+vid+'_ave_'+f'{round(ave,2)}'+'_heatmap.png', dpi=600)
    


# Function to process each video
# def process_video(video_path, ext):
#     vid = os.path.splitext(os.path.basename(video_path))[0]  # Video name without extension
#     print(f"Processing video: {vid}")

#     # Load the associated data file
#     filename = '/Users/henryschnieders/Documents/Research/My_work/Data/'+vid+'_padding_color_mp4.npy'
#     if not os.path.exists(filename):
#         print(f"Data file for video {vid} not found at {filename}. Skipping.")
#         return

#     time.sleep(5)
#     try:
#         data = np.load(filename, allow_pickle=True)
#     except Exception as e:
#         print(e)
#         data=np.load(filename, allow_pickle=True)

#     # Find the face in the last frame
#     roi, frame_rgb = find_face(data[-1], vid)
    
#     if roi is not None:
#         x, y, w, h = roi

#         # Shrink the face region so that only the face is in the area
#         shrink=0.25
#         x = int(x + 0.5 * w * (1 - (1 - shrink)))
#         y = int(y + 0.5 * h * (1 - (1 - shrink)))
#         w = int(w * (1 - shrink))
#         h = int(h * (1 - shrink))

#         cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         #cv2.imshow('frame', frame_rgb)
#         output_face_path = base_dir+'/out/'+vid+'_averagedfaceregion.png'
#         cv2.imwrite(output_face_path, frame_rgb)
#         cv2.waitKey(2000)
#         cv2.destroyAllWindows()

#     # Create the BPM heatmap and histogram
#     process_video_bpm(filename, vid, x, y, w, h)


def process_video(video_path):
    vid = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing video: {vid}")

    filename = '/Users/henryschnieders/Documents/Research/My_work/Data/' + vid + '_padding_color_mp4.npy'
    
    # Check if the file already exists, skip if it does
    if os.path.exists(filename):
        print(f"File {filename} already exists, skipping subprocess.")
    else:
        print(f"File {filename} not found, creating it.")
        subprocess.run(['/usr/local/bin/python3', mp4_run, '--video_data', video_path, '--vid', vid])

    # Load the data file and process it
    try:
        data = np.load(filename, allow_pickle=True)
    except Exception as e:
        print(e)
        return

    roi, frame_rgb = find_face(data[-1], vid)
    
    if roi is not None:
        x, y, w, h = roi
        shrink = 0.25
        x = int(x + 0.5 * w * (1 - (1 - shrink)))
        y = int(y + 0.5 * h * (1 - (1 - shrink)))
        w = int(w * (1 - shrink))
        h = int(h * (1 - shrink))

        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
        output_face_path = base_dir + '/out/' + vid + '_averagedfaceregion.png'
        cv2.imwrite(output_face_path, frame_rgb)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    process_video_bpm(filename, vid, x, y, w, h)

# def process_videos():

#     # Process MP4 files
#     for mp4_file in os.listdir(mp4_folder):
#         if mp4_file.endswith('.mp4'):
#             input_path = os.path.join(mp4_folder, mp4_file)

#             print(f"Running script for MP4 file: {input_path}")
#             subprocess.run(['/usr/local/bin/python3', mp4_run, '--video_data', input_path, '--vid', mp4_file[:-4]])
#             process_video(input_path)

def process_videos():
    for mp4_file in os.listdir(mp4_folder):
        if mp4_file.endswith('.mp4'):
            input_path = os.path.join(mp4_folder, mp4_file)
            process_video(input_path)


process_videos()