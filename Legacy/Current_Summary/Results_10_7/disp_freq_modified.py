import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from denoise import denoise  # Assuming this is your custom function
import os
import subprocess
import time



#path for scripts to convert the avi, mp4 fils to numpy arrays 
avi_run='/Users/henryschnieders/Documents/Research/My_work/Face_detection_methods/corresponding_dataset_conversion/temmatch_driver_explicitframes_avi.py'
mp4_run='/Users/henryschnieders/Documents/Research/My_work/Face_detection_methods/corresponding_dataset_conversion/temmatch_driver_explicitframes_mp4.py'


base_dir = '/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Results_10_7/Videos'

model_path ='/Users/henryschnieders/documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'



# Folder setup

avi_folder = os.path.join(base_dir, 'avi')
mp4_folder = os.path.join(base_dir, 'mp4')

# Process all AVI and MP4 videos
def process_videos():
    # Process AVI files
    for avi_file in os.listdir(avi_folder):
        if avi_file.endswith('.avi'):
            input_path = os.path.join(avi_folder, avi_file)
            print(f"Running script for AVI file: {input_path}")

             #check if file exists
            subprocess.run(['/usr/local/bin/python3', avi_run, '--video_data', input_path, '--vid', avi_file[:-4]])
            process_video(input_path, '.avi')

    # Process MP4 files
    for mp4_file in os.listdir(mp4_folder):
        if mp4_file.endswith('.mp4'):
            input_path = os.path.join(mp4_folder, mp4_file)

            print(f"Running script for MP4 file: {input_path}")
            subprocess.run(['/usr/local/bin/python3', mp4_run, '--video_data', input_path, '--vid', mp4_file[:-4]])
            process_video(input_path, '.mp4')





# Function to process each video
def process_video(video_path, ext):
    vid = os.path.splitext(os.path.basename(video_path))[0]  # Video name without extension
    print(f"Processing video: {vid}")

    # Load the associated data file
    filename = f'/Users/henryschnieders/Documents/Research/My_work/Data/{vid}_padding'+'_'+ext[1:]+ '.npy'
    if not os.path.exists(filename):
        print(f"Data file for video {vid} not found at {filename}. Skipping.")
        return

    time.sleep(5)
    try:
        data = np.load(filename, allow_pickle=True)
    except Exception as e:
        print(e)
        data=np.load(filename, allow_pickle=True)

    # Determine minimum dimensions of the frames
    X_DIM = min([frame.shape[0] for frame in data])
    Y_DIM = min([frame.shape[1] for frame in data])

    bpm_array = np.zeros((X_DIM, Y_DIM), dtype=np.float32)

    for i in range(X_DIM):
        for j in range(Y_DIM):
            # Extract amplitudes as an array of intensities over time for the (i, j) pixel
            amplitudes = np.array([frame[i, j] for frame in data])
            bpm_array[i, j] = denoise(amplitudes)

        print(f'Finished column {i} of {X_DIM}')

    # Find the face in the last frame
    roi, frame_rgb = find_face(data[-1], vid)
    
    if roi is not None:
        x, y, w, h = roi

        # Shrink the face region so that only the face is in the area
        shrink=0.2
        x = int(x + 0.5 * w * (1 - (1 - shrink)))
        y = int(y + 0.5 * h * (1 - (1 - shrink)))
        w = int(w * (1 - shrink))
        h = int(h * (1 - shrink))

        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

        #cv2.imshow('frame', frame_rgb)
        output_face_path = f'/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Results_10_7/out/{vid}'+ext[1:]+'_averagedfaceregion.png'
        cv2.imwrite(output_face_path, frame_rgb)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    # Create the BPM heatmap and histogram
    create_bpm_visualizations(bpm_array, vid, x, y, w, h, ext)

# Face detection helper function
def find_face(frame, vid_name):
    """
    Detect the face in the given frame using the provided face detector.
    """

    if vid_name=='Dark':
        score_threshold=0.02
    else:
        score_threshold=0.1
    
    detector = cv2.FaceDetectorYN_create(model_path, "", (frame.shape[1], frame.shape[0]), score_threshold=score_threshold)

    # Normalize and convert frame to a 3-channel image if needed
    frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)  # Normalize to 0-255 range
    frame_uint8 = frame_normalized.astype(np.uint8)  # Convert to uint8
    frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)

    _, faces = detector.detect(frame_rgb)

    if faces is not None:
        return faces[0][:4].astype(int), frame_rgb
    return None, None

# Create heatmap and histogram visualizations
from matplotlib.colors import TwoSlopeNorm
def create_bpm_visualizations(bpm_array, vid, x, y, w, h, ext, TwoSlopeNorm=TwoSlopeNorm):
    # Normalize BPM values
    max_val = np.max(bpm_array)
    min_val = np.min(bpm_array)
    vcenter = (max_val + min_val) / 2

    # Use TwoSlopeNorm to set the center for heatmap
    norm = TwoSlopeNorm(vmin=min_val, vcenter=vcenter, vmax=max_val)

    # Average BPM within the ROI
    ave = np.average(bpm_array[x:x + w, y:y + h])

    # Create the heatmap
    fig = plt.figure(figsize=(5, 5))
    plt.title(f'{vid} - {ave:.2f} avg bpm')
    plt.imshow(bpm_array, cmap='seismic', interpolation='nearest', norm=norm)
    plt.colorbar()
    plt.savefig(f'/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Results_10_7/out/{vid}_'+ext[1:]+'_heatmap.png')

    # Plot the histogram of BPM values
    bpm_flattened = bpm_array.flatten()
    fig = plt.figure(figsize=(6, 4))
    plt.hist(bpm_flattened, bins=50, color='blue', edgecolor='black', zorder=2)
    plt.title(f'BPM Distribution - {vid}')
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Relative Occurrences')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Results_10_7/out/{vid}_'+ext[1:]+'_histogram.png')
   


# Start processing all videos
process_videos()
