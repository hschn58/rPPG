import numpy as np
from scipy.io import loadmat
import os
import cv2

def load_mat_files(folder_path):
    """
    Loads the main .mat files (I_1 and Q_1) and extracts individual datasets.

    :param folder_path: Path to the folder containing .mat files
    :return: Two dictionaries containing in-phase (I) and quadrature (Q) data
    """
    # Load the main .mat files
    I_data_path = os.path.join(folder_path, 'I_1.mat')
    Q_data_path = os.path.join(folder_path, 'Q_1.mat')

    if not os.path.exists(I_data_path) or not os.path.exists(Q_data_path):
        print(f"Required files not found. Ensure that 'I_1.mat' and 'Q_1.mat' are in the directory: {folder_path}")
        return None, None

    # Load the data from .mat files
    I_data = loadmat(I_data_path)
    Q_data = loadmat(Q_data_path)

    # Extract all datasets from the loaded .mat files
    I_frames = {key: I_data[key] for key in I_data if key.startswith('I_')}
    Q_frames = {key: Q_data[key] for key in Q_data if key.startswith('Q_')}

    return I_frames, Q_frames

def compute_intensity(I_frames, Q_frames):
    """
    Computes the image intensity from in-phase (I) and quadrature (Q) components.

    :param I_frames: Dictionary containing in-phase (I) component data
    :param Q_frames: Dictionary containing quadrature (Q) component data
    :return: Numpy array containing image intensity for each frame
    """
    if not I_frames or not Q_frames:
        print("No data loaded. Please check if the .mat files are correctly placed in the folder.")
        return None

    frame_count = len(I_frames)
    height, width = list(I_frames.values())[0].shape  # Get the shape from any frame

    intensity_frames = np.zeros((frame_count, height, width))

    for i, key in enumerate(I_frames.keys()):
        I_frame = I_frames[key]
        Q_frame = Q_frames[key.replace('I_', 'Q_')]  # Match corresponding Q data
        intensity_frames[i] = np.sqrt(I_frame ** 2 + Q_frame ** 2)

    return intensity_frames

def save_to_npy(intensity_frames, output_path):
    """
    Saves the computed intensity frames to an .npy file.

    :param intensity_frames: Numpy array containing image intensity for each frame
    :param output_path: Path to save the .npy file
    """
    if intensity_frames is not None:
        np.save(output_path, intensity_frames)
        print(f"Intensity frames saved to {output_path}")
    else:
        print("No intensity frames to save.")

def save_to_video(intensity_frames, output_video_path, fps=30):
    """
    Saves the computed intensity frames to a video file.

    :param intensity_frames: Numpy array containing image intensity for each frame
    :param output_video_path: Path to save the output video
    :param fps: Frames per second for the output video
    """
    # Determine the height and width from the first frame
    height, width = intensity_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

    for frame in intensity_frames:
        # Normalize and convert frame to uint8 for OpenCV compatibility
        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        out.write(frame_normalized)
    
    out.release()
    print(f"Video saved to {output_video_path}")

def main():
    folder_path = '/Users/henryschnieders/Documents/Research/My_work/Data/New_sporting/1'  # Update this path to the location of your .mat files
    output_npy_path = '/Users/henryschnieders/Documents/Research/My_work/Data/New_sporting/1/output_intensity_frames.npy'
    output_video_path = '/Users/henryschnieders/Documents/Research/My_work/Data/New_sporting/1/output_intensity_video.mp4'

    # Load I_ and Q_ data from .mat files
    I_frames, Q_frames = load_mat_files(folder_path)

    # Check if data is loaded properly
    if not I_frames or not Q_frames:
        print("Failed to load data. Please check the .mat files.")
        return

    # Compute intensity
    intensity_frames = compute_intensity(I_frames, Q_frames)

    # Save to .npy
    save_to_npy(intensity_frames, output_npy_path)

    # Save to video
    save_to_video(intensity_frames, output_video_path, fps=30)

if __name__ == "__main__":
    main()
