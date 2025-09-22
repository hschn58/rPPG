import numpy as np
import cv2

# Load the .npy file
filename = '/Users/henryschnieders/Documents/Research/My_work/Data/sporting_1_frames.npy'

# Load the intensity data from the .npy file
intensity_data = np.load(filename, allow_pickle=True)

# Convert each frame to a proper NumPy array with a numeric type
fixed_intensity_data = []
for frame in intensity_data:
    try:
        # Ensure that each frame is converted to a 2D NumPy array of type float32
        fixed_frame = np.array(frame, dtype=np.float32)
        fixed_intensity_data.append(fixed_frame)
    except ValueError as e:
        print(f"Error converting frame to array: {e}")
        continue

# Define the output video file path and name
output_video_path = '/Users/henryschnieders/Documents/Research/My_work/Data/reconstructed_video.mp4'

# Get the dimensions of the frames from the data
num_frames = len(fixed_intensity_data)

print(fixed_intensity_data[0].shape)

exit()

frame_height, frame_width = fixed_intensity_data[0].shape

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For '.mp4' files, use 'mp4v' codec
fps = 30  # Set your desired frames per second (adjust as needed)
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

# Loop through the frames and write each to the video file
for frame in fixed_intensity_data:
    # Normalize the frame data to the range 0-255 and convert to uint8
    frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    frame_uint8 = frame_normalized.astype(np.uint8)

    # Write the frame to the video file
    out.write(frame_uint8)

# Release the VideoWriter object and close any OpenCV windows
out.release()
cv2.destroyAllWindows()

print(f"Video reconstruction completed: {output_video_path}")



#try this, if not email zewei. 