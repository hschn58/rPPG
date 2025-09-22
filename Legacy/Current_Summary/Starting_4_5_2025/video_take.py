import cv2
import time
import numpy as np

# Set the duration of the video in seconds
duration = 20  # Record for 20 seconds

description = input("Enter the description of the video: ")

# Initialize the video capture object (0 refers to the default camera)
cap = cv2.VideoCapture(0)

height = 1328
width = 1760

# Set the width and height of the video frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Define the codec (H.264 instead of XVID for potential 10-bit support)
fourcc = cv2.VideoWriter_fourcc(*'H264')  # H.264 codec
output_file = '/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Starting_4_5_2025/Data/' + description + '.mp4'
fps = 50  # Set frames per second

# Create a VideoWriter object
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Check if VideoWriter is opened successfully
if not out.isOpened():
    print("Error: VideoWriter failed to initialize.")
    cap.release()
    exit()

# Calculate the number of frames required for the specified duration
total_frames = int(fps * duration)

# Start recording and capture video frames
start_time = time.time()

for i in range(total_frames):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Get frame dimensions
    height, width, channels = frame.shape

    # Convert frame to 10-bit (simplified approach, scaling 8-bit to 10-bit range)
    # Note: OpenCV captures in 8-bit BGR by default, so we simulate 10-bit by scaling
    frame_10bit = (frame.astype(np.uint16) * 4).clip(0, 1023)  # Scale 0-255 to 0-1023 (10-bit)

    # Flip the frame about the vertical axis
    frame_10bit = cv2.flip(frame_10bit, 1)

    # Convert back to 8-bit for writing (limitation of OpenCV's VideoWriter)
    frame_8bit = (frame_10bit // 4).astype(np.uint8)

    # Write the frame to the video file
    out.write(frame_8bit)

    # Display the resulting frame (optional)
    cv2.imshow('Recording...', frame_8bit)

    # Exit the video window by pressing 'q' (optional)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if the duration has been exceeded
    if time.time() - start_time > duration:
        break

# Release the capture object and the video writer object
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_file}")