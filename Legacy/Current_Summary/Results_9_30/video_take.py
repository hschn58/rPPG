
import cv2
import time

# Set the duration of the video in seconds
duration = 20  # Record for 5 seconds

description=input("Enter the description of the video: ")

# Initialize the video capture object (0 refers to the default camera)
cap = cv2.VideoCapture(0)

# Set the width and height of the video frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the codec and create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = '/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Results_9_30/Videos/'+description+'.avi'  # Specify the output file path
fps = 50  # Set frames per second (you can adjust this value)
out = cv2.VideoWriter(output_file, fourcc, fps, (640, 480))

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

    # Write the frame to the video file
    frame = cv2.flip(frame, 1)  # Flip the frame about vertical axis
    out.write(frame)

    # Display the resulting frame (optional)
    cv2.imshow('Recording...', frame)

    # Exit the video window by pressing 'q' (optional)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if the duration has been exceeded (this is just an extra safety check)
    if time.time() - start_time > duration:
        break

# Release the capture object and the video writer object
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_file}")
