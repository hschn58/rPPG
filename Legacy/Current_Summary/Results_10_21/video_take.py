
import cv2
import time
import os



base_dir =  os.path.dirname(os.path.realpath(__file__))+'/'
# Set the duration of the video in seconds
duration = 20  # Record for 5 seconds

description=input("Enter the description of the video (X_light_bpm): ")

# Initialize the video capture object (0 refers to the default camera)
cap = cv2.VideoCapture(0)



height = 1328
width = 1760
# Set the width and height of the video frame

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Define the codec and create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')


output_file = base_dir+'Videos/'+description+'.mp4'
fps = 50  # Set frames per second (you can adjust this value)


out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Calculate the number of frames required for the specified duration
total_frames = int(fps * duration)

# Start recording and capture video frames
start_time = time.time()

for i in range(total_frames):
    # Capture frame-by-frame
    ret, frame = cap.read()

    height, width, channels = frame.shape

    

    if not ret:
        print("Failed to grab frame")
        break

    # Write the frame to the video file
    frame = cv2.flip(frame, 1)  # Flip the frame 100about vertical axis
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

final_hr = input("Enter your final heart rate (bpm): ")

# Rename the video file to include both the initial and final heart rates
new_output_file = base_dir + 'Videos/' + description + '_' + final_hr + '.mp4'
os.rename(output_file, new_output_file)


print(f"Video saved as {new_output_file}")

