import cv2
import numpy as np
import time
import subprocess
import os
import pickle
import params

# -----------------------------------------------------------------------------
# Parameters for recording and scaling
# -----------------------------------------------------------------------------
duration = 10         # seconds to record
fps = 30
# Full resolution parameters (10-bit recording)
full_width = 1760
full_height = 1328
base_path = "/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Starting_4_5_2025/Data"

# User inputs
video_name = input("Enter video save name (without extension): ")
initial_hr = int(input("Enter initial heart rate (bpm): "))

output_video = f"{base_path}/{video_name}.mp4"
output_data = f"{base_path}/{video_name}_regions.npy"

# -----------------------------------------------------------------------------
# FFmpeg command for 10-bit capture
# -----------------------------------------------------------------------------
ffmpeg_command = [
    'ffmpeg',
    '-f', 'avfoundation',
    '-framerate', str(fps),
    '-i', '0',
    '-pix_fmt', 'yuv420p10le',
    '-c:v', 'libx264',
    '-profile:v', 'high10',
    '-r', str(fps),
    '-s', f'{full_width}x{full_height}',
    '-t', str(duration),
    '-y',  # Overwrite output file if it exists
    output_video
]

print("Recording 10-bit video for 10 seconds...")
process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
if process.returncode != 0:
    print("Error during recording:")
    print(stderr.decode())
    exit(1)

# -----------------------------------------------------------------------------
# Load recorded video frames
# -----------------------------------------------------------------------------
cap = cv2.VideoCapture(output_video)
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print(f"Total frames captured: {len(frames)}")

# -----------------------------------------------------------------------------
# Set up output video writer for processed video (with overlays)
# -----------------------------------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video_filename = output_video.replace(".mp4", "_processed.mp4")
out = cv2.VideoWriter(out_video_filename, fourcc, fps, (full_width, full_height))

# -----------------------------------------------------------------------------
# Define region extraction functions (using correct row,column indexing)
# -----------------------------------------------------------------------------
DELTA         = params.DELTA
CHEEK_HPAD    = params.CHEEK_HPAD
CHEEK_VPAD    = params.CHEEK_VPAD
CHIN_HPAD     = params.CHIN_HPAD
CHIN_VPAD     = params.CHIN_VPAD
MLCHEEK_HPAD  = params.MLCHEEK_HPAD
MRCHEEK_HPAD  = params.MRCHEEK_HPAD
MRCHEEK_VPAD  = params.MRCHEEK_VPAD
MLCHEEK_VPAD  = params.MLCHEEK_VPAD

def left_cheek(frame, bbox, DELTA=DELTA, CHEEK_HPAD=CHEEK_HPAD, CHEEK_VPAD=CHEEK_VPAD):
    # bbox is [x, y, w, h] in full-resolution coordinates
    return frame[(bbox[1] + int(bbox[3] * CHEEK_VPAD) - DELTA):(bbox[1] + int(bbox[3] * CHEEK_VPAD) + DELTA),
                 (bbox[0] + int(bbox[2] * CHEEK_HPAD) - DELTA):(bbox[0] + int(bbox[2] * CHEEK_HPAD) + DELTA)]

def right_cheek(frame, bbox, DELTA=DELTA, CHEEK_HPAD=CHEEK_HPAD, CHEEK_VPAD=CHEEK_VPAD):
    return frame[(bbox[1] + int(bbox[3] * CHEEK_VPAD) - DELTA):(bbox[1] + int(bbox[3] * CHEEK_VPAD) + DELTA),
                 (bbox[0] + int(bbox[2] * (1 - CHEEK_HPAD)) - DELTA):(bbox[0] + int(bbox[2] * (1 - CHEEK_HPAD)) + DELTA)]

def chin(frame, bbox, DELTA=DELTA, CHIN_HPAD=CHIN_HPAD, CHIN_VPAD=CHIN_VPAD):
    return frame[(bbox[1] + int(bbox[3] * CHIN_VPAD) - DELTA):(bbox[1] + int(bbox[3] * CHIN_VPAD) + DELTA),
                 (bbox[0] + int(bbox[2] * CHIN_HPAD) - DELTA):(bbox[0] + int(bbox[2] * CHIN_HPAD) + DELTA)]

def midleft_cheek(frame, bbox, DELTA=DELTA, MLCHEEK_HPAD=MLCHEEK_HPAD, MLCHEEK_VPAD=MLCHEEK_VPAD):
    return frame[(bbox[1] + int(bbox[3] * MLCHEEK_VPAD) - DELTA):(bbox[1] + int(bbox[3] * MLCHEEK_VPAD) + DELTA),
                 (bbox[0] + int(bbox[2] * MLCHEEK_HPAD) - DELTA):(bbox[0] + int(bbox[2] * MLCHEEK_HPAD) + DELTA)]

def midright_cheek(frame, bbox, DELTA=DELTA, MRCHEEK_HPAD=MRCHEEK_HPAD, MRCHEEK_VPAD=MRCHEEK_VPAD):
    return frame[(bbox[1] + int(bbox[3] * MRCHEEK_VPAD) - DELTA):(bbox[1] + int(bbox[3] * MRCHEEK_VPAD) + DELTA),
                 (bbox[0] + int(bbox[2] * MRCHEEK_HPAD) - DELTA):(bbox[0] + int(bbox[2] * MRCHEEK_HPAD) + DELTA)]

# -----------------------------------------------------------------------------
# Face detection / tracking helper functions
# -----------------------------------------------------------------------------
def find_face(frame, detector):
    # detector.detect returns faces in the form [x, y, w, h]
    _, faces = detector.detect(frame)
    if faces is not None and len(faces) > 0:
        return faces[0][:4].astype(int)
    return None

def update_face(frame, template, h, w):
    # Use template matching to update bbox in a frame
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    return [max_loc[0], max_loc[1], w, h]

# -----------------------------------------------------------------------------
# Create a "small" detector for 300x300 frames
# -----------------------------------------------------------------------------
small_size = (300, 300)
model_path = '/Users/henryschnieders/documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
detector_small = cv2.FaceDetectorYN_create(model_path, "", small_size, score_threshold=0.5)

# -----------------------------------------------------------------------------
# Set up scale factors for mapping small-frame detections to full resolution.
# -----------------------------------------------------------------------------
scale_x = full_width / small_size[0]
scale_y = full_height / small_size[1]

# -----------------------------------------------------------------------------
# Process frames: Face tracking and region extraction
# -----------------------------------------------------------------------------
region_data = []  # will be list of frames; each frame is a list of 5 RGB patch arrays

# Process the first frame: detect face in the 300x300 version and scale bbox to full resolution.
first_frame_full = frames[0]
first_frame_small = cv2.resize(first_frame_full, small_size)
bbox_small = find_face(first_frame_small, detector_small)
if bbox_small is None:
    print("Error: No face detected in the first frame (small resolution)!")
    exit(1)

# Scale bbox_small to full resolution:
bbox_full = [int(bbox_small[0] * scale_x), int(bbox_small[1] * scale_y), 
             int(bbox_small[2] * scale_x), int(bbox_small[3] * scale_y)]

# Create a template for tracking in the small domain.
template_small = first_frame_small[bbox_small[1]:bbox_small[1] + bbox_small[3],
                                   bbox_small[0]:bbox_small[0] + bbox_small[2]]
template_small_gray = cv2.cvtColor(template_small, cv2.COLOR_BGR2GRAY)

# We'll use bbox_small for tracking in the 300x300 image, then scale it to full resolution.
current_bbox_small = bbox_small

# Process each full resolution frame.
for idx, full_frame in enumerate(frames):
    # Create a small version for detection/tracking.
    small_frame = cv2.resize(full_frame, small_size)
    small_frame_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # For frames after the first, update face location using template matching in the small frame.
    if idx > 0:
        # Update bbox in small resolution.
        current_bbox_small = update_face(small_frame_gray, template_small_gray, current_bbox_small[3], current_bbox_small[2])
    
    # Scale the updated small bbox to the full resolution.
    bbox_full = [ int(current_bbox_small[0] * scale_x), int(current_bbox_small[1] * scale_y),
                  int(current_bbox_small[2] * scale_x), int(current_bbox_small[3] * scale_y) ]
    
    # Extract the five regions from the full resolution frame.
    region_left     = left_cheek(full_frame, bbox_full)
    region_right    = right_cheek(full_frame, bbox_full)
    region_chin     = chin(full_frame, bbox_full)
    region_midleft  = midleft_cheek(full_frame, bbox_full)
    region_midright = midright_cheek(full_frame, bbox_full)
    
    # Save the five regions for this frame (each as an RGB patch).
    region_data.append([region_left.copy(), region_right.copy(), region_chin.copy(), 
                        region_midleft.copy(), region_midright.copy()])
    
    # Overlay the five region rectangles on the full resolution frame.
    # Note: We re-compute the rectangle coordinates based on bbox_full.
    # Left cheek overlay:
    left_top = (bbox_full[0] + int(bbox_full[2] * CHEEK_HPAD) - DELTA, 
                bbox_full[1] + int(bbox_full[3] * CHEEK_VPAD) - DELTA)
    left_bottom = (left_top[0] + 2*DELTA, left_top[1] + 2*DELTA)
    cv2.rectangle(full_frame, left_top, left_bottom, (0, 255, 0), 1)
    
    # Right cheek overlay:
    r_top = (bbox_full[0] + int(bbox_full[2] * (1 - CHEEK_HPAD)) - DELTA, 
             bbox_full[1] + int(bbox_full[3] * CHEEK_VPAD) - DELTA)
    r_bottom = (r_top[0] + 2*DELTA, r_top[1] + 2*DELTA)
    cv2.rectangle(full_frame, r_top, r_bottom, (0, 255, 0), 1)
    
    # Chin overlay:
    chin_top = (bbox_full[0] + int(bbox_full[2] * CHIN_HPAD) - DELTA, 
                bbox_full[1] + int(bbox_full[3] * CHIN_VPAD) - DELTA)
    chin_bottom = (chin_top[0] + 2*DELTA, chin_top[1] + 2*DELTA)
    cv2.rectangle(full_frame, chin_top, chin_bottom, (0, 255, 0), 1)
    
    # Mid left cheek overlay:
    ml_top = (bbox_full[0] + int(bbox_full[2] * MLCHEEK_HPAD) - DELTA, 
              bbox_full[1] + int(bbox_full[3] * MLCHEEK_VPAD) - DELTA)
    ml_bottom = (ml_top[0] + 2*DELTA, ml_top[1] + 2*DELTA)
    cv2.rectangle(full_frame, ml_top, ml_bottom, (0, 255, 0), 1)
    
    # Mid right cheek overlay:
    mr_top = (bbox_full[0] + int(bbox_full[2] * MRCHEEK_HPAD) - DELTA, 
              bbox_full[1] + int(bbox_full[3] * MRCHEEK_VPAD) - DELTA)
    mr_bottom = (mr_top[0] + 2*DELTA, mr_top[1] + 2*DELTA)
    cv2.rectangle(full_frame, mr_top, mr_bottom, (0, 255, 0), 1)
    
    # Overlay heart rate text: initial HR on first frame, final HR on last frame.
    if idx == 0:
        cv2.putText(full_frame, f"Initial HR: {initial_hr} bpm", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    elif idx == len(frames) - 1:
        final_hr = int(input("Enter heart rate at the end (bpm): "))
        cv2.putText(full_frame, f"Final HR: {final_hr} bpm", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    # Optionally, also draw the full face bounding box.
    cv2.rectangle(full_frame, (bbox_full[0], bbox_full[1]), (bbox_full[0]+bbox_full[2], bbox_full[1]+bbox_full[3]), (255,0,0), 2)
    
    # Write the overlayed full-resolution frame to the output video.
    out.write(full_frame)
    
    # Optional: display the processed frame.
    cv2.imshow("Processed Video", full_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# Save the region data as a .npy file (format: data[FRAME][REGION])
# -----------------------------------------------------------------------------
region_data = np.array(region_data, dtype=object)
np.save(output_data, region_data)

print(f"Processed video saved to {out_video_filename}")
print(f"Region data saved to {output_data}")
