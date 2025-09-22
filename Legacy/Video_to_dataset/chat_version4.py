import cv2
import numpy as np
import time
import subprocess
import os
import pickle
import params
import av  # PyAV for high-bit decoding

# =============================================================================
# (A) Recording: Use FFmpeg to capture a 10-bit video at full resolution
# =============================================================================
duration = 10         # seconds
fps = 30
full_width = 1760
full_height = 1328
base_path = "/Users/henryschnieders/Documents/Research/My_Data"

video_name = input("Enter video save name (without extension): ")
initial_hr = int(input("Enter heart rate at the beginning (bpm): "))

output_video = f"{base_path}/{video_name}.mp4"
output_data  = f"{base_path}/{video_name}_regions.npy"

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

# =============================================================================
# (B) Use OpenCV (8-bit) to obtain the face coordinates (tracking using full-resolution)
# =============================================================================

# For initial detection we use a small resolution version (300×300) required by YuNet.
small_size = (300, 300)
model_path = '/Users/henryschnieders/documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
detector_small = cv2.FaceDetectorYN_create(model_path, "", small_size, score_threshold=0.5)

# Open the recorded video with VideoCapture (8-bit frames)
cap_cv = cv2.VideoCapture(output_video)
frames_cv = []
while cap_cv.isOpened():
    ret, frame = cap_cv.read()
    if not ret:
        break
    frames_cv.append(frame.copy())
cap_cv.release()
print(f"Total frames (8-bit CV2): {len(frames_cv)}")

# Compute scale factors for mapping from small (300x300) to full resolution.
scale_x = full_width / small_size[0]
scale_y = full_height / small_size[1]

# --- Define region extraction functions (using row, column indexing) ---
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

def find_face_small(frame, detector):
    # Face detection on a 300x300 frame. Returns bbox as [x,y,w,h].
    _, faces = detector.detect(frame)
    if faces is not None and len(faces) > 0:
        return faces[0][:4].astype(int)
    return None

# --- Initial detection on the small image ---
first_frame_full = frames_cv[0]
first_frame_small = cv2.resize(first_frame_full, small_size)
bbox_small = find_face_small(first_frame_small, detector_small)
if bbox_small is None:
    print("Error: No face detected in the first frame (small resolution)!")
    exit(1)
# Scale detected bbox to full resolution.
bbox_full = [ int(bbox_small[0]*scale_x), int(bbox_small[1]*scale_y),
              int(bbox_small[2]*scale_x), int(bbox_small[3]*scale_y) ]

# Now extract a full-resolution template for tracking from the first frame.
template_full = first_frame_full[bbox_full[1]:bbox_full[1]+bbox_full[3],
                                 bbox_full[0]:bbox_full[0]+bbox_full[2]]
template_full_gray = cv2.cvtColor(template_full, cv2.COLOR_BGR2GRAY)

# Initialize the current bbox in full resolution.
current_bbox_full = bbox_full.copy()

# We save the face bbox for each frame in a list.
bbox_list = []
overlay_frames = []  # For visualization output (overlayed on 8-bit frames)

for idx, full_frame in enumerate(frames_cv):
    # Use the full resolution frame (convert to grayscale) for template matching.
    full_gray = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
    
    if idx > 0:
        # Update bbox using full resolution template matching.
        res = cv2.matchTemplate(full_gray, template_full_gray, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        # Update current_bbox_full while preserving width and height.
        current_bbox_full = [max_loc[0], max_loc[1], current_bbox_full[2], current_bbox_full[3]]
    
    bbox_list.append(current_bbox_full.copy())
    
    # For overlay, draw the face bbox and the region rectangles.
    proc_frame = full_frame.copy()
    cv2.rectangle(proc_frame, (current_bbox_full[0], current_bbox_full[1]),
                  (current_bbox_full[0]+current_bbox_full[2], current_bbox_full[1]+current_bbox_full[3]), (255,0,0), 2)
    
    # Compute region coordinates and draw rectangles.
    left_top = (current_bbox_full[0] + int(current_bbox_full[2] * CHEEK_HPAD) - DELTA,
                current_bbox_full[1] + int(current_bbox_full[3] * CHEEK_VPAD) - DELTA)
    cv2.rectangle(proc_frame, left_top, (left_top[0]+2*DELTA, left_top[1]+2*DELTA), (0,255,0), 1)
    
    r_top = (current_bbox_full[0] + int(current_bbox_full[2] * (1 - CHEEK_HPAD)) - DELTA,
             current_bbox_full[1] + int(current_bbox_full[3] * CHEEK_VPAD) - DELTA)
    cv2.rectangle(proc_frame, r_top, (r_top[0]+2*DELTA, r_top[1]+2*DELTA), (0,255,0), 1)
    
    chin_top = (current_bbox_full[0] + int(current_bbox_full[2] * CHIN_HPAD) - DELTA,
                current_bbox_full[1] + int(current_bbox_full[3] * CHIN_VPAD) - DELTA)
    cv2.rectangle(proc_frame, chin_top, (chin_top[0]+2*DELTA, chin_top[1]+2*DELTA), (0,255,0), 1)
    
    ml_top = (current_bbox_full[0] + int(current_bbox_full[2] * MLCHEEK_HPAD) - DELTA,
              current_bbox_full[1] + int(current_bbox_full[3] * MLCHEEK_VPAD) - DELTA)
    cv2.rectangle(proc_frame, ml_top, (ml_top[0]+2*DELTA, ml_top[1]+2*DELTA), (0,255,0), 1)
    
    mr_top = (current_bbox_full[0] + int(current_bbox_full[2] * MRCHEEK_HPAD) - DELTA,
              current_bbox_full[1] + int(current_bbox_full[3] * MRCHEEK_VPAD) - DELTA)
    cv2.rectangle(proc_frame, mr_top, (mr_top[0]+2*DELTA, mr_top[1]+2*DELTA), (0,255,0), 1)
    
    # Add initial heart rate text on the first frame.
    if idx == 0:
        cv2.putText(proc_frame, f"Initial HR: {initial_hr} bpm", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    overlay_frames.append(proc_frame)

# =============================================================================
# (C) Extract 10-bit region data from the video using PyAV.
# =============================================================================

def convert_to_10bit(img):
    """
    Map 16-bit pixel data (range 0-65535) to a 10-bit range (0-1023).
    """
    # Convert to float, scale, round, and convert back to uint16.
    return np.clip(((img.astype(np.float32) / 65535.0) * 1023.0).round(), 0, 1023).astype(np.uint16)

# Open the 10-bit video with PyAV.
container = av.open(output_video)
region_data = []  # region_data[frame][region] where each region is a 10-bit patch
frame_index = 0
for packet in container.demux(video=0):
    for frame in packet.decode():
        # Decode the frame in high bit-depth ("rgb48" gives 16-bit per channel)
        frame_16bit = frame.to_ndarray(format='rgb48')
        # Use the corresponding bounding box computed in our CV2 workflow.
        if frame_index >= len(bbox_list):
            break
        bbox_full = bbox_list[frame_index]
        
        reg_left     = left_cheek(frame_16bit, bbox_full)
        reg_right    = right_cheek(frame_16bit, bbox_full)
        reg_chin     = chin(frame_16bit, bbox_full)
        reg_midleft  = midleft_cheek(frame_16bit, bbox_full)
        reg_midright = midright_cheek(frame_16bit, bbox_full)
        
        # Convert each region from 16-bit to 10-bit.
        reg_left     = convert_to_10bit(reg_left)
        reg_right    = convert_to_10bit(reg_right)
        reg_chin     = convert_to_10bit(reg_chin)
        reg_midleft  = convert_to_10bit(reg_midleft)
        reg_midright = convert_to_10bit(reg_midright)
        
        region_data.append([reg_left, reg_right, reg_chin, reg_midleft, reg_midright])
        frame_index += 1

# =============================================================================
# (D) Write output video with overlays.
# =============================================================================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video_filename = output_video.replace(".mp4", "_processed.mp4")
out = cv2.VideoWriter(out_video_filename, fourcc, fps, (full_width, full_height))

# Ask for final heart rate and overlay on the last frame.
final_hr = int(input("Enter heart rate at the end (bpm): "))
overlay_frames[-1] = overlay_frames[-1].copy()
cv2.putText(overlay_frames[-1], f"Final HR: {final_hr} bpm", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

for frame in overlay_frames:
    out.write(frame)
    cv2.imshow("Processed Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cv2.destroyAllWindows()

# =============================================================================
# (E) Save the region data as a .npy file (format: data[FRAME][REGION])
# =============================================================================
region_data = np.array(region_data, dtype=object)
np.save(output_data, region_data)

print(f"Processed video saved to {out_video_filename}")
print(f"Region data (10-bit) saved to {output_data}")
