import cv2
import numpy as np
import time
import subprocess
import os
import pickle
import params
import av  # PyAV for decoding 10-bit frames

# =============================================================================
# Recording settings and paths (10-bit FFmpeg capture)
# =============================================================================
duration = 10         # seconds
fps = 30
full_width = 1760
full_height = 1328
base_path = "/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Starting_4_5_2025/Data"

video_name = input("Enter video save name (without extension): ")
initial_hr = int(input("Enter heart rate at the beginning (bpm): "))

output_video = f"{base_path}/{video_name}.mp4"
output_data = f"{base_path}/{video_name}_regions.npy"

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
# STEP 1: Use OpenCV to process the video (8-bit) for face detection and region coordinates.
# =============================================================================

# For our face detector we need to work on 300x300 resolution
small_size = (300, 300)

# Create a small-face detector (YuNet requires 300x300)
model_path = '/Users/henryschnieders/documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
detector_small = cv2.FaceDetectorYN_create(model_path, "", small_size, score_threshold=0.5)

# Open the video with cv2.VideoCapture (this will decode to 8-bit)
cap_cv = cv2.VideoCapture(output_video)
frames_cv = []
while cap_cv.isOpened():
    ret, frame = cap_cv.read()
    if not ret:
        break
    # We work with full-res frame here (note: these are 8-bit copies)
    frames_cv.append(frame.copy())
cap_cv.release()
print(f"Total frames (8-bit from CV2): {len(frames_cv)}")

# To set up scaling from small resolution (300x300) to full resolution:
scale_x = full_width / small_size[0]
scale_y = full_height / small_size[1]

# Define your region extraction functions (make sure to use row, column order)
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
    # Face detection on a small (300x300) image. Returns bbox as [x,y,w,h].
    _, faces = detector.detect(frame)
    if faces is not None and len(faces) > 0:
        return faces[0][:4].astype(int)
    return None

def update_face_small(frame_gray, template_gray, bbox, w, h):
    # Template matching in the small image
    res = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    return [max_loc[0], max_loc[1], w, h]

# Process the first frame: detect face on small version.
first_frame_full = frames_cv[0]
first_frame_small = cv2.resize(first_frame_full, small_size)
bbox_small = find_face_small(first_frame_small, detector_small)
if bbox_small is None:
    print("No face detected in the first frame (small resolution)!")
    exit(1)
# Scale to full resolution:
bbox_full = [int(bbox_small[0]*scale_x), int(bbox_small[1]*scale_y),
             int(bbox_small[2]*scale_x), int(bbox_small[3]*scale_y)]

# Create a template from the small image for tracking (using the detected face region)
template_small = first_frame_small[bbox_small[1]:bbox_small[1]+bbox_small[3],
                                   bbox_small[0]:bbox_small[0]+bbox_small[2]]
template_small_gray = cv2.cvtColor(template_small, cv2.COLOR_BGR2GRAY)

# Store region coordinates (or simply store the updated bbox per frame) using the cv2 workflow.
bbox_list = []  # Save face bbox coordinates (in full resolution) for each frame.
overlay_frames = []  # Processed frames for the overlay video

current_bbox_small = bbox_small
for idx, full_frame in enumerate(frames_cv):
    # Create a small version for detection/tracking.
    small_frame = cv2.resize(full_frame, small_size)
    small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    if idx > 0:
        current_bbox_small = update_face_small(small_gray, template_small_gray, current_bbox_small, current_bbox_small[2], current_bbox_small[3])
    
    # Scale small bbox to full resolution
    bbox_full = [ int(current_bbox_small[0]*scale_x), int(current_bbox_small[1]*scale_y),
                  int(current_bbox_small[2]*scale_x), int(current_bbox_small[3]*scale_y) ]
    bbox_list.append(bbox_full)
    
    # (Optional) Draw overlays on the 8-bit frame for inspection.
    proc_frame = full_frame.copy()
    # Face bounding box:
    cv2.rectangle(proc_frame, (bbox_full[0], bbox_full[1]), (bbox_full[0]+bbox_full[2], bbox_full[1]+bbox_full[3]), (255,0,0), 2)
    # Compute region rectangles using same logic:
    left_top = (bbox_full[0] + int(bbox_full[2] * CHEEK_HPAD) - DELTA,
                bbox_full[1] + int(bbox_full[3] * CHEEK_VPAD) - DELTA)
    cv2.rectangle(proc_frame, left_top, (left_top[0]+2*DELTA, left_top[1]+2*DELTA), (0,255,0), 1)
    
    r_top = (bbox_full[0] + int(bbox_full[2] * (1 - CHEEK_HPAD)) - DELTA,
             bbox_full[1] + int(bbox_full[3] * CHEEK_VPAD) - DELTA)
    cv2.rectangle(proc_frame, r_top, (r_top[0]+2*DELTA, r_top[1]+2*DELTA), (0,255,0), 1)
    
    chin_top = (bbox_full[0] + int(bbox_full[2] * CHIN_HPAD) - DELTA,
                bbox_full[1] + int(bbox_full[3] * CHIN_VPAD) - DELTA)
    cv2.rectangle(proc_frame, chin_top, (chin_top[0]+2*DELTA, chin_top[1]+2*DELTA), (0,255,0), 1)
    
    ml_top = (bbox_full[0] + int(bbox_full[2] * MLCHEEK_HPAD) - DELTA,
              bbox_full[1] + int(bbox_full[3] * MLCHEEK_VPAD) - DELTA)
    cv2.rectangle(proc_frame, ml_top, (ml_top[0]+2*DELTA, ml_top[1]+2*DELTA), (0,255,0), 1)
    
    mr_top = (bbox_full[0] + int(bbox_full[2] * MRCHEEK_HPAD) - DELTA,
              bbox_full[1] + int(bbox_full[3] * MRCHEEK_VPAD) - DELTA)
    cv2.rectangle(proc_frame, mr_top, (mr_top[0]+2*DELTA, mr_top[1]+2*DELTA), (0,255,0), 1)
    
    # Add HR text: initial on first frame, and (later) final on last frame.
    if idx == 0:
        cv2.putText(proc_frame, f"Initial HR: {initial_hr} bpm", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2,cv2.LINE_AA)
    overlay_frames.append(proc_frame)

# =============================================================================
# STEP 2: Use PyAV to decode the video in 10-bit (rgb48) and extract regions using stored coordinates.
# =============================================================================

# Open the video with PyAV to get high-bit-depth frames.
container = av.open(output_video)
region_data = []  # region_data[frame][region] with high-bit pixels

# We assume the video frames in PyAV and those in our cv2 workflow match in order.
frame_index = 0
for packet in container.demux(video=0):
    for frame in packet.decode():
        # Get the corresponding high bit-depth frame as RGB48 (16-bit per channel)
        # Note: The effective 10-bit values will be scaled into 16-bit space.
        frame_10bit = frame.to_ndarray(format='rgb48')
        
        # Use stored bbox (full resolution) for this frame.
        if frame_index >= len(bbox_list):
            break
        bbox_full = bbox_list[frame_index]
        
        # Extract regions from the high-bit frame
        reg_left     = left_cheek(frame_10bit, bbox_full)
        reg_right    = right_cheek(frame_10bit, bbox_full)
        reg_chin     = chin(frame_10bit, bbox_full)
        reg_midleft  = midleft_cheek(frame_10bit, bbox_full)
        reg_midright = midright_cheek(frame_10bit, bbox_full)
        region_data.append([reg_left.copy(), reg_right.copy(), reg_chin.copy(), reg_midleft.copy(), reg_midright.copy()])
        
        frame_index += 1

# =============================================================================
# STEP 3: Write the output video with overlays (using the full-resolution 8-bit frames)
# =============================================================================
# We will write the overlay video based on our original cv2 frames with drawn rectangles.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video_filename = output_video.replace(".mp4", "_processed.mp4")
out = cv2.VideoWriter(out_video_filename, fourcc, fps, (full_width, full_height))

# For the final frame, ask for final HR
final_hr = int(input("Enter heart rate at the end (bpm): "))
# Overlay final HR text on the last overlay frame
overlay_frames[-1] = overlay_frames[-1].copy()
cv2.putText(overlay_frames[-1], f"Final HR: {final_hr} bpm", (50,50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

for frame in overlay_frames:
    out.write(frame)
    cv2.imshow("Processed Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cv2.destroyAllWindows()

# =============================================================================
# STEP 4: Save the high-bit region data as .npy (data[FRAME][REGION])
# =============================================================================
region_data = np.array(region_data, dtype=object)
np.save(output_data, region_data)

print(f"Processed video saved to {out_video_filename}")
print(f"Region data (10-bit) saved to {output_data}")
