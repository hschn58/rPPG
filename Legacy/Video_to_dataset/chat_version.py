import cv2
import numpy as np
import time
import pickle
import os
import params  # Ensure your params module is in the PYTHONPATH

# =============================================================================
# Parameters (from params)
# =============================================================================
DELTA         = params.DELTA
CHEEK_HPAD    = params.CHEEK_HPAD
CHEEK_VPAD    = params.CHEEK_VPAD
CHIN_HPAD     = params.CHIN_HPAD
CHIN_VPAD     = params.CHIN_VPAD
MLCHEEK_HPAD  = params.MLCHEEK_HPAD
MRCHEEK_HPAD  = params.MRCHEEK_HPAD
MRCHEEK_VPAD  = params.MRCHEEK_VPAD
MLCHEEK_VPAD  = params.MLCHEEK_VPAD

FRAME_WIDTH   = params.FRAME_WIDTH
FRAME_HEIGHT  = params.FRAME_HEIGHT

# =============================================================================
# Region Extraction Functions (using correct row,column indexing)
# =============================================================================
def left_cheek(frame, bbox, DELTA=DELTA, CHEEK_HPAD=CHEEK_HPAD, CHEEK_VPAD=CHEEK_VPAD):
    # bbox structure: [x, y, w, h]
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

# =============================================================================
# Face Detection and Template Matching Helpers
# =============================================================================
def find_face(frame, detector):
    # Use YuNet face detector; expects bbox as [x, y, w, h]
    _, faces = detector.detect(frame)
    if faces is not None and len(faces) > 0:
        return faces[0][:4].astype(int)
    return None

def update_face(frame, roi, h, w):
    # Use template matching to locate the face in subsequent frames.
    res = cv2.matchTemplate(frame, roi, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    return [max_loc[0], max_loc[1], w, h]

# =============================================================================
# Main Recording Settings and Initialization
# =============================================================================
# Request heart rate inputs
initial_hr = int(input("Enter heart rate at the beginning (bpm): "))

# Set up output paths (adjust as needed)
output_video_filename = os.path.join(os.getcwd(), "video_processed.mp4")
output_data_filename = os.path.join(os.getcwd(), "region_data.npy")

# Set up camera capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define capture duration (10 seconds) and frame rate
record_duration = 10  # seconds
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback to 30 if undetected
start_time = time.time()

# Prepare containers for frames and regions
frames = []
region_data = []  # This will be a list of length [# frames], each element is a list of 5 region arrays (RGB)

# Initialize output video writer (using same resolution as captured)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_filename, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

# Load YuNet face detector (update model_path as necessary)
model_path = '/Users/henryschnieders/documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
detector = cv2.FaceDetectorYN_create(model_path, "", (FRAME_WIDTH, FRAME_HEIGHT), score_threshold=0.5)

# =============================================================================
# 1. Capture Video for 10 Seconds
# =============================================================================
print("Recording 10-second video...")
while (time.time() - start_time) < record_duration:
    ret, frame = cap.read()
    if not ret:
        continue
    # Optionally, you may flip/mirror the frame (if desired)
    frame = cv2.flip(frame, 1)
    
    # Resize frame if needed (here we use the FRAME_WIDTH and FRAME_HEIGHT from params)
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    
    frames.append(frame.copy())
    
    # For visual feedback during recording, show the raw feed
    cv2.imshow("Recording...", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After recording 10 seconds, ask for the final heart rate.
final_hr = int(input("Enter heart rate at the end (bpm): "))

print("Finished recording.")
cap.release()

# =============================================================================
# 2. Process Frames: Track Face, Extract Regions and Create Overlay Video
# =============================================================================
# We assume at least one frame was recorded.
if len(frames) == 0:
    print("No frames captured!")
    exit()

# Use the first frame to detect the initial face and define a template for tracking.
initial_bbox = find_face(frames[0], detector)
if initial_bbox is None:
    print("Error: No face detected in the first frame!")
    exit()

# Extract face region as a template (convert to grayscale for matching)
template = frames[0][initial_bbox[1]:initial_bbox[1] + initial_bbox[3],
                     initial_bbox[0]:initial_bbox[0] + initial_bbox[2]]
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Set the initial bounding box to use for the first frame.
bbox = initial_bbox

# Process every frame to update face location, extract regions, overlay markings, and store region data.
for idx, frame in enumerate(frames):
    # For tracking using template matching, convert frame to grayscale.
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # For the very first frame, we use the initial detection.
    if idx > 0:
        bbox = update_face(frame_gray, template_gray, initial_bbox[3], initial_bbox[2])
    
    # Extract five regions from the current frame using the corrected functions.
    region_left     = left_cheek(frame, bbox)
    region_right    = right_cheek(frame, bbox)
    region_chin     = chin(frame, bbox)
    region_midleft  = midleft_cheek(frame, bbox)
    region_midright = midright_cheek(frame, bbox)
    
    # Save the full-color (RGB) region data in a list: [region_left, region_right, region_chin, region_midleft, region_midright]
    region_data.append([region_left.copy(), region_right.copy(), region_chin.copy(), region_midleft.copy(), region_midright.copy()])
    
    # --- Overlay: Draw rectangles on the face image for the regions.
    # Compute rectangle coordinates for each region (each region is 2*DELTA in width/height).
    # Left cheek overlay:
    left_top    = (bbox[0] + int(bbox[2] * CHEEK_HPAD) - DELTA, bbox[1] + int(bbox[3] * CHEEK_VPAD) - DELTA)
    right_bottom= (left_top[0] + 2*DELTA, left_top[1] + 2*DELTA)
    cv2.rectangle(frame, left_top, right_bottom, (0, 255, 0), 1)
    
    # Right cheek overlay:
    r_top    = (bbox[0] + int(bbox[2] * (1 - CHEEK_HPAD)) - DELTA, bbox[1] + int(bbox[3] * CHEEK_VPAD) - DELTA)
    r_bottom = (r_top[0] + 2*DELTA, r_top[1] + 2*DELTA)
    cv2.rectangle(frame, r_top, r_bottom, (0, 255, 0), 1)
    
    # Chin overlay:
    chin_top    = (bbox[0] + int(bbox[2] * CHIN_HPAD) - DELTA, bbox[1] + int(bbox[3] * CHIN_VPAD) - DELTA)
    chin_bottom = (chin_top[0] + 2*DELTA, chin_top[1] + 2*DELTA)
    cv2.rectangle(frame, chin_top, chin_bottom, (0, 255, 0), 1)
    
    # Mid left cheek overlay:
    ml_top    = (bbox[0] + int(bbox[2] * MLCHEEK_HPAD) - DELTA, bbox[1] + int(bbox[3] * MLCHEEK_VPAD) - DELTA)
    ml_bottom = (ml_top[0] + 2*DELTA, ml_top[1] + 2*DELTA)
    cv2.rectangle(frame, ml_top, ml_bottom, (0, 255, 0), 1)
    
    # Mid right cheek overlay:
    mr_top    = (bbox[0] + int(bbox[2] * MRCHEEK_HPAD) - DELTA, bbox[1] + int(bbox[3] * MRCHEEK_VPAD) - DELTA)
    mr_bottom = (mr_top[0] + 2*DELTA, mr_top[1] + 2*DELTA)
    cv2.rectangle(frame, mr_top, mr_bottom, (0, 255, 0), 1)
    
    # Add heart rate text: initial HR on first frame, final HR on last frame.
    if idx == 0:
        cv2.putText(frame, f"Initial HR: {initial_hr} bpm", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    elif idx == len(frames) - 1:
        cv2.putText(frame, f"Final HR: {final_hr} bpm", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Optionally also draw the bounding box around the face.
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 2)
    
    # Write the frame with overlays to the output video.
    out.write(frame)
    
    # Optionally display the processed frame:
    cv2.imshow("Processed Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()

# =============================================================================
# 3. Save the Region Data as .npy File in the Format data[FRAME][REGION]
# =============================================================================
region_data = np.array(region_data, dtype=object)  # region_data[frame][region] where region is an RGB patch
np.save(output_data_filename, region_data)
print(f"Processed video saved to {output_video_filename}")
print(f"Region data saved to {output_data_filename}")
