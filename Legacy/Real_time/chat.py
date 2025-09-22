import params
from signal_processing import plot_locs_cv2, butter_bandpass_filter, bpm_measure
from methods import compute_roi_mean

import cv2
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Get parameters
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

frame_interval = params.FRAME_INTERVAL
frame_width    = params.FRAME_WIDTH
frame_height   = params.FRAME_HEIGHT

lowcut_heart   = params.lowcut_heart
highcut_heart  = params.highcut_heart

graph_height   = params.graph_height
graph_width    = params.graph_width

# -----------------------------------------------------------------------------
# Set up plotting for the real-time heart rate signal
# -----------------------------------------------------------------------------
plt.ion()
graph_width_plot = 75  # plot window width

with plt.ion():
    fig, ax = plt.subplots()
    ax.set_xlim(0, graph_width_plot)
    ax.set_ylim(-1.5, 1.5)
    line, = ax.plot([], [])
    plt.xlabel('Time (s)')
    fig.canvas.manager.set_window_title('Real-Time Heart Rate Signal')

heart_rate_data = []
time_data       = []

def set_title(bpm, fig):
    fig.canvas.manager.set_window_title(f'Real-Time Heart Rate Signal: {bpm:.2f} bpm')

past_time      = time.time()
reference_time = past_time

def update_graph(signal, fig, ax, past_time, graph_width=graph_width_plot, reference_time=reference_time):
    global time_data, heart_rate_data
    cur_delt = time.time() - past_time
    rel_delt = past_time - reference_time

    # Extend time data based on the signal length
    time_data += list(np.linspace(0 + rel_delt, cur_delt + rel_delt, len(signal)))
    heart_rate_data += list(signal)

    if len(time_data) > graph_width:
        time_data = time_data[-graph_width:]
        heart_rate_data = heart_rate_data[-graph_width:]
        ax.set_xlim(time_data[0], time_data[-1])
        ax.set_ylim(min(heart_rate_data), max(heart_rate_data))

    line.set_xdata(time_data)
    line.set_ydata(heart_rate_data)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.plot(time_data, heart_rate_data, color='b')

# -----------------------------------------------------------------------------
# Define helper functions for face detection/tracking and region extraction
# -----------------------------------------------------------------------------
def find_face(frame, detector):
    # Detect faces using YuNet
    _, faces = detector.detect(frame)
    if faces is not None:
        return faces[0][:4].astype(int)  # returns [x, y, w, h]
    return None

def update_face(frame, roi, h, w):
    # Use template matching to update the face bounding box
    res = cv2.matchTemplate(frame, roi, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    return [top_left[0], top_left[1], w, h]

# The following functions extract a small region (of size ~2*DELTA) at a specified fractional position
# Note that indexing is done as frame[rows, columns] (i.e., [y, x]). So we use bbox[1] for rows and bbox[0] for columns.
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

# -----------------------------------------------------------------------------
# Initialize face detector (YuNet) and camera
# -----------------------------------------------------------------------------
model_path = '/Users/henryschnieders/documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
detector = cv2.FaceDetectorYN_create(model_path, "", (frame_width, frame_height), score_threshold=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
time.sleep(5)  # allow the camera to warm up

fps = cap.get(cv2.CAP_PROP_FPS)
sampling_rate = fps
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Failed to capture image.")
    exit()

frames    = []
face_area = []  # will store a list of region sets (each set contains 5 regions)

# -----------------------------------------------------------------------------
# Main loop: capture frames, track the face and extract regions
# -----------------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save a copy for debugging, flip (mirror) and resize
    cv2.imwrite('/Users/henryschnieders/desktop/Camera_Feed.jpg', frame)
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (300, 300))
    frames.append(frame)
    frame_no = len(frames)
    
    bbox = find_face(frames[-1], detector)
    if bbox is None:
        print("No face detected.")
        continue

    # bbox structure: [x, y, w, h]
    h, w = bbox[3], bbox[2]
    
    if frame_no % frame_interval == 0:
        # Get a region of interest (ROI) from a previous frame for template matching.
        if frame_no > frame_interval:
            roi = frames[-(1 + frame_interval)][bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
        else:
            roi = frames[-1][bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]

        # Update the face bounding box using template matching.
        bbox = update_face(frames[-1], roi, h, w)
        
        if bbox is not None:
            # For all frames in the current interval, extract the 5 regions.
            for i in range(frame_no - frame_interval, frame_no):
                # Using our corrected extraction functions:
                left_region     = left_cheek(frames[i], bbox)
                right_region    = right_cheek(frames[i], bbox)
                chin_region     = chin(frames[i], bbox)
                midleft_region  = midleft_cheek(frames[i], bbox)
                midright_region = midright_cheek(frames[i], bbox)
                
                # Convert the extracted regions to grayscale and store them.
                regions_gray = [cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                                for region in [left_region, right_region, chin_region, midleft_region, midright_region]]
                face_area.append(regions_gray)
            
            # For visualization, draw rectangles showing the extracted regions on recent frames.
            plot_loc = plot_locs_cv2(bbox)
            for fram in frames[-frame_interval:]:
                for loc in plot_loc:
                    cv2.rectangle(fram, loc[0], loc[1], (0, 255, 0), thickness=1)
                cv2.imshow('Camera Feed', fram)
            
            # Process the collected regions to compute the ROI mean, filter the signal, and measure BPM.
            roi_mean = compute_roi_mean(face_area)
            roi_mean_filtered = butter_bandpass_filter(roi_mean, 1, 6, sampling_rate, order=5)
            heart_rate_signal = butter_bandpass_filter(roi_mean_filtered, lowcut_heart, highcut_heart, sampling_rate)
            bpm = bpm_measure(roi_mean, fps)
            set_title(bpm, fig)
            update_graph(signal=heart_rate_signal, fig=fig, ax=ax, past_time=past_time)
            
            past_time = time.time()
        else:
            for i in range(max(0, frame_no - frame_interval + 1), frame_no + 1):
                print(f'No face detected in frame {i}, using previous face area.')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Optionally save the extracted regions data for offline analysis
with open('/Users/henryschnieders/desktop/face_regions.pkl', 'wb') as f:
    pickle.dump(face_area, f)
    
cap.release()
cv2.destroyAllWindows()
