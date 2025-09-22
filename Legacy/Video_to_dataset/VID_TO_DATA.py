import cv2
import numpy as np
import subprocess
import os
import pickle
import params  # Your params module

# Import parameters from your params module
DELTA = params.DELTA
CHEEK_HPAD = params.CHEEK_HPAD
CHEEK_VPAD = params.CHEEK_VPAD
CHIN_HPAD = params.CHIN_HPAD
CHIN_VPAD = params.CHIN_VPAD
MLCHEEK_HPAD = params.MLCHEEK_HPAD
MRCHEEK_HPAD = params.MRCHEEK_HPAD
MRCHEEK_VPAD = params.MRCHEEK_VPAD
MLCHEEK_VPAD = params.MLCHEEK_VPAD
FRAME_INTERVAL = params.FRAME_INTERVAL
FRAME_WIDTH = params.FRAME_WIDTH
FRAME_HEIGHT = params.FRAME_HEIGHT
TOT_FRAMES = params.TOT_FRAMES


# Initial face detection with YuNet
def find_initial_face(frame, detector):
    _, faces = detector.detect(frame)
    if faces is not None and len(faces) > 0:
        x, y, w, h = faces[0][:4].astype(int)
        return (x, y, w, h), frame[y:y + h, x:x + w]  # Return bbox and ROI
    return None, None

# Template matching to track face
def track_face(frame, template):
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    
    w, h = template.shape[1], template.shape[0]  # Width and height from template
    return (top_left[0], top_left[1], w, h)

# Extract regions using your params
def extract_regions(frame, bbox):
    regions = [
        # chin: DELTA height
        params.chin(frame, bbox),
        # Left cheek: CHEEK_HPAD and CHEEK_VPAD
        params.left_cheek(frame, bbox),
        
        # Right cheek: 1-CHEEK_HPAD and CHEEK_VPAD
        params.right_cheek(frame, bbox),
        # midleft cheek: 
        params.midleft_cheek(frame, bbox),
        # midright cheek: 
        params.midright_cheek(frame, bbox)
    ]
    gray_regions = [cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) for region in regions if region.size > 0]
    return gray_regions

# Process frames with template matching
def process_frames(frames, detector, initial_hr, final_hr, out):
    all_regions = []
    
    # Initial face detection on first frame
    initial_bbox, template = find_initial_face(frames[0], detector)
    if initial_bbox is None or template is None:
        raise ValueError("No face detected in the first frame!")
    
    # Convert template to grayscale for matching
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    for i, frame in enumerate(frames):
        # Flip frame vertically (as requested)
        frame = cv2.flip(frame, 1)
        
        # Convert frame to grayscale for template matching
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Track face using template matching
        if i == 0:
            bbox = initial_bbox  # Use AI detection for first frame
        else:
            bbox = track_face(gray_frame, template)  # Use grayscale frame
        
        x, y, w, h = bbox
        # Extract regions (use original BGR frame for regions)
        regions = extract_regions(frame, bbox)
        
        if regions:
            # Draw rectangle around the tracked face
            cv2.rectangle(frames[i], (bbox[0], bbox[1]), (bbox[1] + bbox[3], bbox[0] + bbox[2]), (0, 255, 0), 2)
            
            # Draw rectangles around the 5 regions
            region_coords = [
                [(bbox[1]+int(bbox[3]*CHEEK_VPAD)-DELTA), (bbox[0]+int(bbox[2]*(CHEEK_HPAD))-DELTA), (bbox[1]+int(bbox[3]*CHEEK_VPAD)+DELTA), (bbox[0]+int(bbox[2]*(CHEEK_HPAD))+DELTA)],  # Left cheek
                [(bbox[1]+int(bbox[3]*CHEEK_VPAD)-DELTA), (bbox[0]+int(bbox[2]*(1-CHEEK_HPAD))-DELTA), (bbox[1]+int(bbox[3]*CHEEK_VPAD)+DELTA), (bbox[0]+int(bbox[2]*(1-CHEEK_HPAD))+DELTA)],  # Right cheek
                [(bbox[1]+int(bbox[3]*CHIN_VPAD)-DELTA), (bbox[0]+int(bbox[2]*CHIN_HPAD)-DELTA), (bbox[1]+int(bbox[3]*CHIN_VPAD)+DELTA), (bbox[0]+int(bbox[2]*CHIN_HPAD)+DELTA)],  # Chin
                [(bbox[1]+int(bbox[3]*MLCHEEK_VPAD)-DELTA), (bbox[0]+int(bbox[2]*(MLCHEEK_HPAD))-DELTA), (bbox[1]+int(bbox[3]*MLCHEEK_VPAD)+DELTA), (bbox[0]+int(bbox[2]*(MLCHEEK_HPAD))+DELTA)],  # Midleft cheek]
                [(bbox[1]+int(bbox[3]*MRCHEEK_VPAD)-DELTA), (bbox[0]+int(bbox[2]*(MRCHEEK_HPAD))-DELTA), (bbox[1]+int(bbox[3]*MRCHEEK_VPAD)+DELTA), (bbox[0]+int(bbox[2]*(MRCHEEK_HPAD))+DELTA)]   # Midright cheek
            ]
            for (rx, ry, rw, rh) in region_coords:
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 1)
            
            # Store the 5 grayscale regions
            all_regions.append(regions)
        
        # Overlay heart rate
        if i == 0:
            cv2.putText(frame, f"Initial HR: {initial_hr} bpm", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif i == len(frames) - 1:
            cv2.putText(frame, f"Final HR: {final_hr} bpm", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Write flipped frame to video
        out.write(frame)
    
    return all_regions

# Main script
if __name__ == "__main__":
    # Recording parameters
    duration = 10  # 10 seconds
    fps = 30
    width = 1760
    height = 1328
    base_path = "/Users/henryschnieders/Documents/Research/My_Data"

    # User inputs
    video_name = input("Enter video save name (without extension): ")
    initial_hr = int(input("Enter initial heart rate (bpm): "))
    
    output_video = f"{base_path}/{video_name}.mp4"
    output_data = f"{base_path}/{video_name}_regions.npy"

    # FFmpeg command for 10-bit capture
    ffmpeg_command = [
        'ffmpeg',
        '-f', 'avfoundation',
        '-framerate', str(fps),
        '-i', '0',
        '-pix_fmt', 'yuv420p10le',
        '-c:v', 'libx264',
        '-profile:v', 'high10',
        '-r', str(fps),
        '-s', f'{width}x{height}',
        '-t', str(duration),
        '-y',  # Overwrite output file if it exists
        output_video
    ]

    # Record video
    print("Recording 10-bit video for 10 seconds...")
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("Error during recording:")
        print(stderr.decode())
        exit(1)

    # Load recorded video frames
    cap = cv2.VideoCapture(output_video)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Set up video writer for output with overlays
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video.replace(".mp4", "_processed.mp4"), fourcc, fps, (width, height))

    # Load face detector
    model_path = "/Users/henryschnieders/Documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    detector = cv2.FaceDetectorYN_create(model_path,
                                     "",
                                    (width, height),
                                    score_threshold=0.5)

    # Get final heart rate
    final_hr = int(input("Enter final heart rate (bpm): "))

    # Process frames and extract regions
    print("Processing frames with template matching...")
    all_regions = process_frames(frames, detector, initial_hr, final_hr, out)

    # Save regions to .npy file using pickle
    with open(output_data, 'wb') as f:
        pickle.dump(all_regions, f)
    print(f"Saved region data to {output_data}")

    # Clean up
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved as {output_video.replace('.mp4', '_processed.mp4')}")