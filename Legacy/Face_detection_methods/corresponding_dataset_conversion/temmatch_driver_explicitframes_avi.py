import cv2
import numpy as np

def get_frames(video_path, desired_frames=1000):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize an empty list to store frames
    frames = []

    # Frame processing loop
    frame_count = 0
    while cap.isOpened() and frame_count < desired_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add the frame to the list
        frames.append(frame)
        
        frame_count += 1

    # Release the video capture object
    cap.release()

    return frames, fps, frames[0].shape[1], frames[0].shape[0]

def find_face(frame, detector):
    _, faces = detector.detect(frame)
    
    if faces is not None:
        return faces[0][:4].astype(int)
    return None

def main(frames, roi, h, w):
    padding = 0.1
    area = []
    for frame in frames[1:]:
        # Apply template Matching
        res = cv2.matchTemplate(frame, roi, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)

        # Draw a rectangle around the matched region
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        area.append(np.array(cv2.cvtColor(frame[(top_left[1]-int(w*padding)):(bottom_right[1]+int(w*padding)), 
                                                (top_left[0]-int(h*padding)):(bottom_right[0]+int(h*padding))], cv2.COLOR_BGR2GRAY)))

        cv2.rectangle(frame, (top_left[0]-int(w*padding), top_left[1]-int(h*padding)), 
                      (bottom_right[0]+int(w*padding), bottom_right[1]+int(h*padding)), (0,0,255), 1)
        cv2.imshow('Tracking', frame)
        
        # Reassign roi to account for potential movement
        roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        out.write(frame)
    return area


#########################################################################################################
#########################################################################################################
#########################################################################################################





fnum = 1000  # Just needs to be more than the number of frames in the video




import argparse
# Fix the argument name in argparse
parser = argparse.ArgumentParser(description='Process video and convert to .npy')
parser.add_argument('--video_data', type=str, required=True, help='Path to the video file')
parser.add_argument('--vid', type=str, required=True, help='Video identifier (filename without extension)')
args = parser.parse_args()

# Correct usage of video_data
video_data = args.video_data
vid = args.vid


print(f"Processing video: {video_data} with identifier: {vid}")

# Your existing code to process video_data and vid goes here






video_data = '/Users/henryschnieders/documents/Research/My_work/Data/'+vid+'.avi'

video_data = args.video_data

output_path_vid = '/Users/henryschnieders/Documents/Research/My_work/Face_detection_methods/data_visualization/'+vid+'_tempmatch.avi'
output_path_data = '/Users/henryschnieders/Documents/Research/My_work/Data/'+vid+'_padding_avi.npy'
model_path = '/Users/henryschnieders/documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'






frames, fps, frame_width, frame_height = get_frames(video_data)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi files
out = cv2.VideoWriter(output_path_vid, fourcc, fps, (frame_width, frame_height))

detector = cv2.FaceDetectorYN_create(
    model_path,
    "", 
    (frame_width, frame_height),
    score_threshold=0.5
)

roii = find_face(frames[0], detector)

wdelta=int(frame_width*0.05)
hdelta=int(frame_height*0.05)

roi = frames[0][(roii[1]-wdelta):(roii[1] + roii[3]+wdelta), (roii[0]-hdelta):(roii[0] + roii[2]+hdelta)]

h = roi.shape[0]
w = roi.shape[1]

data = main(frames, roi, h, w)

np.save(output_path_data, data)

print(f'Initial video fps is {fps}')
cv2.destroyAllWindows()
