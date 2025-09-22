import cv2
import numpy as np


"""
This script demonstrates how to use template matching to track a face in a video.
The script reads a video file, extracts frames from the video, and applies template matching to each frame.
The script then displays the frames with a bounding box around the detected face.

The script first uses YuNet face detector to detect the face in the first frame of the video.
The detected face is then used as the template for template matching in the subsequent frames.
The roi is updated in each frame to account for the movement of the face.
    
"""
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
        
        # Resize frame to the desired dimensions
        #frame_resized = cv2.resize(frame, (desired_dimensions, desired_dimensions))

        
        
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
    area=[]
    for frame in frames[1:]:
        # Apply template Matching
        res = cv2.matchTemplate(frame, roi, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)

        # Draw a rectangle around the matched region
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)


        area+=[frame[(top_left[1]-int(w*padding)):(bottom_right[1]+int(w*padding)), (top_left[0]-int(h*padding)):(bottom_right[0]+int(h*padding)),:]]
       # area+=[cv2.cvtColor(frame[(top_left[1]):(bottom_right[1]), (top_left[0]):(bottom_right[0])],cv2.COLOR_BGR2GRAY)]

        cv2.rectangle(frame, (top_left[0]-int(w*padding), top_left[1]-int(h*padding)), (bottom_right[0]+int(w*padding), bottom_right[1]+int(h*padding)), (0,0,255), 1)
        cv2.imshow('Tracking', frame)
        
        #reassign roi to account for potential movement
        roi=frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        out.write(frame)
    return area



import argparse

# Argument parser to get video_data and vid from the command line
# Fix the argument name in argparse
parser = argparse.ArgumentParser(description='Process video and convert to .npy')
parser.add_argument('--video_data', type=str, required=True, help='Path to the video file')
parser.add_argument('--vid', type=str, required=True, help='Video identifier (filename without extension)')
args = parser.parse_args()

# Correct usage of video_data

vid = args.vid


# Your existing code to process video_data and vid goes here


fnum=1000 #just needs to be more than the number of frames in the video


#video_data='/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Results_10_21/Videos/trimmed/mp4/100_light.mp4'

video_data=args.video_data


#print(f"Processing video: {video_data} with identifier: {vid}")

#vid='100_light'

output_path_vid='/Users/henryschnieders/Documents/Research/My_work/Face_detection_methods/data_visualization/'+vid+'_tempmatch.mp4'
output_path_data='/Users/henryschnieders/Documents/Research/My_work/Data/'+vid+'_padding_color_mp4.npy'
model_path='/Users/henryschnieders/documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'

frames, fps, frame_width, frame_height = get_frames(video_data)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_path_vid, fourcc, fps, (frame_width, frame_height))

# h=50
# spoint=500
# lpoint=825

for frame in frames:
    if frame.shape != (frame_height, frame_width, 3):
        print(frame.shape)




detector=cv2.FaceDetectorYN_create(model_path,
                        "", 
                        (frame_width, frame_height),
                        score_threshold=0.1) 


frames=frames[1:]

roii = find_face(frames[0], detector)

print(roii)




wdelta=int(frame_width*0.05)
hdelta=int(frame_height*0.05)

roi = frames[0][(roii[1]-wdelta):(roii[1] + roii[3]+wdelta), (roii[0]-hdelta):(roii[0] + roii[2]+hdelta),:]
h=roi.shape[0]
w=roi.shape[1]

data=main(frames, roi, h, w)

np.save(output_path_data, data)

print(f'Initial video fps is {fps}')
cv2.destroyAllWindows()
