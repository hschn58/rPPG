import cv2
import numpy as np
import time



score_threshold = 0.1

# Set parameters

def get_frames(video_path, desired_frames):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    return frames, fps, frame_width, frame_height

def find_face(frame, detector):

    _, faces = detector.detect(frame)
    
    if faces is not None:
        return faces[0][:4].astype(int)
    return None
        



# def get_IB_QB(frames,target):

#     #get IB, QB data from the area in each frame that is not in target (face) region

#     padding=10



#     n_frames = len(frames) // 2
#     return I_1d, Q_1d

def main():

    tot_frames=2000
    frame_interval=1

    output_path='/Users/henryschnieders/Desktop/face_detection_yunet_relax.mp4'
    model_path='/Users/henryschnieders/Documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
    video_data='/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Results_10_21/Videos/trimmed/mp4/100_light.mp4'

    frames, fps, frame_width, frame_height = get_frames(video_data, tot_frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    detector=cv2.FaceDetectorYN_create(model_path,
                          "", 
                          (frame_width, frame_height),
                          score_threshold=score_threshold) 
    
    for frame_idx in range(0, len(frames), frame_interval):
        frame = frames[frame_idx]
        face = find_face(frame, detector)

        if face is not None:
            bbox = face

            #plot on all frames
            for i in range(max(0, frame_idx - frame_interval + 1), frame_idx + 1):
                cv2.rectangle(frames[i], (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

                # cv2.circle(frames[i], center=(bbox[0]+int(bbox[2]*(1/6)), bbox[1]+int(bbox[3]/2)), radius=3, color=(0,0,255), thickness=-1)
                # cv2.circle(frames[i], center=(bbox[0]+int(bbox[2]*(5/6)), bbox[1]+int(bbox[3]/2)), radius=3, color=(0,0,255), thickness=-1)

                # cv2.circle(frames[i], center=(bbox[0]+int(bbox[2]/2), bbox[1]+int(bbox[3]*(7/8))), radius=3, color=(0,0,255), thickness=-1)
                
                out.write(frames[i])
        else:
            for i in range(max(0, frame_idx - frame_interval + 1), frame_idx + 1):
                out.write(frames[i])
        
if __name__=="__main__":

    start=time.time()
    main()
    end=time.time()
    print("Time taken:", end-start)
    

        






# '/Users/henryschnieders/Desktop/Research/My work/Data/video.MOV'







# print("Shape of the saved array:", frames_array.shape)
