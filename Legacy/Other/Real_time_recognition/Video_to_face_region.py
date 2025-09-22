import cv2
import numpy as np
import time

# Set parameters

def get_frames(video_path, desired_frames, desired_dimensions):
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
        frame_resized = cv2.resize(frame, (desired_dimensions, desired_dimensions))

        
        # Add the frame to the list
        frames.append(frame_resized)
        
        frame_count += 1

    # Release the video capture object
    cap.release()

    return frames, fps

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

def bulk():

    #frame parameters
    tot_frames=900
    frame_width=300
    frame_height=frame_width
    frame_interval=15

    #face detection parameters
    RADIUS=3
    COLOR=(0,0,255) #green
    THICKNESS=-1 #filled circle

    #cheek detection parameters-distance from vertical edge of detection:
    CHEEK_HPAD=1/6

    #cheek detection parameters-fractional distance from top edge of detection:
    CHEEK_VPAD=1/2
    #chin location-horizontal
    CHIN_HPAD=1/2
    
    #chin location-vertical (fraction from the top)
    CHIN_VPAD=7/8


    output_path='/Users/henryschnieders/Desktop/Research/My_work/Real_time_recognition/All_face_sections/amplitude heatmap/Results/face_detection_yunet_noface.mp4'
    model_path='/Users/henryschnieders/Desktop/Research/My_work/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
    video_data='/Users/henryschnieders/Desktop/Research/My_work/Data/inorganic.MOV'

    frames, fps = get_frames(video_data, tot_frames, frame_width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    detector=cv2.FaceDetectorYN_create(model_path,
                          "", 
                          (frame_width, frame_height),
                          score_threshold=0.005) 
    
    face_area=[]

    for frame_idx in range(0, len(frames), frame_interval):
        
        frame = frames[frame_idx]
        face = find_face(frame, detector)


        if face is not None:
            bbox = face

            #plot on all frames
            for i in range(max(0, frame_idx - frame_interval + 1), frame_idx + 1):

                face_area.append(frames[i][bbox[1]:bbox[3], bbox[0]:bbox[2]])
        
        else:
            for i in range(max(0, frame_idx - frame_interval + 1), frame_idx + 1):

                face_area.append(face_area[-1])
                print(f'No face detected in frame {i}, adding the previous face area to the list.')
   
    return face_area

if __name__=="__main__":

    start=time.time()

    Amplitude_=bulk()

    end=time.time()
    print("Time taken:", end-start)
    

        






# '/Users/henryschnieders/Desktop/Research/My work/Data/video.MOV'







# print("Shape of the saved array:", frames_array.shape)
