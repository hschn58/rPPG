import params
import cv2
import numpy as np
import time



DELTA=params.DELTA
CHEEK_HPAD=params.CHEEK_HPAD
CHEEK_VPAD=params.CHEEK_VPAD
CHIN_HPAD=params.CHIN_HPAD
CHIN_VPAD=params.CHIN_VPAD
MLCHEEK_HPAD=params.MLCHEEK_HPAD
MRCHEEK_HPAD=params.MRCHEEK_HPAD
MRCHEEK_VPAD=params.MRCHEEK_VPAD
MLCHEEK_VPAD=params.MLCHEEK_VPAD
frame_interval=params.FRAME_INTERVAL
frame_width=params.FRAME_WIDTH
frame_height=params.FRAME_HEIGHT
TOT_FRAMES=params.TOT_FRAMES
X_SIZE=params.X_SIZE
Y_SIZE=params.Y_SIZE




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

def find_dimensions(frames):
    # Find the maximum width and height
    min_heights=[]
    min_widths=[]

    for area_num in range(len(frames[0])):
        min_height = min([sect[area_num].shape[0] for sect in frames])
        min_width = min([sect[area_num].shape[1] for sect in frames])
        min_heights.append(min_height)
        min_widths.append(min_width)

    return min_heights, min_widths

def resize_frame(frame, target_height, target_width):
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

def bulk(tot_frames, frame_width, frame_height, frame_interval, DELTA, CHEEK_HPAD, CHEEK_VPAD, CHIN_HPAD, CHIN_VPAD, MLCHEEK_HPAD, MRCHEEK_HPAD, MRCHEEK_VPAD, MLCHEEK_VPAD, X_SIZE=X_SIZE, Y_SIZE=Y_SIZE):

    # output_path='/Users/henryschnieders/Desktop/Research/My work/From_video/face_detection_yunet.mp4'
    model_path='/Users/henryschnieders/Desktop/Research/My_work/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
    video_data='/Users/henryschnieders/Desktop/Research/My_work/Data/relax.mp4'
    

    # frame parameters
    tot_frames=tot_frames
    frame_width=frame_width
    frame_height=frame_width
    frame_interval=frame_interval
    DELTA=DELTA

    # #face detection parameters
    # RADIUS=3
    # COLOR=(0,0,255) #green
    # THICKNESS=-1 #filled circle

    # #cheek detection parameters-distance from vertical edge of detection:
    CHEEK_HPAD=CHEEK_HPAD
    # #cheek detection parameters-fractional distance from top edge of detection:
    CHEEK_VPAD=CHEEK_VPAD

    # #middle left cheek detection parameters-distance from vertical edge of detection:
    MLCHEEK_HPAD=MLCHEEK_HPAD
    # #middle right cheek detection parameters-fractional distance from top edge of detection:
    MRCHEEK_VPAD=MRCHEEK_VPAD

    # #chin location-horizontal
    CHIN_HPAD=CHIN_HPAD
    # #chin location-vertical (fraction from the top)
    CHIN_VPAD=CHIN_VPAD

    

    frames, fps = get_frames(video_data, tot_frames, frame_width)

    lenghth=len(frames)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    # out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    detector=cv2.FaceDetectorYN_create(model_path,
                          "", 
                          (frame_width, frame_height),
                          score_threshold=0.5) 
    
    face_area=np.zeros(499,dtype=np.ndarray)

    for frame_idx in range(0, len(frames), frame_interval):
        
        frame = frames[frame_idx]
        bbox = find_face(frame, detector)
        bbox=[int(coord) for coord in bbox[:4]]

        if bbox is not None:
            

            #plot on all frames
            
            for fnum in range(max(0, frame_idx - frame_interval+1), frame_idx+1):

                # cv2.rectangle(frames[i], (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                # cv2.circle(frames[i], center=(bbox[0]+int(bbox[2]*(CHEEK_HPAD)), bbox[1]+int(bbox[3]*CHEEK_VPAD)), radius=RADIUS, color=COLOR, thickness=THICKNESS)
                # cv2.circle(frames[i], center=(bbox[0]+int(bbox[2]*(1-CHEEK_HPAD)), bbox[1]+int(bbox[3]*CHEEK_VPAD)), radius=RADIUS, color=COLOR, thickness=THICKNESS)

                # cv2.circle(frames[i], center=(bbox[0]+int(bbox[2]*CHIN_HPAD), bbox[1]+int(bbox[3]*CHIN_VPAD)), radius=RADIUS, color=COLOR, thickness=THICKNESS)
                
                face=cv2.cvtColor(frames[fnum][bbox[1]:(bbox[1]+bbox[3]),bbox[0]:(bbox[0]+bbox[2])], cv2.COLOR_BGR2GRAY)


                rect_width = bbox[2] 
                rect_height = bbox[3] 

                # List to hold the smaller arrays
                sub_array = np.zeros((rect_height, rect_width))
                # Loop through each grid cell
                for i in range(1,rect_width):
                    for j in range(1,rect_height):
                        # Calculate the start and end indices for slicing
                        x_start = i 
                        y_start = j 
                        # Extract the ub-array corresponding to this grid cell
                        sub_array[j,i] = face[y_start, x_start]
                face_area[fnum]=sub_array

                 #append the list of arrays per i
                
        else:
            for fnum in range(max(0, frame_idx - frame_interval + 1), frame_idx + 1):
                
                # out.write(frames[i])
                face_area[fnum]=face_area[frame_idx-frame_interval]
                print(f'No face detected in frame {i}, adding the previous face area to the list.')
   
    return face_area, lenghth

if __name__=="__main__":

    vid_name='relax'

    data_output_path='/Users/henryschnieders/Desktop/Research/My_work/Real_time_recognition/All_face_sections/Approaches/Each_pixel/Data/'+f'{vid_name}'+'_frames_pixels.npy'

    start=time.time()
    Amplitude, lenght = bulk(tot_frames=TOT_FRAMES, 
                            frame_width=frame_width, 
                            frame_height=frame_height, 
                            frame_interval=frame_interval, 
                            DELTA=DELTA, 
                            CHEEK_HPAD=CHEEK_HPAD, 
                            CHEEK_VPAD=CHEEK_VPAD, 
                            CHIN_HPAD=CHIN_HPAD, 
                            CHIN_VPAD=CHIN_VPAD,
                            MLCHEEK_HPAD=MLCHEEK_HPAD,
                            MRCHEEK_HPAD=MRCHEEK_HPAD,
                            MRCHEEK_VPAD=MRCHEEK_VPAD,
                            MLCHEEK_VPAD=MLCHEEK_VPAD)


    print("Number of frames:", lenght)

    # import pickle
    # # Serialize the list using pickle and save it as a .npy file
    # with open(data_output_path, 'wb') as f:
    #     np.save(f, pickle.dumps(Amplitude))
   
    np.save(data_output_path, Amplitude)
    
    end=time.time()
    print("Time taken:", end-start)
    

        






# '/Users/henryschnieders/Desktop/Research/My work/Data/video.MOV'







# print("Shape of the saved array:", frames_array.shape)
