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






###############################################################################################################################################

#define face locations as functions for later easy access

def left_cheek(frame, bbox, DELTA=DELTA, CHEEK_HPAD=CHEEK_HPAD, CHEEK_VPAD=CHEEK_VPAD):
    return frame[(bbox[0]+int(bbox[2]*(CHEEK_HPAD))-DELTA):(bbox[0]+int(bbox[2]*(CHEEK_HPAD))+DELTA), (bbox[1]+int(bbox[3]*CHEEK_VPAD)-DELTA):(bbox[1]+int(bbox[3]*CHEEK_VPAD)+DELTA)] #left cheek

def right_cheek(frame, bbox, DELTA=DELTA, CHEEK_HPAD=CHEEK_HPAD, CHEEK_VPAD=CHEEK_VPAD):
    return frame[(bbox[0]+int(bbox[2]*(1-CHEEK_HPAD))-DELTA):(bbox[0]+int(bbox[2]*(1-CHEEK_HPAD))+DELTA), (bbox[1]+int(bbox[3]*CHEEK_VPAD)-DELTA):(bbox[1]+int(bbox[3]*CHEEK_VPAD)+DELTA)] #right cheek

def chin(frame, bbox, DELTA=DELTA, CHIN_HPAD=CHIN_HPAD, CHIN_VPAD=CHIN_VPAD):
    return frame[(bbox[0]+int(bbox[2]*CHIN_HPAD)-DELTA):(bbox[0]+int(bbox[2]*CHIN_HPAD)+DELTA), (bbox[1]+int(bbox[3]*CHIN_VPAD)-DELTA):(bbox[1]+int(bbox[3]*CHIN_VPAD)+DELTA)] #chin

def midleft_cheek(frame, bbox, DELTA=DELTA, MLCHEEK_HPAD=MLCHEEK_HPAD, MLCHEEK_VPAD=MLCHEEK_VPAD):
    return frame[(bbox[0]+int(bbox[2]*(MLCHEEK_HPAD))-DELTA):(bbox[0]+int(bbox[2]*(MLCHEEK_HPAD))+DELTA), (bbox[1]+int(bbox[3]*MLCHEEK_VPAD)-DELTA):(bbox[1]+int(bbox[3]*MLCHEEK_VPAD)+DELTA)] #left middle cheek

def midright_cheek(frame, bbox, DELTA=DELTA, MRCHEEK_HPAD=MRCHEEK_HPAD, MRCHEEK_VPAD=MRCHEEK_VPAD):
    return frame[(bbox[0]+int(bbox[2]*(MRCHEEK_HPAD))-DELTA):(bbox[0]+int(bbox[2]*(MRCHEEK_HPAD))+DELTA), (bbox[1]+int(bbox[3]*MRCHEEK_VPAD)-DELTA):(bbox[1]+int(bbox[3]*MRCHEEK_VPAD)+DELTA)] #right middle cheek

################################################################################################################################################



def bulk(tot_frames, frame_width, frame_height, frame_interval, DELTA, CHEEK_HPAD, CHEEK_VPAD, CHIN_HPAD, CHIN_VPAD, MLCHEEK_HPAD, MRCHEEK_HPAD, MRCHEEK_VPAD, MLCHEEK_VPAD):

    # output_path='/Users/henryschnieders/Desktop/Research/My work/From_video/face_detection_yunet.mp4'
    model_path='/Users/henryschnieders/Desktop/Research/My_work/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
    video_data='/Users/henryschnieders/Desktop/Research/My_work/Data/sporting 1.mp4'
    

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
    
    face_area=[]
    area_locations=[]
    for frame_idx in range(0, len(frames), frame_interval):
        
        frame = frames[frame_idx]
        face = find_face(frame, detector)


        if face is not None:
            bbox = face
            area_locations.append(bbox)

            #plot on all frames
            for i in range(max(0, frame_idx - frame_interval + 1), frame_idx + 1):

                # cv2.rectangle(frames[i], (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                # cv2.circle(frames[i], center=(bbox[0]+int(bbox[2]*(CHEEK_HPAD)), bbox[1]+int(bbox[3]*CHEEK_VPAD)), radius=RADIUS, color=COLOR, thickness=THICKNESS)
                # cv2.circle(frames[i], center=(bbox[0]+int(bbox[2]*(1-CHEEK_HPAD)), bbox[1]+int(bbox[3]*CHEEK_VPAD)), radius=RADIUS, color=COLOR, thickness=THICKNESS)

                # cv2.circle(frames[i], center=(bbox[0]+int(bbox[2]*CHIN_HPAD), bbox[1]+int(bbox[3]*CHIN_VPAD)), radius=RADIUS, color=COLOR, thickness=THICKNESS)
                
                # out.write(frames[i])

                
                left_cheek_area=left_cheek(frames[i], bbox)
                right_cheek_area=right_cheek(frames[i], bbox)
                chin_area=chin(frames[i], bbox)
                midleft_cheek_area=midleft_cheek(frames[i], bbox)
                midright_cheek_area=midright_cheek(frames[i], bbox)
                

                face_area+=[[cv2.cvtColor(area, cv2.COLOR_BGR2GRAY) for area in [left_cheek_area,right_cheek_area,chin_area, midleft_cheek_area, midright_cheek_area]]] #append the list of arrays per i
        
        else:
            for i in range(max(0, frame_idx - frame_interval + 1), frame_idx + 1):
                
                # out.write(frames[i])
                face_area.append(face_area[-1])
                print(f'No face detected in frame {i}, adding the previous face area to the list.')
   
    return face_area, lenghth, frames[-1], area_locations[-1]

if __name__=="__main__":

    vid_name='sporting_1'

    data_output_path='/Users/henryschnieders/Desktop/Research/My_work/Data/'+f'{vid_name}'+'_frames.npy'
    ref_frame_output_path='/Users/henryschnieders/Desktop/Research/My_work/Data/'+f'{vid_name}'+'_frames_ref_frame.npy'
    area_locations_path='/Users/henryschnieders/Desktop/Research/My_work/Data/'+f'{vid_name}'+'_frames_area_locations.npy'

    start=time.time()
    Amplitude, lenght, ref_frame, area_locations = bulk(tot_frames=TOT_FRAMES, 
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

    optimal_height, optimal_width = find_dimensions(Amplitude)

    resized_frames = [[resize_frame(sect[i], optimal_height[i], optimal_width[i]) for i in range(5)] for sect in Amplitude]

    final_array = np.array(resized_frames)

    np.save(data_output_path, final_array)
    np.save(ref_frame_output_path, ref_frame)
    np.save(area_locations_path, area_locations)

    end=time.time()
    print("Time taken:", end-start)
    

        






# '/Users/henryschnieders/Desktop/Research/My work/Data/video.MOV'







# print("Shape of the saved array:", frames_array.shape)
