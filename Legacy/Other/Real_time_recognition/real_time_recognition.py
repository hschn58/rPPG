import params
import video_to_dataset
from signal_processing import plot_locs_cv2, butter_bandpass_filter
from methods import compute_roi_mean

import cv2
import numpy as np

# roi_mean_ = butter_bandpass_filter(roi_mean, 1, 6, sampling_rate, order=5)
#get already-specified parameters 

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

sampling_rate=params.sampling_rate
lowcut_heart=params.lowcut_heart
highcut_heart=params.highcut_heart

graph_height=params.graph_height
graph_width=params.graph_width

graph = 255*np.ones((graph_height, graph_width, 3), dtype=np.uint8)
heart_rate_data = []

def update_graph(signal):
    global graph, heart_rate_data
    # Shift the graph to the left
    graph[:, :-1] = graph[:, 1:]

    heart_data_length=len(heart_rate_data)
    # Append the new signal value
    heart_rate_data+=list(signal)
    
    delt=heart_data_length-graph_width
    if delt>0:
        for _ in range(delt):
            heart_rate_data.pop(0)

    # Draw the updated graph
    for i in range(max(0,heart_data_length-frame_interval), heart_data_length):
        cv2.line(graph, (i-1, graph_height - int(heart_rate_data[i-1] * graph_height)),
                 (i, graph_height - int(heart_rate_data[i] * graph_height)), (255, 0, 0), thickness=1)

    return graph

def find_face(frame, detector):

    _, faces = detector.detect(frame)
    
    if faces is not None:
        return faces[0][:4].astype(int)
    return None

def average(list_of_arrays):
    return sum(list_of_arrays)/len(list_of_arrays)


model_path='/Users/henryschnieders/Documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
detector=cv2.FaceDetectorYN_create(model_path,
                          "", 
                          (frame_width, frame_height),
                          score_threshold=0.5) 



# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

ret, frame = cap.read() #ret is bool, (retrieved)
if not ret:
    print("Error: Failed to capture image.")
    exit()
# Loop to continuously capture frames

########################################################################


frames=[]
while True:
    

    #I can modify this to only store the last frame_interval frames at a time

    _,frame=cap.read()
    frame = cv2.flip(frame, 1) #normal camera feed is mirrored
    frame = cv2.resize(frame, (300, 300))
    frames.append(frame)

    frame_no=len(frames)
    face_area=[]
    if frame_no%frame_interval==0:
       
        bbox = find_face(frames[-1], detector)
            
        if bbox is not None:
            for i in range(frame_no - frame_interval, frame_no):


            ##############################################################################################################
            #get the current corners of each rectangle on the current frame for left cheek, right cheek, chin, midleft cheek, midright cheek
                left_cheek=video_to_dataset.left_cheek(frames[i], bbox)
                right_cheek=video_to_dataset.right_cheek(frames[i], bbox)
                chin=video_to_dataset.chin(frames[i], bbox)
                midleft_cheek=video_to_dataset.midleft_cheek(frames[i], bbox)
                midright_cheek=video_to_dataset.midright_cheek(frames[i], bbox)

                face_area+=[[cv2.cvtColor(area, cv2.COLOR_BGR2GRAY) for area in [left_cheek,right_cheek,chin, midleft_cheek, midright_cheek]]]
                face_area[-1]=average(face_area[-1]) #append the list of arrays per i

        
            ##############################################################################################################
            #camera plotting
            plot_loc=plot_locs_cv2(bbox)
            for fram in frames[i-(frame_interval-1):i]:
                for loc in plot_loc:
                    cv2.rectangle(fram, loc[0], loc[1], (0, 255, 0), thickness=1)
                cv2.imshow('Camera Feed', fram)
    
            ##############################################################################################################
            #signal processing  
            roi_mean=compute_roi_mean(face_area)
            roi_mean_ = butter_bandpass_filter(roi_mean, 1, 6, sampling_rate, order=5)
            heart_rate_signal = butter_bandpass_filter(roi_mean_, lowcut_heart, highcut_heart, sampling_rate)

            update_graph(heart_rate_signal)
            cv2.imshow(f'Heart Rate Signal', graph)

        else:
            for i in range(max(0, frame_no - frame_interval + 1), frame_no + 1):
                
                # out.write(frames[i])
                #face_area.append(face_area[-1])
                print(f'No face detected in frame {i}, adding the previous face area to the list.')
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Break the loop when 'q' key is pressed

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
