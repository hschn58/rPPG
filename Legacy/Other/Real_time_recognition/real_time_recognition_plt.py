import params
import video_to_dataset
from signal_processing import plot_locs_cv2, butter_bandpass_filter, bpm_measure
from methods import compute_roi_mean

import cv2
import numpy as np
import time

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
time_data=[]

import matplotlib.pyplot as plt
plt.ion()
graph_width=75

with plt.ion():
    fig,ax=plt.subplots()
    ax.set_xlim(0, graph_width)

    #tentative settings
    ax.set_ylim(-1.5, 1.5)

    line, = ax.plot(time_data, heart_rate_data) 

    plt.xlabel('Time (s)')
    fig.canvas.manager.set_window_title('Real-Time Heart Rate Signal')

def set_title(bpm,fig):
    fig.canvas.manager.set_window_title(f'Real-Time Heart Rate Signal: {bpm:.2f} bpm')

#start the signal figure
past_time=time.time()
reference_time=past_time

def update_graph(signal,fig,ax,past_time, graph_width=graph_width, reference_time=reference_time):
    global time_data,heart_rate_data

    cur_delt=time.time()-past_time
    rel_delt=past_time-reference_time

    time_data+=list(np.linspace(0+rel_delt,cur_delt+rel_delt, len(signal)))
    heart_rate_data+=list(signal)

    if len(time_data)>graph_width:
        time_data=time_data[-graph_width:]
        heart_rate_data=heart_rate_data[-graph_width:]
        ax.set_xlim(time_data[0], time_data[-1])
        ax.set_ylim(min(heart_rate_data), max(heart_rate_data))

    line.set_xdata(time_data)
    line.set_ydata(heart_rate_data)

    # Redraw the plot
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.plot(time_data, heart_rate_data, color='b')

def find_face(frame, detector):

    _, faces = detector.detect(frame)
    
    if faces is not None:
        return faces[0][:4].astype(int)
    return None

def average(list_of_arrays):
    return sum(list_of_arrays)/len(list_of_arrays)


model_path='/Users/henryschnieders/documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
detector=cv2.FaceDetectorYN_create(model_path,
                          "", 
                          (frame_width, frame_height),
                          score_threshold=0.5) 


# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

fps = cap.get(cv2.CAP_PROP_FPS)

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

            bpm=bpm_measure(roi_mean, fps)
            set_title(bpm,fig)
            update_graph(signal=heart_rate_signal, fig=fig, ax=ax, past_time=past_time)

            past_time=time.time()

        else:
            for i in range(max(0, frame_no - frame_interval + 1), frame_no + 1):
                
                # out.write(frames[i])
                #face_area.append(face_area[-1])
                print(f'No face detected in frame {i}, adding the previous face area to the list.')
        

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Break the loop when 'q' key is pressed


#free up resources
del frames
del face_area
del heart_rate_data
del time_data

cap.release()
cv2.destroyAllWindows()

