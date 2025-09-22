import cv2
import numpy as np
import time


# Set parameters
def get_frames(video_path, desired_frames):
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
        
        frames.append(frame)
        # Resize frame to the desired dimensions
        
        frame_count += 1

    # Release the video capture object
    cap.release()

    return frames, fps, frames[0].shape[1], frames[0].shape[0]

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

def bulk(tot_frames=1000):


    # output_path='/Users/henryschnieders/Desktop/Research/My work/From_video/face_detection_yunet.mp4'
    model_path='/Users/henryschnieders/Desktop/Research/My_work/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
    video_data='/Users/henryschnieders/Desktop/Research/My_work/Data/relax.mp4'


    frames, _, frame_width, frame_height = get_frames(video_data, tot_frames)

    lenghth=len(frames)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    # out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    detector=cv2.FaceDetectorYN_create(model_path,
                          "", 
                          (frame_width, frame_height),
                          score_threshold=0.05) 

    roi_0= find_face(frames[0], detector)

    cv2.rectangle(frames[0], (roi_0[0], roi_0[1]), (roi_0[0]+roi_0[2], roi_0[1]+roi_0[3]), (0, 255, 0), 2)
    cv2.imshow('frame', frames[0])
    cv2.waitKey(5000)
    cv2.destroyAllWindows()


    roi=frames[0][roi_0[1]:(roi_0[1]+roi_0[3]), roi_0[0]:(roi_0[0]+roi_0[2])]

    
    face_area=np.zeros(499,dtype=np.ndarray)

    pad=50
    for fnum in range(0, len(frames)):
        
        frame = frames[fnum]
    
        # Apply template matching within the detected face region
        res = cv2.matchTemplate(frame[(roi_0[1]-pad//2):(roi_0[1]+roi_0[3]+pad//2), (roi_0[0]-pad//2):(roi_0[0]+roi_0[2]+pad//2)], roi, cv2.TM_CCOEFF_NORMED)

        _, _, _, max_loc = cv2.minMaxLoc(res)

        # Map the max location back to the original frame
        top_left = (max_loc[0] + roi_0[0], max_loc[1] + roi_0[1])
        h, w = roi.shape[:2]  # Height and width of the ROI/template
        bottom_right = (top_left[0] + w, top_left[1] + h)



        top_left=[int(x) for x in top_left]
        bottom_right=[int(x) for x in bottom_right]

        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

        exit()


        face_area[fnum]=cv2.cvtColor(frame[top_left[1]:(top_left[1]+h),top_left[0]:(top_left[0]+w)], cv2.COLOR_BGR2GRAY)
    
    return face_area, lenghth

    #     face=cv2.cvtColor(frames[fnum][bbox[1]:(bbox[1]+bbox[3]),bbox[0]:(bbox[0]+bbox[2])], cv2.COLOR_BGR2GRAY)

    #     rect_width = bbox[2] 
    #     rect_height = bbox[3] 

    #     # List to hold the smaller arrays
    #     sub_array = np.zeros((rect_height, rect_width))
    #     # Loop through each grid cell
    #     for i in range(1,rect_width):
    #         for j in range(1,rect_height):
    #             # Calculate the start and end indices for slicing
    #             x_start = i 
    #             y_start = j 
    #             # Extract the ub-array corresponding to this grid cell
    #             sub_array[j,i] = face[y_start, x_start]
    #     face_area[fnum]=sub_array

    #         #append the list of arrays per i
        
   
    # return face_area, lenghth

if __name__=="__main__":

    vid_name='relax'

    data_output_path='/Users/henryschnieders/Desktop/Research/My_work/Real_time_recognition/All_face_sections/frequency_heatmap/Approaches/Each_pixel/Template_matching/freq_map_tempmatch/Data/'+f'{vid_name}'+'_frames_pixels.npy'

    start=time.time()
    Amplitude, lenght = bulk()


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
