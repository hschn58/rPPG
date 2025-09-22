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

def main(frames):
    area=[]
    for frame in frames:

        region=find_face(frame, detector)[:4]

        if region is not None:
            top_left = (region[0], region[1])
            bottom_right = (region[0] + region[2], region[1] + region[3])
            area+=[cv2.cvtColor(frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]],cv2.COLOR_BGR2GRAY)]

        else:
            area+=[area[-1]]

        cv2.rectangle(frame, top_left, bottom_right, (0,0,255), 2)
        cv2.imshow('Tracking', frame)
        
        out.write(frame)

     # Ensure all frames have the same size
    min_x = min([frame.shape[1] for frame in area])
    min_y = min([frame.shape[0] for frame in area])
    area = [cv2.resize(frame, (min_x, min_y)) for frame in area]

    return area

if __name__=="__main__":


    fnum=1000
    video_data='/Users/henryschnieders/Documents/Research/My_work/Data/relax.mp4'
    output_path_vid='/Users/henryschnieders/Documents/Research/My_work/Real_time_recognition/All_face_sections/amplitude_heatmap/Each_pixel/Pixel_matching_methods/Template_Matching/Results/relax_facematch.mp4'
    output_path_data='/Users/henryschnieders/Documents/Research/My_work/Real_time_recognition/All_face_sections/amplitude_heatmap/Each_pixel/Pixel_matching_methods/Template_Matching/Data/relax_facematch.npy'
    model_path='/Users/henryschnieders/Documents/Research/My_work/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'

    frames, fps, frame_width, frame_height = get_frames(video_data)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path_vid, fourcc, fps, (frame_width, frame_height))
    
    # h=50
    # spoint=500
    # lpoint=825

    detector=cv2.FaceDetectorYN_create(model_path,
                          "", 
                          (frame_width, frame_height),
                          score_threshold=0.5) 
    
    data=main(frames)

    np.save(output_path_data, data)
    cv2.destroyAllWindows()
