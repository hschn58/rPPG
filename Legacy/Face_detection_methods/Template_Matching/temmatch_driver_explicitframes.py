import cv2
import numpy as np


# h is added twice here!!

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
    for frame in frames[1:]:
        # Apply template Matching
        res = cv2.matchTemplate(frame, roi, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)

        # Draw a rectangle around the matched region
        top_left = max_loc
        bottom_right = (top_left[0] + h, top_left[1] + h)
        area+=[cv2.cvtColor(frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]],cv2.COLOR_BGR2GRAY)]

        cv2.rectangle(frame, top_left, bottom_right, (0,0,255), 2)
        cv2.imshow('Tracking', frame)
        
        out.write(frame)
    return area
if __name__=="__main__":

    fnum=1000
    video_data='/Users/henryschnieders/Desktop/Research/My_work/Data/relax.mp4'
    output_path_vid='/Users/henryschnieders/Desktop/Research/My_work/Real_time_recognition/All_face_sections/amplitude_heatmap/Each_pixel/Pixel_matching_methods/Template_Matching/Results/relax_tempmatch.mp4'
    output_path_data='/Users/henryschnieders/Desktop/Research/My_work/Real_time_recognition/All_face_sections/amplitude_heatmap/Each_pixel/Pixel_matching_methods/Template_Matching/Data/relax_tempmatch.npy'
    model_path='/Users/henryschnieders/Desktop/Research/My_work/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'

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
    


    roii = find_face(frames[0], detector)
    roi=frames[0][roii[1]:(roii[1]+roii[3]), roii[0]:(roii[0]+roii[2])]
    
    print(roi.shape) 
    h=roi.shape[0]
    w=roi.shape[1]

    data=main(frames)
    
    np.save(output_path_data, data)
    cv2.destroyAllWindows()
