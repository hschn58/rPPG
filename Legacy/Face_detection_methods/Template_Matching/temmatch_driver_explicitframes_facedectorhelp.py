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


def main(frames, roi, detector, out):
    ress = []
    for frame in frames[1:]:
        # Apply face detection to limit the template matching area
        face = find_face(frame, detector)

        try:
            if face is not None:
                # Define the region of interest for template matching based on detected face
                face_roi = frame[face[1]:(face[1] + face[3]), face[0]:(face[0] + face[2])]

                # Apply template matching within the detected face region
                res = cv2.matchTemplate(face_roi, roi, cv2.TM_CCOEFF_NORMED)
                ress.append(res)
                _, _, _, max_loc = cv2.minMaxLoc(res)

                # Map the max location back to the original frame
                top_left = (max_loc[0] + face[0], max_loc[1] + face[1])
                h, w = roi.shape[:2]  # Height and width of the ROI/template
                bottom_right = (top_left[0] + w, top_left[1] + h)

                # Draw a rectangle around the matched region
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
                cv2.imshow('Tracking', frame)

                out.write(frame)

            else:
                # Apply template matching to the whole frame if no face is detected
                res = cv2.matchTemplate(frame, roi, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)

                # Draw a rectangle around the matched region
                h, w = roi.shape[:2]
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
                cv2.imshow('Tracking', frame)

                out.write(frame)

        except Exception as e:
            print(e)

    return ress

        

if __name__=="__main__":

    fnum=1000
    video_data='/Users/henryschnieders/Desktop/Research/My_work/Data/video2.MOV'
    output_path='/Users/henryschnieders/Desktop/Research/My_work/Real_time_recognition/All_face_sections/amplitude_heatmap/Each_pixel/Pixel_matching_methods/Template_Matching/Results/inorganic_temmatch_3.mp4'
    model_path='/Users/henryschnieders/Desktop/Research/My_work/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'

    
    frames, fps, frame_width, frame_height = get_frames(video_data)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


    detector=cv2.FaceDetectorYN_create(model_path,
                        "", 
                        (frame_width, frame_height),
                        score_threshold=0.1) 
    h=50
    spoint=550
    lpoint=775

    roi = frames[0][lpoint:lpoint+h, spoint:spoint+h]  # Define the region of interest (ROI) to track

    faces=main(frames, roi, detector, out)
    print(faces)
    cv2.destroyAllWindows()
