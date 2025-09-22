import cv2
import numpy as np

def get_frames_from_npy(npy_file_path):
    """
    Load frames from an .npy file.

    :param npy_file_path: Path to the .npy file containing intensity frames
    :return: Loaded frames as a list, number of frames, frame width, and frame height
    """
    # Load the data from the .npy file
    data = np.load(npy_file_path)
    frame_count = data.shape[0]
    frame_height = data.shape[1]
    frame_width = data.shape[2]

    # Convert intensity frames to a list
    frames = [data[i].astype(np.float32) for i in range(frame_count)]  # Convert to float32 for OpenCV compatibility
    
    return frames, frame_count, frame_width, frame_height

def find_face(frame, detector):
    """
    Detect the face in the given frame using the provided face detector.

    :param frame: The frame to detect the face in
    :param detector: The face detector object
    :return: Bounding box of the detected face, or None if no face is detected
    """
    # Normalize and convert frame to a 3-channel image if needed
    frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)  # Normalize to 0-255 range
    frame_uint8 = frame_normalized.astype(np.uint8)  # Convert to uint8
    frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)

    _, faces = detector.detect(frame_rgb)
    
    if faces is not None:
        return faces[0][:4].astype(int)
    return None

def main(frames, roi, h, w, output_video_path, fps):
    """
    Perform template matching to track the face in subsequent frames.

    :param frames: List of frames
    :param roi: Region of interest (face template)
    :param h: Height of the ROI
    :param w: Width of the ROI
    :param output_video_path: Path to save the output video
    :param fps: Frames per second of the output video
    :return: List of areas containing detected faces
    """
    padding = 0.1
    area = []
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames[1:]:
        # Apply template matching
        res = cv2.matchTemplate(frame, roi, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)

        # Draw a rectangle around the matched region
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        matched_area = frame[max(0, top_left[1] - int(h * padding)):min(frame.shape[0], bottom_right[1] + int(h * padding)),
                             max(0, top_left[0] - int(w * padding)):min(frame.shape[1], bottom_right[0] + int(w * padding))]
        
        area.append(matched_area)

        # Draw rectangle on the frame
        cv2.rectangle(frame, (max(0, top_left[0] - int(w * padding)), max(0, top_left[1] - int(h * padding))),
                      (min(frame.shape[1], bottom_right[0] + int(w * padding)), min(frame.shape[0], bottom_right[1] + int(h * padding))),
                      (0, 0, 255), 1)
        cv2.imshow('Tracking', frame)
        
        # Update ROI for the next frame
        roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Write the frame to the output video
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)
        cv2.waitKey(1)

    # Release the video writer
    out.release()
    return area

if __name__ == "__main__":
    npy_file_path = '/Users/henryschnieders/Documents/Research/My_work/Data/New_sporting/1/output_intensity_frames.npy'
    output_path_vid = '/Users/henryschnieders/Documents/Research/My_work/Face_detection_methods/data_visualization/sporting_tempmatch.mp4'
    output_path_data = '/Users/henryschnieders/Documents/Research/My_work/Data/sporting_padding.npy'

    model_path = '/Users/henryschnieders/documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'

    # Load frames from the .npy file
    frames, frame_count, frame_width, frame_height = get_frames_from_npy(npy_file_path)
    
    # Initialize face detector
    detector = cv2.FaceDetectorYN_create(model_path, "", (frame_width, frame_height), score_threshold=0.1)

    # Detect face in the first frame
    roii = find_face(frames[0], detector)
    if roii is not None:
        roi = frames[0][roii[1]:(roii[1] + roii[3]), roii[0]:(roii[0] + roii[2])]
        h, w = roi.shape
        data = main(frames, roi, h, w, output_path_vid, fps=30)  # Assuming fps=30; adjust if necessary
        np.save(output_path_data, data)
        print("Processing complete. Output saved.")

    else:
        print("No face detected in the first frame.")

    cv2.destroyAllWindows()
