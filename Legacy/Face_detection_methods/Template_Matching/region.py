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
        

import numpy as np

def generate_pixel_locations(start_point, end_point):
    """
    Generate an array of shape (-1, 1, 2*n) containing the (x, y) locations of all pixels
    between an arbitrary start and end point within an image.

    Parameters:
    - start_point: A tuple (x_start, y_start) representing the starting coordinates.
    - end_point: A tuple (x_end, y_end) representing the ending coordinates.

    Returns:
    - A numpy array of shape (-1, 1, 2*n) with pixel (x, y) coordinates between the start and end points.
    """
    # Extract start and end coordinates
    x_start, y_start = start_point
    x_end, y_end = end_point
    
    # Generate the x and y coordinates for the specified range
    y, x = np.indices((y_end - y_start + 1, x_end - x_start + 1))
    
    # Offset by the starting point
    x += x_start
    y += y_start
    
    # Flatten the x and y coordinates and stack them along the last axis
    pixel_locations = np.stack((x.flatten(), y.flatten()), axis=-1)
    
    # Reshape the array to the desired shape (-1, 1, 2*n)
    output_array = pixel_locations.reshape(-1, 1, 2).astype(np.float32)
    
    return output_array


# def get_IB_QB(frames,target):

#     #get IB, QB data from the area in each frame that is not in target (face) region

#     padding=10



#     n_frames = len(frames) // 2
#     return I_1d, Q_1d

def main():

    tot_frames=900
    frame_width=500
    frame_height=frame_width
    size=frame_width


    output_path='/Users/henryschnieders/Desktop/Research/My_work/Real_time_recognition/All_face_sections/amplitude_heatmap/Each_pixel/Pixel_matching_methods/Optical_Flow/Results/optical_flow.mp4'
    video_data='/Users/henryschnieders/Desktop/Research/My_work/Data/inorganic.MOV'

    frames, fps = get_frames(video_data, tot_frames, frame_width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    p0=generate_pixel_locations((350,300), (370,320))
    

    old_frame=frames[0]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray = cv2.resize(old_gray, (size, size))

    for frame_idx in range(0, len(frames)):
        frame = frames[frame_idx]

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.resize(frame_gray, (size, size))

        # Calculate optical flow
        p1, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)

        print('iteration complete')
        p2=p1[0][0]
        
        cv2.circle(frame, center=(int(p2[0]),int(p2[1])), radius=15, color=(0,0,255), thickness=-1)
        out.write(frame)

        p0 = p1.reshape(-1, 1, 2)
    
if __name__=="__main__":

    start=time.time()
    main()
    end=time.time()
    print("Time taken:", end-start)
    

        






# '/Users/henryschnieders/Desktop/Research/My work/Data/video.MOV'







# print("Shape of the saved array:", frames_array.shape)
