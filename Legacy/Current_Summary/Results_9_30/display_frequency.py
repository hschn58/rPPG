
from denoise import denoise
import matplotlib.pyplot as plt
import numpy as np
import cv2


#recovery: 92
#relax: 75
#sporting: 123



vid='today'



output_description='padding_tempmatch_denoise'
filename='/Users/henryschnieders/Documents/Research/My_work/Data/'+vid+'_padding'+'.npy'


data=np.load(filename)

# print(np.array([frame[12,12] for frame in data]))
# exit()
# for frame in data:
#     print(frame.shape)
#     print(type(frame))
#     print('\n')

# # # for frame in data:
# # #     print(type(frame))
# exit()
# # # print(test)

#/Users/henryschnieders/Documents/Research/My_work/Face_detection_methods/corresponding_dataset_conversion/temmatch_driver_explicitframes_avi.py


# cv2.imshow('frame', data[0])
# cv2.waitKey(10000)
# cv2.destroyAllWindows()
# exit()
X_DIM=min([frame.shape[0] for frame in data])
Y_DIM=min([frame.shape[1] for frame in data])

bpm_array=np.zeros((X_DIM,Y_DIM), dtype=np.float32)



for i in range(X_DIM):
    for j in range(Y_DIM):
        
        # Extract amplitudes as an array of intensities over time for the (i, j) pixel
        #try:
        amplitudes = np.array([frame[i, j] for frame in data])

        bpm_array[i,j]=denoise(amplitudes)

        #except Exception as e:
        #    print(e)
        # # Check if amplitudes have the correct structure
        # if amplitudes.ndim == 1 and len(amplitudes) > 1:  # Ensure it's a 1D array with time-series data
        #     print(denoise(amplitudes))  # Pass as a list or array
        # else:
        #     print(f"Invalid data structure for pixel ({i}, {j}):", amplitudes)

    print(f'Finished column {i} of {X_DIM}')




#########################################################################################################
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
        return faces[0][:4].astype(int), frame_rgb
    return None


model_path = '/Users/henryschnieders/documents/Research/My_work/Other/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
detector = cv2.FaceDetectorYN_create(model_path, "", (data[-1].shape[1], data[-1].shape[0]), score_threshold=0.1)

roi, frame_rgb = find_face(data[-1], detector)

if roi is not None:
    x, y, w, h = roi
    
    #shrink roi a bit so no area is outside face
    shrink=0.2
    x=int(x+0.5*w*(1-(1-shrink)))
    y=int(y+0.5*h*(1-(1-shrink)))
    w=int(w*(1-shrink))
    h=int(h*(1-shrink))

    
    cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('frame', frame_rgb)
    cv2.imwrite('/Users/henryschnieders/Documents/Research/My_work/Face_detection_methods/data_visualization/'+vid+output_description+'facex'+'.png', frame_rgb)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


#########################################################################################################







from matplotlib.colors import TwoSlopeNorm



max_val=np.max(bpm_array)
min_val=np.min(bpm_array)
vmin = min_val  # Minimum value for the data range
vmax = max_val # Maximum value for the data range
vcenter= (max_val-min_val)/2

# Use TwoSlopeNorm to set the center
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

ave=np.average(bpm_array[x:x+w,y:y+h])

fig=plt.figure(figsize=(5,5))   

plt.title(f'{vid} -{ave:.2f} avgbpm')
plt.imshow(bpm_array, cmap='seismic', interpolation='nearest', norm=norm)
plt.colorbar()

fig.canvas.manager.set_window_title(f'Measured heart rate: {vcenter} bpm')

plt.savefig('/Users/henryschnieders/Documents/Research/My_work/Face_detection_methods/data_visualization/'+vid+output_description+'heatmap'+'.png')

bpm_flattened = bpm_array.flatten()

# Plot the histogram of BPM values
fig=plt.figure(figsize=(6, 4))
plt.hist(bpm_flattened, bins=50, color='blue', edgecolor='black', zorder=2)
plt.title('BPM Distribution- '+vid)    
plt.xlabel('Heart Rate (BPM)')
plt.ylabel('Relative Occurrences')
plt.grid(axis='y', linestyle='--', alpha=0.7)

fig.canvas.manager.set_window_title(f'Denoise BPM Distribution: {vid}')

plt.savefig('/Users/henryschnieders/Documents/Research/My_work/Face_detection_methods/data_visualization/'+vid+output_description+'histogram'+'.png')
plt.show()

