
from denoise import denoise_and_return_intensity
import matplotlib.pyplot as plt
import numpy as np


#let the face be found by the detector, and then set the face to the template matching,
#compare to face detection results




#filename='/Users/henryschnieders/Desktop/Research/My_work/Real_time_recognition/All_face_sections/amplitude_heatmap/Data/resized_inorganic.npy'

vid='recovery'


output_description='padding_tempmatch_denoise'
filename='/Users/henryschnieders/Documents/Research/My_work/Data/'+vid+'_padding.npy'



data=np.load(filename, allow_pickle=True)
data=data[:-5]

X_DIM=min([frame.shape[0] for frame in data])
Y_DIM=min([frame.shape[1] for frame in data])

bpm_array=np.zeros((X_DIM,Y_DIM), dtype=np.float32)


# plt.figure(figsize=(5,5))
for i in range(X_DIM):
    for j in range(Y_DIM):
        try:
            amplitudes=np.array([frame[i,j] for frame in data], dtype=np.float32)

            #roi_mean=compute_roi_mean(amplitudes)   
            # print(type(roi_mean))
            bpm_array[i,j]=denoise_and_return_intensity(amplitudes)

            print('done with ', X_DIM*i+j,'of', X_DIM*Y_DIM)

            # plt.plot(np.arange(len(roi_mean)), roi_mean)
            # plt.pause(0.01)
        except Exception as e:
            print(e, f'[{i,j}]')
            continue
        
    print(f'done with {i} of {X_DIM} ')
        # amplitudes=[frame[i,j] for frame in data]

        # roi_mean=compute_roi_mean(amplitudes)   
        # # print(type(roi_mean))
        # bpm_array[i,j]=signal_process(roi_mean=roi_mean)

# # plt.figure(figsize=(5,5))
# for i in range(X_DIM):
#     for j in range(Y_DIM):
        
#         amplitudes=[frame[i,j] for frame in data]

#         roi_mean=compute_roi_mean(amplitudes)   
#         # print(type(roi_mean))
#         bpm_array[i,j]=signal_process(roi_mean=roi_mean)

#             # plt.plot(np.arange(len(roi_mean)), roi_mean)
#             # plt.pause(0.01)
        

# plt.savefig('/Users/henryschnieders/Desktop/Research/My_work/Real_time_recognition/All_face_sections/Data/'+f'{vid_name}'+'_bpm.png')
#get rid of bad data
# from matplotlib.colors import TwoSlopeNorm


# max_val=np.max(bpm_array)
# min_val=np.min(bpm_array)
# vmin = min_val  # Minimum value for the data range
# vmax = max_val # Maximum value for the data range
# vcenter = (max_val+min_val)/2 #  Set this to the desired midpoint (zero)

# # Use TwoSlopeNorm to set the center
# norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

ave=np.average(bpm_array)

plt.figure(figsize=(5,5))   

plt.title(f' {ave:.2f} avg amplitude')
plt.xlabel('pixels')
plt.ylabel('pixels')
plt.imshow(bpm_array, cmap='autumn_r', interpolation='nearest')

clb=plt.colorbar()
clb.set_label('relative coefficient amplitude', rotation=270, labelpad=20)
plt.savefig('/Users/henryschnieders/Documents/Research/My_work/Face_detection_methods/data_visualization/'+vid+'_amplitude_tempmatch.png')
plt.show()