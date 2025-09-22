import params
from sp_datadisplay import compute_roi_mean, signal_process
import matplotlib.pyplot as plt
import numpy as np
import cv2


X_SIZE, Y_SIZE = params.X_SIZE, params.Y_SIZE

vid_name='relax'
filename='/Users/henryschnieders/Desktop/Research/My_work/Real_time_recognition/All_face_sections/Data/'+f'{vid_name}'+'_frames.npy'

data=np.load(filename, allow_pickle=True)

#last 3 frames are type int for some reason
data=data[:-3]
bpm_array=np.zeros((X_SIZE,Y_SIZE), dtype=np.float32)



# plt.figure(figsize=(5,5))
for i in range(X_SIZE):
    for j in range(Y_SIZE):
        try:
            amplitudes=[frame[i,j] for frame in data]

            roi_mean=compute_roi_mean(amplitudes)   
            # print(type(roi_mean))
            bpm_array[i,j]=signal_process(roi_mean=roi_mean)

            # plt.plot(np.arange(len(roi_mean)), roi_mean)
            # plt.pause(0.01)
        except Exception as e:
            print(e, f'[{i,j}]')
            continue
        
        # amplitudes=[frame[i,j] for frame in data]

        # roi_mean=compute_roi_mean(amplitudes)   
        # # print(type(roi_mean))
        # bpm_array[i,j]=signal_process(roi_mean=roi_mean)


# plt.savefig('/Users/henryschnieders/Desktop/Research/My_work/Real_time_recognition/All_face_sections/Data/'+f'{vid_name}'+'_bpm.png')

plt.figure(figsize=(5,5))   
plt.imshow(bpm_array, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.savefig('/Users/henryschnieders/Desktop/Research/My_work/Real_time_recognition/All_face_sections/Data/'+f'{vid_name}'+'_bpm_array.png')
plt.show()