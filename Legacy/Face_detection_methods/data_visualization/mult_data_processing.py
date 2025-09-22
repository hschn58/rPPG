



from sp_datadisplay import signal_process_alter
import matplotlib.pyplot as plt
import numpy as np
import cv2



#make this a function...
    vid_name='relax'
    filename='/Users/henryschnieders/Documents/Research/My_work/Real_time_recognition/All_face_sections/amplitude_heatmap/Each_pixel/Pixel_matching_methods/Template_Matching/Data/relax_tempmatch.npy'


    data=np.load(filename, allow_pickle=True)
    data=data[:-5]

    test=[frame[0,0] for frame in data]



    # cv2.imshow('frame', data[0])
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()
    # exit()
    X_DIM=min([frame.shape[0] for frame in data])
    Y_DIM=min([frame.shape[1] for frame in data])

    bpm_array=np.zeros((X_DIM,Y_DIM), dtype=np.float32)


    # plt.figure(figsize=(5,5))
    for i in range(X_DIM):
        for j in range(Y_DIM):
            try:
                amplitudes=np.array([frame[i,j] for frame in data],dtype=np.float32)

                #roi_mean=compute_roi_mean(amplitudes)   
                # print(type(roi_mean))
                bpm_array[i,j]=signal_process_alter(amplitudes, fps=59.94005994005994)

                # plt.plot(np.arange(len(roi_mean)), roi_mean)
                # plt.pause(0.01)
            except Exception as e:
                print(e, f'[{i,j}]')
                continue
        
        print(f'finished column {i} of {X_DIM}')
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
    from matplotlib.colors import TwoSlopeNorm



    max_val=np.max(bpm_array)
    min_val=np.min(bpm_array)
    vmin = min_val  # Minimum value for the data range
    vmax = max_val # Maximum value for the data range
    vcenter = 75 # Set this to the desired midpoint (zero)

    # Use TwoSlopeNorm to set the center
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    ave=np.average(bpm_array)

    plt.figure(figsize=(5,5))   

    plt.title(f'{vid_name} bpm heatmap-{ave:.2f} avgbpm')
    plt.imshow(bpm_array, cmap='seismic', interpolation='nearest', norm=norm)
    plt.colorbar()
    plt.savefig('/Users/henryschnieders/Documents/Research/My_work/Real_time_recognition/All_face_sections/frequency_heatmap/Approaches/Each_pixel/Template_matching/freq_map_tempmatch/Results/'+f'{vid_name}'+'bpm_heatmap_tempmatch_diffsignalprocess.png')
    plt.show()