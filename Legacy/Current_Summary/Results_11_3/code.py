
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt


filename='/Users/henryschnieders/Documents/Research/My_work/Data/100_light_94_94_padding_color_mp4.npy'
vid_path='/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Results_10_21/Videos/mp4/100_light_94_94.mp4'



data=np.load(filename, allow_pickle=True)

height, width, _ = data[-1].shape


def get_frames(video_path, desired_frames=2000):
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

    return fps

fps=get_frames(vid_path)


import matplotlib.pyplot as plt
import cv2

xcoord=width//3
ycoord=height//2

wide = 20

point0 = [frame[(ycoord - wide // 2):(ycoord + wide // 2), (xcoord - wide // 2):(xcoord + wide // 2)][0] for frame in data]

frame = data[-1]
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.imshow(frame)

# Calculate rectangle coordinates
rect_x = xcoord - wide // 2
rect_y = ycoord - wide // 2

# Add rectangle
plt.gca().add_patch(plt.Rectangle((rect_x, rect_y), wide, wide, linewidth=0.5, edgecolor='red', facecolor='none'))
plt.show()

point0 = [np.mean(frame[(ycoord - wide // 2):(ycoord + wide // 2), (xcoord - wide // 2):(xcoord + wide // 2),0]) for frame in data]
point1 = [np.mean(frame[(ycoord - wide // 2):(ycoord + wide // 2), (xcoord - wide // 2):(xcoord + wide // 2),1]) for frame in data]
point2 = [np.mean(frame[(ycoord - wide // 2):(ycoord + wide // 2), (xcoord - wide // 2):(xcoord + wide // 2),2]) for frame in data]

print(len(point0), len(point1), len(point2))
pix_intensity1=point0
pix_intensity2=point1
pix_intensity3=point2


def signal_process_FFT(pix_intensity, fps):
    sampling_rate = fps

    lowcut_heart = 0.5
    highcut_heart = 3.0

    N = len(pix_intensity)  # 信号长度
    T = 1.0 / sampling_rate  # 采样间隔
    coefs = np.fft.fft(pix_intensity)  # 傅里叶变换
    freqs = np.fft.fftfreq(N, T) # 获取频率

    indices = np.where((np.abs(freqs) >= lowcut_heart) & (np.abs(freqs) <= highcut_heart))

    coefsinrange=coefs[indices]
    freqsinrange=freqs[indices]
    max_indicies=np.argsort(complex_mag(coefsinrange))[-5:]
    coefsinrange=coefsinrange[max_indicies]
    freqsinrange=freqsinrange[max_indicies]

    FFT_heart_rate=np.average(np.abs(freqsinrange), weights=complex_mag(coefsinrange))

    return FFT_heart_rate*60


import numpy as np
import matplotlib.pyplot as plt

def complex_mag(x):
    return np.absolute(x)

def analyze_pix_intensity(pix_intensity, sampling_rate, lowcut_heart, highcut_heart, color, label='', start=50):
    N_total = len(pix_intensity)
    T = 1.0 / sampling_rate


    # 1. Compute FFT over the entire data to find the prominent frequencies
    coefs_total = np.fft.fft(pix_intensity)
    freqs_total = np.fft.fftfreq(N_total, T)

    # Select frequency range
    indices = np.where((np.abs(freqs_total) >= lowcut_heart) & (np.abs(freqs_total) <= highcut_heart))
    coefsinrange_total = coefs_total[indices]
    freqsinrange_total = freqs_total[indices]

    # Find top 5 frequencies
    max_indices = np.argsort(complex_mag(coefsinrange_total))[-5:]
    top_coefs = coefsinrange_total[max_indices]
    top_freqs = freqsinrange_total[max_indices]

    # Initialize an array to store coefficients for each n
    coefs_vs_n = np.zeros((len(top_freqs), N_total-start), dtype=complex)

    for n in range(1, N_total-start+1):
        # Get data up to the nth array
        pix_intensity_n = pix_intensity[:(n+start-1)]
        N_n = len(pix_intensity_n)
        T_n = T  # Sampling interval remains the same

        # Compute FFT over data up to the nth array
        coefs_n = np.fft.fft(pix_intensity_n)
        freqs_n = np.fft.fftfreq(N_n, T_n)

        # For each of the top frequencies, get the Fourier coefficients
        for i, f in enumerate(top_freqs):
            # Find the index in freqs_n closest to f
            idx = np.argmin(np.abs(freqs_n - f))
            coefs_vs_n[i, n - 1] = coefs_n[idx]

    # 2. Plot the coefficient magnitudes with respect to n


    
    ydata=[np.mean(complex_mag(freqs)) for freqs in coefs_vs_n.reshape((N_total-start, len(top_freqs)))]
    plt.plot(range(start, N_total), ydata, label=f'Averaged top 5 frequencies', color=color)
    
    print(f'{label} freq is {signal_process_FFT(ydata, fps)}')
    
plt.ion()



# Parameters (adjust as needed)
sampling_rate = fps  # Frame rate of the video
lowcut_heart = 0.5   # Lower bound of heart rate frequency in Hz
highcut_heart = 3.0  # Upper bound of heart rate frequency in Hz

# Assuming 
# , pix_intensity2, pix_intensity3 are 1D numpy arrays
# Replace these with your actual data
# pix_intensity1 = np.array([...])
# pix_intensity2 = np.array([...])
# pix_intensity3 = np.array([...])

colors = ['red', 'green', 'blue']
analyze_pix_intensity(pix_intensity1, sampling_rate, lowcut_heart, highcut_heart, label='Blue Channel', color=colors[2])
analyze_pix_intensity(pix_intensity2, sampling_rate, lowcut_heart, highcut_heart, label='Green Channel', color=colors[1])
analyze_pix_intensity(pix_intensity3, sampling_rate, lowcut_heart, highcut_heart, label='Red Channel', color=colors[0])

plt.xlabel('Number of Frames')
plt.ylabel('Coefficient Magnitude')
plt.title(f'Coefficient Magnitude for all channels')

plt.grid('on')
plt.savefig('/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Results_10_28/results/'+'all_'+'orig_' 'averaged.png', dpi=600)
plt.show()
