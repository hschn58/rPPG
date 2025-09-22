
import pywt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import butter, filtfilt, find_peaks
import cv2
import time
import os



def load_noise():
    # check if the noise is got previously.
    filename_I = '/Users/henryschnieders/Desktop/Research/My work/Data/noise_IB.npy'
    filename_Q = '/Users/henryschnieders/Desktop/Research/My work/Data/noise_QB.npy'
    if os.path.isfile(filename_I) and os.path.isfile(filename_Q):
        noise_I = np.load(filename_I)
        noise_Q = np.load(filename_Q)
        # print("noise_I and noise_Q are loaded. \n")
    else:
        print("Please load the noise at first...\n")

    # Calculate mean along axis 0
    bg_mean_Q = np.mean(noise_Q, axis=0)
    bg_mean_I = np.mean(noise_I, axis=0)

    return bg_mean_Q, bg_mean_I


def get_amplitude(bg_mean_Q, bg_mean_I, I_data, Q_data, n_frames):
    # 创建空列表用于存储每一帧的强度信号
    frames_list = []

    for i in range(n_frames):
        frame_Q = Q_data[i]
        frame_I = I_data[i]

        frame_Q = frame_Q.astype(np.float32)
        frame_I = frame_I.astype(np.float32)

        # 计算强度信号
        frame = np.sqrt((frame_Q - bg_mean_Q) ** 2 + (frame_I - bg_mean_I) ** 2)

        # 可选：对信号进行进一步处理，如归一化、滤波等
        # 左旋90°
        frame = np.rot90(frame)
        # 将每一帧的强度信号存储到列表中
        frames_list.append(frame)

    # 将列表转换为 numpy 数组
    Amplitude = np.array(frames_list, dtype=np.float32)
    # 保存为 .npy 文件
    # np.save('Amplitude.npy', Amplitude)
    #
    # print("强度信号保存成功！")

    return Amplitude[1:]


def compute_roi_mean(video_data):
    # 检查视频数据的形状
    num_frames=len(video_data)
    roi_mean = np.zeros(num_frames)  # 存储每一帧ROI均值的数组


    def normalize_to_range(frame, min_val=0, max_val=255):
        """将图像数据缩放到指定的范围 [min_val, max_val]"""
        min_frame = np.min(frame)
        max_frame = np.max(frame)

        # 避免除以零的情况
        if max_frame == min_frame:
            return np.full_like(frame, min_val, dtype=np.uint8)

        normalized_frame = (frame - min_frame) / (max_frame - min_frame) * (max_val - min_val) + min_val
        return np.clip(normalized_frame, min_val, max_val).astype(np.uint8)
    count = 0
    

    for i in range(num_frames):
        frame = video_data[i]
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = normalize_to_range(frame, min_val=0, max_val=255)


        # 人脸检测

         # 选择第一个检测到的人脸区域

        # 更新上一帧的ROI
    
        # 计算ROI区域的均值
       
        roi = frame

        if roi.size > 0:
            roi_mean[i] = np.mean(roi)
        else:
            roi_mean[i] = np.mean(frame)  # 如果ROI区域为空，使用整帧均值

    # print("----detect_rate:", detect_rate)
    return roi_mean

# 定义滤波器
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, padlen=14):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, padlen=padlen)
    return y



import numpy as np
import cv2

def resize_video_frames(video_data):
    """
    Resize each frame of the video to half its original size and normalize the pixel values.

    Parameters:
    video_data (numpy.ndarray): 3D array with shape (frames, height, width) where frames is the number of frames.

    Returns:
    numpy.ndarray: Resized and normalized video with shape (frames, new_height, new_width).
    """
    # Ensure the video data is in the expected format
    if len(video_data.shape) != 3:
        raise ValueError("Input video_data must be a 3D array (frames, height, width).")

    # Get video dimensions
    frames, height, width = video_data.shape

    # Prepare to store resized frames
    new_height = height // 2
    new_width = width // 2
    resized_video = np.zeros((frames, new_height, new_width), dtype=np.float32)

    # Iterate over each frame
    for i in range(frames):
        # Normalize frame to [0, 255]
        frame = video_data[i]
        frame_normalized = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        frame_normalized = np.uint8(frame_normalized)

        # Resize frame to half its size
        frame_resized = cv2.resize(frame_normalized, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Store the resized frame
        resized_video[i] = frame_resized

    # Normalize resized frames back to [0, 1]
    resized_video = resized_video / 255.0

    return resized_video

def signal_process_bpm(roi_mean):
    # # 1. 读取视频数据

    #video_data=Amplitude
    # 2. 通过人脸识别确定ROI

    #w,h=video_data.shape[1],video_data.shape[2]
    #x0,y0,x1,y1=0,0,w,h
    
    # 定义滤波器参数
    sampling_rate = 19.802 # 95.2381  # 假设视频帧率为30帧每秒
    # 提取心率和呼吸信号的频率范围
    # 心率通常在 0.5 - 4 Hz
    # 呼吸信号通常在 0.1 - 0.5 Hz
    lowcut_heart = 1.0
    highcut_heart = 3.0
    #lowcut_breath = 0.3
    #highcut_breath = 1.0

    # 傅里叶变换
    N = len(roi_mean)  # 信号长度
    T = 1.0 / sampling_rate  # 采样间隔
    yf = np.fft.fft(roi_mean)
    xf = np.fft.fftfreq(N, T)[:N//2] # 获取频率

    # 先进行一次带通滤波
    roi_mean_ = butter_bandpass_filter(roi_mean, 1, 6, sampling_rate, order=5)
    #roi_mean_wavelet = butter_bandpass_filter(roi_mean, 0.1, 9, sampling_rate, order=5)
    # print(N, T, len(xf), xf[1:10])


    # # 计算频谱（取模，且除以信号长度来归一化）
    frequency_spectrum = 2.0/N * np.abs(yf[:N//2])

    indices = np.where((xf >= lowcut_heart) & (xf <= highcut_heart))
    # 获取该频段内的频率和幅值
    frequencies_in_range = xf[indices]
    spectrum_in_range = frequency_spectrum[indices]
    # print("frequencies_in_range:", frequencies_in_range)
    # print("spectrum_in_range:", spectrum_in_range)
    # 获取前5个最大幅度的索引
    top5_indices = np.argsort(spectrum_in_range)[-3:]

    top5_frequencies = frequencies_in_range[top5_indices]
    top5_spectrum = spectrum_in_range[top5_indices]
    # 计算加权平均
    #np.save(f'/Users/henryschnieders/Desktop/Research/My_work/Real_time_recognition/All_face_sections/Approaches/Each_pixel/Data/top5_spectrum_{i}_{j}.npy',top5_spectrum)
    FFT_heart_rate = np.average(top5_frequencies, weights=top5_spectrum)

    wavelet = 'morl' # 'morl'
    scales = np.arange(1, 128)
    #coef_, frequencies_ = pywt.cwt(roi_mean_wavelet, scales, wavelet, T)
    coef, frequencies = pywt.cwt(roi_mean_, scales, wavelet, T)

    # 计算每个尺度的功率谱
    average_spectrum = np.mean(np.abs(coef) ** 2, axis=1)
    wavelet_peaks, _ = find_peaks(average_spectrum, height=0)
    # 输出指定索引对应的频率
    # print(wavelet_peaks)
    WaveLet_heart_rate = 1.3
    if wavelet_peaks is not None:
        index_u = []
        flag_heart = True
        # flag_breath = True
        for index in reversed(wavelet_peaks):
            # print(f"索引 {index}: 频率 {frequencies[index]:.2f} Hz")
            if frequencies[index] > 1.0 and flag_heart:
                flag_heart = False
                WaveLet_heart_rate = frequencies[index]
                index_u.append(index)
                # print(f'WaveLet Heart: {WaveLet_heart_rate * 60:.2f} bpm, {WaveLet_heart_rate:.2f} Hz')
   
    heart_rate_signal = butter_bandpass_filter(roi_mean_, lowcut_heart, highcut_heart, sampling_rate)
    peaks_heart, _ = find_peaks(heart_rate_signal, height=0)
    Butter_heart_rate = len(peaks_heart) / (len(roi_mean_) / sampling_rate)  # 频率 (bpm)
   
    heart_rate = (FFT_heart_rate+WaveLet_heart_rate+Butter_heart_rate)/3*60
   
    
    return heart_rate



def signal_process_amplitude(roi_mean):
    # # 1. 读取视频数据

    #video_data=Amplitude
    # 2. 通过人脸识别确定ROI

    #w,h=video_data.shape[1],video_data.shape[2]
    #x0,y0,x1,y1=0,0,w,h
    
    # 定义滤波器参数
    sampling_rate = 19.802 # 95.2381  # 假设视频帧率为30帧每秒
    # 提取心率和呼吸信号的频率范围
    # 心率通常在 0.5 - 4 Hz
    # 呼吸信号通常在 0.1 - 0.5 Hz
    lowcut_heart = 1.0
    highcut_heart = 3.0
    #lowcut_breath = 0.3
    #highcut_breath = 1.0

    # 傅里叶变换
    N = len(roi_mean)  # 信号长度
    T = 1.0 / sampling_rate  # 采样间隔
    yf = np.fft.fft(roi_mean)
    xf = np.fft.fftfreq(N, T)[:N//2] # 获取频率

    # 先进行一次带通滤波
    roi_mean_ = butter_bandpass_filter(roi_mean, 1, 6, sampling_rate, order=5)
    #roi_mean_wavelet = butter_bandpass_filter(roi_mean, 0.1, 9, sampling_rate, order=5)
    # print(N, T, len(xf), xf[1:10])


    # # 计算频谱（取模，且除以信号长度来归一化）
    frequency_spectrum = 2.0/N * np.abs(yf[:N//2])

    indices = np.where((xf >= lowcut_heart) & (xf <= highcut_heart))
    # 获取该频段内的频率和幅值
    frequencies_in_range = xf[indices]
    
    target=np.where(xf[indices]==max(frequencies_in_range))

    return np.abs(yf[target])



if __name__ == '__main__':

    import wholeframe_todata
    x_size, y_size = wholeframe_todata.bulk.x_size, wholeframe_todata.bulk.y_size

    vid_name='sporting_1'
    filename='/Users/henryschnieders/Desktop/Research/My_work/Real_time_recognition/All_face_sections/Data/'+f'{vid_name}'+'_frames.npy'
    Amplitude_ = np.load(filename)
    print(Amplitude_.shape)
    
    # for section in range(x_size*y_size):
    #     for frame in range(Amplitude_.shape[0]):

    #         roi_mean = compute_roi_mean(Amplitude_[:,:,:,face_section])
    #         heart_rate = signal_process(roi_mean, Amplitude_[:,:,:,face_section], ref_frame, plot_loc[face_section], vid_name, sect_names[face_section])
    #         print(f"Final heart_rate: {heart_rate:.2f} bpm in region {sect_names[face_section]}\n")


    #Amplitude_ = resize_video_frames(Amplitude_)

    


 