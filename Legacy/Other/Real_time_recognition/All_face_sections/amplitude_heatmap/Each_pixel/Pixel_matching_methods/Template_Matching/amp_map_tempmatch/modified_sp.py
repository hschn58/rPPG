
import pywt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import butter, filtfilt, find_peaks
import cv2
import time
import os



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

############################################################################################################

def signal_process_bpm(roi_mean, fps):
    # # 1. 读取视频数据

    #video_data=Amplitude
    # 2. 通过人脸识别确定ROI

    #w,h=video_data.shape[1],video_data.shape[2]
    #x0,y0,x1,y1=0,0,w,h
    
    # 定义滤波器参数
    sampling_rate = fps # 95.2381  # 假设视频帧率为30帧每秒
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

