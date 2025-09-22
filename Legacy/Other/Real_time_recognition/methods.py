import numpy as np

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

