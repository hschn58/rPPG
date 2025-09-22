import cv2

def crop_and_resize_video(input_file, output_file, crop_area, new_size):

    cap = cv2.VideoCapture(input_file)

    fps = cap.get(cv2.CAP_PROP_FPS)

    x, y, w, h = crop_area

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    out = cv2.VideoWriter(output_file, fourcc, fps, new_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[y:y+h, x:x+w]
        resized_frame = cv2.resize(cropped_frame, new_size)
        out.write(resized_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 示例调用
input_video = r'C:\Users\zshao56\Desktop\50_light.mp4'  # 输入视频文件路径
output_video = r'C:\Users\zshao56\Desktop\50_light_resized.mp4'  # 输出视频文件路径
crop_area = (500, 600, 700, 700)  # 剪裁区域 (x, y, width, height)
new_size = (320, 240)  # 新的视频大小 (width, height)

crop_and_resize_video(input_video, output_video, crop_area, new_size)

