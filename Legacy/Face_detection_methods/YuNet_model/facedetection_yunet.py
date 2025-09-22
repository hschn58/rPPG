import cv2
import numpy as np

model_path='/Users/henryschnieders/Desktop/Research/My_work/YuNet-Package/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
data="/Users/henryschnieders/Desktop/Research/My_work/Data/intensity_map.npy"
#img_out_path=".../png_detection_nparray.png"

detector=cv2.FaceDetectorYN_create(model_path,
                          "", 
                          (300, 300),
                          score_threshold=0.5) 

image=np.load(data)

# Resize the image to match the input size of the model
image_resized = cv2.resize(image, (300, 300))
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR) #the model needs a 3-channel image, this copies the grayscale image to all 3 channels

_, faces = detector.detect(image_rgb)

rad=3
color=(0,0,255)
thickness=-1   #fill the circle

print(faces[0][:4])
if faces is not None:
    for face in faces:
        bbox = face[:4].astype(int) 

        #bounding box
        cv2.rectangle(image_rgb, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        
        #cheeks and chin
        cv2.circle(image_rgb, center=(bbox[0]+int(bbox[2]*(1/6)), bbox[1]+int(bbox[3]/2)), radius=rad, color=color, thickness=thickness)
        cv2.circle(image_rgb, center=(bbox[0]+int(bbox[2]*(5/6)), bbox[1]+int(bbox[3]/2)), radius=rad, color=color, thickness=thickness)

        cv2.circle(image_rgb, center=(bbox[0]+int(bbox[2]/2), bbox[1]+int(bbox[3]*(7/8))), radius=rad, color=color, thickness=thickness)
        
cv2.imshow("image",image_rgb)
cv2.waitKey(5000)
cv2.destroyAllWindows()