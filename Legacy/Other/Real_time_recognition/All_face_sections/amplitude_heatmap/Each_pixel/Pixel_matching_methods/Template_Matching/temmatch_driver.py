import cv2
import numpy as np

cap = cv2.VideoCapture('/Users/henryschnieders/Desktop/Research/My_work/Data/inorganic.MOV')

# Read the first frame
ret, frame = cap.read()
h=50
spoint=500
roi = frame[spoint:spoint+h, spoint:spoint+h]  # Define the region of interest (ROI) to track

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply template Matching
    res = cv2.matchTemplate(frame, roi, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    # Draw a rectangle around the matched region
    top_left = max_loc
    bottom_right = (top_left[0] + h, top_left[1] + h)
    cv2.rectangle(frame, top_left, bottom_right, (0,0,255), -1)
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
