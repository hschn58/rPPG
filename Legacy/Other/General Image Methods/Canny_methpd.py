import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image (assuming the image is already in grayscale intensity map form)
# If your image is a NumPy array, skip the reading part
image = np.load('intensity_map.npy')  # Load your intensity map

# Apply Gaussian blur to smooth the image and reduce noise
#blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Use Canny edge detection
edges = cv2.Canny(image, threshold1=50, threshold2=150)

# Display the result
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.show()

# Optionally, save the edges as an image or process further
#cv2.imwrite('face_edges.png', edges)
