import numpy as np
from PIL import Image, ImageOps

# Load the converted PNG image
image = Image.open('/Users/henryschnieders/Desktop/Research/My work/Data/facepic.png')


# Convert to grayscale (intensity map)
image_gray = image.convert('L')  # 'L' mode is for grayscale

# Resize to the desired shape (542, 512)
image_resized = image_gray.resize((512, 542))

# Convert to a NumPy array for further processing
intensity_map = np.array(image_resized)

# Save the intensity map as a .npy file
np.save('/Users/henryschnieders/Desktop/Research/My work/Data/intensity_map.npy', intensity_map)

# Optionally, you can display the image or perform further processing
image_resized.save('/Users/henryschnieders/Desktop/Research/My work/Data/facepic_resized.png')
