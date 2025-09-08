### Simple thresholding ###
# uses a single threshold value to classify pixel intensities.
# If a pixel's intensity is greater than the threshold, 
# it is set to 255 (white); otherwise, it is set to 0 (black).

import cv2
import numpy as np
from PIL import Image

img = cv2.imread('test_files/phantom_scan.png', cv2.IMREAD_GRAYSCALE)
threshold = 10

mask_array = np.where(img >= threshold, 255, 0).astype(np.uint8)

cv2.imwrite("simple_mask.png", mask_array)

### Adaptive thresholding ###
# calculates the threshold for small regions of the image

### MEAN
mask = cv2.adaptiveThreshold(
    src=img,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,  # Mean of the neighborhood
    thresholdType=cv2.THRESH_BINARY,
    blockSize=31,   # Size of neighborhood (must be odd and >1) 
                    # Width and height (in pixels) of the square neighborhood 
    C=-2            # Constant subtracted from mean
                    # Negative number good if ROI is brighter than background
)

cv2.imwrite("mean_adaptive_mask.png", mask)


### GAUSSIAN

mask = cv2.adaptiveThreshold(
    src=img,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Gaussian window
    thresholdType=cv2.THRESH_BINARY,
    blockSize=5001,   # Size of neighborhood (must be odd)
    C=-2             # Constant subtracted from weighted mean
)

# Save result
cv2.imwrite("gaussian_adaptive_mask.png", mask)


### NIBLACK'S and SAUVOLA'S

from skimage.util import view_as_windows

# Parameters
block_size = 15  # Odd number
k = 0.7         # Controls how much std dev influences threshold
R = 128          # dynamic range for std dev in 8-bit images. usually 128

# Pad the image to handle borders
pad = block_size // 2
padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)

# Create sliding windows
windows = view_as_windows(padded_img, (block_size, block_size))

# Compute local means and stds
means = np.mean(windows, axis=(2, 3))
stds = np.std(windows, axis=(2, 3))

# Niblack formula
niblack_thresholds = means + k * stds
niblack_mask = (img > niblack_thresholds).astype(np.uint8) * 255

# Sauvola threshold matrix
sauvola_thresholds = means * (1 + k * ((stds / R) - 1))

# Apply threshold
sauvola_mask = (img > sauvola_thresholds).astype(np.uint8) * 255

cv2.imwrite("niblack_mask.png", niblack_mask)
cv2.imwrite("sauvola_mask.png", sauvola_mask)


### OTSU'S
_, otsu_thresh = cv2.threshold(
    img,                # input image
    0,                  # initial threshold (ignored)
    255,                # max value for binary
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

cv2.imwrite("otsu_mask.png", otsu_thresh)


from cleaning import dots, blobs, blur_smooth, all_clean, clean

#dots('simple_mask.png', 11)
#blobs('dots_simple_mask.png', 100000)
#blur_smooth('blobs_dots_simple_mask.png', 91)

all_clean('simple_mask.png', 11, 100000, 91)
all_clean('gaussian_adaptive_mask.png', 11, 100000, 91 )
all_clean('otsu_mask.png', 11, 50000, 91)
