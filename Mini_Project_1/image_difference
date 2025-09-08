import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pydicom import dcmread


def difference(filename1, filename2, threshold):
    # Load grayscale frames (as 2D arrays)
    frame1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)

    # Compute absolute pixel differences
    diff = np.abs(frame1.astype(np.int16) - frame2.astype(np.int16))

    # Set low differences below threshold to 0 to ignore noise
    diff_filtered = np.where(diff >= threshold, diff, 0)

    # Display heatmap
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title(filename1)
    plt.imshow(frame1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(filename2)
    plt.imshow(frame2, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Pixel Difference (≥{threshold})")
    plt.imshow(diff_filtered, cmap='hot')
    plt.colorbar(label='Intensity Difference')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

#difference("frame325.png", "frame326.png", 10)

def frame_difference(DICOM_filename, frame_num1, frame_num2, threshold):
    DICOM = dcmread(DICOM_filename)
    pixel_array = DICOM.pixel_array
    grayscale_array = np.mean(pixel_array, axis=-1).astype(np.uint8)  # shape: (577, 649, 850)
    
    frame1 = grayscale_array[frame_num1]
    frame2 = grayscale_array[frame_num2]

     # Compute absolute pixel differences
    diff = np.abs(frame1.astype(np.int16) - frame2.astype(np.int16))

    # Set low differences below threshold to 0 to ignore noise
    diff_filtered = np.where(diff >= threshold, diff, 0)

    # Display heatmap
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title(f'Frame {frame_num1}')
    plt.imshow(frame1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f'Frame {frame_num2}')
    plt.imshow(frame2, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Pixel Difference (≥{threshold})")
    plt.imshow(diff_filtered, cmap='hot')
    plt.colorbar(label='Intensity Difference')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

#frame_difference('starting_US_test_c', 325, 326, 5)
