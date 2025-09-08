import cv2
import numpy as np


# Crop the image such that only the US sector is showing (not the metadata),
# estimate the size (height, width) of the kidney,
# choose two images and display the differences between them

def sector_crop(file_name):
    # Load image with unchanged flag to preserve any alpha channel
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)

    # Find contours to detect the sector
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create mask from the largest contour (sector)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)  # Draw filled contour

    isolated_sector = cv2.bitwise_and(gray, mask)

    # Save the result (single-channel grayscale)
    output_name = f'crop_{file_name}'
    cv2.imwrite(output_name, isolated_sector)
    print(f"Saved isolated grayscale sector image as: {output_name}")


# Example usage
#sector_crop('frame327.png')
#sector_crop('Sample_US_img.png')
