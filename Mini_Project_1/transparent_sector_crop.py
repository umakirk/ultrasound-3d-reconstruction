import cv2
import numpy as np


# Cropping the image such that only the US sector is showing (not the metadata),
# now making the background (everything other than sector) transparent 

def transparent_sector_crop(file_name):
    # Load image with unchanged flag to preserve any alpha channel
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    
    # grayscale image
    if img.ndim == 2:
        gray = img
    
    #RGB image, needs to be converted to grayscale
    elif img.ndim == 3 and img.shape[2] == 3: # shape: (H, W, 3), 3 channels RGB
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #RBGA image, strip alpha and convert to grayscale
    elif img.ndim == 3 and img.shape[2] == 4: # shape: (H, W, 4), 4 channels RBGA
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)

    # Find contours to detect the sector
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create mask from the largest contour (sector)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)  # Draw filled contour

    # Stack grayscale image into 3 channels
    rgb_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Create alpha channel: 255 inside the sector, 0 outside
    alpha_channel = np.where(mask == 255, 255, 0).astype(np.uint8)

    # Combine RGB and alpha to form RGBA image
    rgba_img = np.dstack((rgb_img, alpha_channel))

    # Save result
    output_name = f'transparent_crop_{file_name}'
    cv2.imwrite(output_name, rgba_img)
    print(f"Saved cropped sector with transparency as: {output_name}")

# Example usage
#transparent_sector_crop('frame327.png')
#transparent_sector_crop('Sample_US_img.png')
