# Removing small white dots, filling in small black dots

import cv2
import numpy as np


def dots(mask_filename, kernel_size):

    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

    # Structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Morphological opening: removes small white noise
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Morphological closing: fills small black holes
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(f"dots_{mask_filename}", cleaned)

# Removing small blobs/noise

def blobs(mask_filename, min_blob_size, max_blob_size):
    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

    # Ensure mask is in binary
    mask = (mask > 0).astype(np.uint8) * 255

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # keep only components larger than min_size
    output = np.zeros_like(mask)
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_blob_size and area <= max_blob_size: # blob area must be >= threshold
            output[labels == i] = 255

    cv2.imwrite(f"blobs_{mask_filename}", output)


# Smoothing using contours

def blur_smooth(mask_filename, kernel_size):

    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

        # Blur then re-threshold to binary
    blurred = cv2.GaussianBlur(mask.astype(np.uint8), (kernel_size, kernel_size), 0)
    smoothed = (blurred > 127).astype(np.uint8) * 255

    cv2.imwrite(f"blurred_{mask_filename}", smoothed)

def all_clean(mask_filename, dots_kernel_size, min_blob_size, max_blob_size, blur_kernel_size):

    
    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

    #DOTS CLEANING
    # Structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dots_kernel_size, dots_kernel_size))

    # Morphological opening: removes small white noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Morphological closing: fills small black holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #BLOBS CLEANING
    # Connected components

    # Ensure mask is in binary
    mask = (mask > 0).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # keep only components larger than min_size
    mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_blob_size and area <= max_blob_size: # blob area must be >= threshold
            mask[labels == i] = 255


        # Blur then re-threshold to binary
    mask = cv2.GaussianBlur(mask.astype(np.uint8), (blur_kernel_size, blur_kernel_size), 0)
    mask = (mask > 127).astype(np.uint8) * 255

    cv2.imwrite(f"allclean_{mask_filename}", mask)

#same as all_clean but reads in mask instead of mask_filename
def clean(mask, mask_filename, base_filename, dots_kernel_size, min_blob_size, max_blob_size, blur_kernel_size):

    #DOTS CLEANING
    # Structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dots_kernel_size, dots_kernel_size))

    # Morphological opening: removes small white noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Morphological closing: fills small black holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #BLOBS CLEANING
    # Connected components

    # Ensure mask is in binary
    mask = (mask > 0).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # keep only components larger than min_size
    mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if max_blob_size is None:
            if area >= min_blob_size: # blob area must be >= threshold
                mask[labels == i] = 255

        else:
            if area >= min_blob_size and max_blob_size is not None and area <= max_blob_size: # blob area must be >= threshold
                mask[labels == i] = 255

        # Blur then re-threshold to binary
    mask = cv2.GaussianBlur(mask.astype(np.uint8), (blur_kernel_size, blur_kernel_size), 0)
    mask = (mask > 127).astype(np.uint8) * 255

    cv2.imwrite(f"mask_{mask_filename}_{base_filename}", mask)
    
    return mask
