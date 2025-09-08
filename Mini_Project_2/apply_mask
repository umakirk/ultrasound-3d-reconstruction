# Apply segmentation mask over original image

import cv2
import numpy as np
import os


def apply_mask(mask_filename, image_filename, inverted = False):
    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    if inverted == True:
        img = 255 - img
    base_filename = os.path.splitext(os.path.basename(image_filename))[0]

    
    segmented_img = np.where(mask>0, img, 0).astype(np.uint8)

    cv2.imwrite(f"segmented_{base_filename}_{mask_filename}.png", segmented_img)



#apply_mask('allclean_simple_mask.png', 'test_files/phantom_scan.png')


# same as apply_mask but reads in mask itself instead of mask filename
def apply(mask, mask_filename, image_filename, base_filename, inverted = False):
    img = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    if inverted == True:
        img = 255 - img
    base_filename = os.path.splitext(os.path.basename(image_filename))[0]

    
    segmented_img = np.where(mask>0, img, 0).astype(np.uint8)

    cv2.imwrite(f"segmented_{mask_filename}_{base_filename}.png", segmented_img)
