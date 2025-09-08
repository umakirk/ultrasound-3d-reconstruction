# Full segmentation function

import cv2
import numpy as np
import os

def segment(image_filename, masktype, threshold = None, blockSize = None, C = None, k = None, dots_kernel_size = None, min_blob_size = None, max_blob_size = None, blur_kernel_size = None, inverted = False):
    img = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    try:
        base_filename = f"{os.path.splitext(os.path.basename(image_filename))[0]}.png"
    except:
        base_filename = image_filename

    if masktype == "simple":
        
        mask = np.where(img >= threshold, 255, 0).astype(np.uint8)

    elif masktype == "mean_adaptive":
        
        mask = cv2.adaptiveThreshold(
            src=img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,  # Mean of the neighborhood
            thresholdType=cv2.THRESH_BINARY,
            blockSize=blockSize,   # Size of neighborhood (must be odd and >1) 
                            # Width and height (in pixels) of the square neighborhood 
            C=C            # Constant subtracted from mean
                            # Negative number good if ROI is brighter than background
        )

    elif masktype == "gaussian_adaptive":
        mask = cv2.adaptiveThreshold(
            src=img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Gaussian window
            thresholdType=cv2.THRESH_BINARY,
            blockSize=blockSize,   # Size of neighborhood (must be odd)
            C=C             # Constant subtracted from weighted mean
        )

    elif masktype == "sauvola":
        
        from skimage.util import view_as_windows

        # Parameters
        #blockSize = 15  # Odd number
        #k = 0.7         # Controls how much std dev influences threshold
        R = 128          # dynamic range for std dev in 8-bit images. usually 128

        # Pad the image to handle borders
        pad = blockSize // 2
        padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)

        # Create sliding windows
        windows = view_as_windows(padded_img, (blockSize, blockSize))

        # Compute local means and stds
        means = np.mean(windows, axis=(2, 3))
        stds = np.std(windows, axis=(2, 3))

        # Sauvola threshold matrix
        sauvola_thresholds = means * (1 + k * ((stds / R) - 1))

        # Apply threshold
        mask = (img > sauvola_thresholds).astype(np.uint8) * 255


    elif masktype == "otsu":
                
        _, mask = cv2.threshold(
            img,                # input image
            0,                  # initial threshold (ignored)
            255,                # max value for binary
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    
    #cv2.imwrite(f"{masktype}_mask_{base_filename}.png", mask)
    from cleaning import clean
    cleaned_mask = clean(mask, masktype, base_filename, dots_kernel_size, min_blob_size, max_blob_size, blur_kernel_size)
    
    from apply_mask import apply
    if inverted == False:
        apply(cleaned_mask, masktype, image_filename, base_filename, inverted = False)
    else:
        apply(cleaned_mask, masktype, image_filename, base_filename, inverted = True)

    return cleaned_mask


segment('test_files/phantom_scan.png', 'simple', threshold = 10, dots_kernel_size = 11, min_blob_size = 100000, blur_kernel_size = 91)


segment('test_files/inverted_IJcarotid.png', 'simple', threshold = 215, dots_kernel_size = 3, min_blob_size = 1500, max_blob_size = 10000 , blur_kernel_size = 31, inverted = True)


segment('test_files/inverted_carotid.png', 'simple', threshold = 235, dots_kernel_size = 3, min_blob_size = 2000, max_blob_size = 2900 , blur_kernel_size = 31, inverted = True)


# from pydicom import dcmread
# carotid_img_dicom = dcmread("test_files/Carotid img sample")
# pixel_array = carotid_img_dicom.pixel_array
# #print("Shape:", pixel_array.shape)  # (577, 649, 850, 3)
# grayscale_array = np.mean(pixel_array, axis=-1).astype(np.uint8)  # shape: (577, 649, 850)
# cv2.imwrite("carotid_dicom.png", grayscale_array)
# from crop import sector_crop
# sector_crop("carotid_dicom.png")

# img = cv2.imread("test_files/crop_carotid_dicom.png")
# img_inverted = cv2.bitwise_not(img)
# cv2.imwrite("inverted_carotid_dicom.png", img_inverted)

segment('test_files/inverted_carotid_dicom.png', 'simple', threshold = 135, dots_kernel_size = 27, min_blob_size = 4500, max_blob_size = 5000 , blur_kernel_size = 51, inverted = True)


# from pydicom import dcmread
# dicom = dcmread("test_files/Forearm bone img sample")
# pixel_array = dicom.pixel_array
# #print("Shape:", pixel_array.shape)  # (577, 649, 850, 3)
# grayscale_array = np.mean(pixel_array, axis=-1).astype(np.uint8)  # shape: (577, 649, 850)
# cv2.imwrite("dicom.png", grayscale_array)
# from crop import sector_crop
# sector_crop("dicom.png")

# img = cv2.imread("crop_dicom.png")
# img_inverted = cv2.bitwise_not(img)
# cv2.imwrite("inverted_bone.png", img_inverted)

segment('test_files/inverted_carotid_dicom.png', 'simple', threshold = 135, dots_kernel_size = 27, min_blob_size = 4500, max_blob_size = 5000 , blur_kernel_size = 51, inverted = True)
