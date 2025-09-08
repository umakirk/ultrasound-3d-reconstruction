import numpy as np
from scipy import ndimage
from unified_check_helper_funcs.load_dicom import get_dicom_array
from pydicom import dcmread


def blur_frames(image_array, blur_indices, sigma_value):
    blurred_array = image_array.copy()

    for i in blur_indices:
        # Higher sigma values result in more blurring.
        frame = image_array[i]
        blurred_frame = ndimage.gaussian_filter(frame, sigma=sigma_value)
        blurred_array[i] = blurred_frame

    blurred_array = blurred_array.astype(image_array.dtype)

    return blurred_array
