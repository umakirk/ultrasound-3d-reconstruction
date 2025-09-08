import numpy as np
from scipy import ndimage
from unified_check_helper_funcs.load_dicom import get_dicom_array
from pydicom import dcmread
import cv2


def motion_blur_frames(image_array, blur_indices, kernel_size=15, direction='horizontal'):
    """
    Applies directional motion blur to selected frames.
    direction: 'vertical' or 'horizontal'
    """
    blurred_array = image_array.copy()

    # Create directional motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    if direction == 'vertical':
        kernel[:, kernel_size // 2] = 1.0
    elif direction == 'horizontal':
        kernel[kernel_size // 2, :] = 1.0
  

    kernel /= kernel_size  # Normalize

    for i in blur_indices:
        frame = image_array[i]
        motion_blurred = cv2.filter2D(frame, -1, kernel)
        blurred_array[i] = motion_blurred

    blurred_array = blurred_array.astype(image_array.dtype)
    return blurred_array


import numpy as np
from scipy.ndimage import gaussian_filter

def add_fuzzy_shadow(
    scan_array,
    rect_size,          # (shadow_height, shadow_width)
    top_mid,            # (top_row, mid_col)
    frame_range,        # (start_frame, end_frame) inclusive
    fuzz_sigma=5        # controls blur/edge softness
):
    """
    Add a fuzzy-edged black shadow rectangle to specific frames in a 3D ultrasound scan.
    The rectangle is defined by the midpoint of its top edge.
    """
    shadow_height, shadow_width = rect_size
    top_row, mid_col = top_mid
    start_frame, end_frame = frame_range

    # Compute top-left based on midpoint
    left_col = int(mid_col - shadow_width // 2)

    # Ensure valid frame range
    start_frame = max(0, start_frame)
    end_frame = min(scan_array.shape[0] - 1, end_frame)

    # Create a mask for the shadow rectangle
    mask = np.zeros((scan_array.shape[1], scan_array.shape[2]), dtype=float)
    mask[top_row:top_row+shadow_height, left_col:left_col+shadow_width] = 1.0

    # Blur the mask for soft edges
    fuzzy_mask = gaussian_filter(mask, sigma=fuzz_sigma)
    fuzzy_mask /= fuzzy_mask.max()  # Normalize to [0, 1]

    # Apply to each frame in the specified range
    for f in range(start_frame, end_frame + 1):
        scan_array[f] = scan_array[f] * (1 - fuzzy_mask)

    return scan_array


if __name__ == "__main__":

    # Pass 5
    image_array = np.load(r"multipass_givens\multipass_phantom(2375, 649, 850).npy")[1183:1382]

    motion_blur_hor_array = motion_blur_frames(image_array, range(50, 100), kernel_size=150, direction='horizontal')
    #motion_blur_vert_array = motion_blur_frames(motion_blur_hor_array, range(50, 85), kernel_size=150, direction='vertical')

    shadowed_array = add_fuzzy_shadow(
        scan_array=motion_blur_hor_array,
        rect_size=(450, 200),      # height=60, width=40
        top_mid=(275, 425),       # top edge midpoint at row=80, col=128
        frame_range=(130, 170),
        fuzz_sigma=50
    )
    
    np.save("poster_artifact_added_pass5_frames.npy", shadowed_array)
