import numpy as np
import scipy.ndimage as ndi

def filter_small_segments(segmentation_array: np.ndarray, min_area: int) -> np.ndarray:
    """
    Filters out connected components smaller than min_area in each 2D frame of a 3D segmentation array.

    Parameters:
    - segmentation_array (np.ndarray): 3D array of shape (frames, height, width), binary or labeled.
    - min_area (int): Minimum area (in pixels) that a component must have to be kept.

    Returns:
    - np.ndarray: Filtered 3D segmentation array with small objects removed.
    """
    filtered = np.zeros_like(segmentation_array)
    
    for i in range(segmentation_array.shape[0]):
        frame = segmentation_array[i]
        labeled_frame, num_features = ndi.label(frame)
        sizes = ndi.sum(frame, labeled_frame, index=range(1, num_features + 1))

        for label_idx, size in enumerate(sizes, start=1):
            if size >= min_area:
                filtered[i][labeled_frame == label_idx] = 1  # Keep object

    return filtered


if __name__ == "__main__":

    # seg_array = np.load(r"pass4_groundtruth_UNet_segmentations.npy")
    # filtered_seg = filter_small_segments(seg_array, min_area=10000)

    # Save result if needed
    # np.save('filtered_segmentation_array.npy', filtered_seg)
    pass
