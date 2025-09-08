import nibabel as nib
import numpy as np
import os


def nifti_segmentations_to_array(name):

    pred_dir = f"C:/Users/umakirk/Desktop/nnUnet_all/inference_predictions/{name}"
    
    output_array = []


    for fname in sorted(os.listdir(pred_dir)):
        if fname.endswith(".nii.gz"):
            img = nib.load(os.path.join(pred_dir, fname))
            data = img.get_fdata().astype(np.uint8)
            output_array.append(data[..., 0])  # assuming 2D slice

    output_array = np.stack(output_array, axis=0)  # shape: (frames, rows, cols)
    np.save(f"UNet_seg_arrays/{name}_UNet_segmentations.npy", output_array)
