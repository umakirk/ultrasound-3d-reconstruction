import numpy as np
import os
import nibabel as nib
import shutil
from unified_check_helper_funcs.metadata_removal import remove_metadata, template


def load_inference_data(frames_array, name):

    # Clean frames
    array = [remove_metadata(frames_array[frame_num], template) for frame_num in range(frames_array.shape[0])]
    print(f"num frames after metadataremova: {len(array)}")

    # Output folders
    images_out_dir = f"inference_data/{name}"

    # Clear existing folder if it exists
    if os.path.exists(images_out_dir):
        shutil.rmtree(images_out_dir)
    os.makedirs(images_out_dir)


    # Save each frame
    for i, img in enumerate(array):
        # Normalize image and convert to float32
        img = img.astype(np.float32)

        # Add channel axis to image (shape: H, W, 1)
        img_3d = img[..., np.newaxis]

        # Save with matching names (no _0000)
        images_base_name = f"kidney_{i+1:04d}_0000.nii.gz"
        
        nib.save(nib.Nifti1Image(img_3d, affine=np.eye(4)), os.path.join(images_out_dir, images_base_name))

    print(f"Saved {len(array)} frames to '{images_out_dir}'")
