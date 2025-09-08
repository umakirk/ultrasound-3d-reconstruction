import numpy as np
import os
import nibabel as nib

# ---- Load arrays ----
array = np.load(r"multipass_phantom\multipass_phantom(2375, 649, 850).npy")[400:575]  # shape (175, H, W)
seg_array = np.load(r"multipass_phantom\multipass_segmentations(2375, 649, 850).npy")[400:575]

# ---- Sample every 3rd frame ----
sampled_indices = np.arange(0, 175, 3)  # 59 frames
sampled_imgs = array[sampled_indices]
sampled_segs = seg_array[sampled_indices]

# ---- Output folders ----
images_out_dir = r"C:\Users\umakirk\nnUNet_raw\Dataset100_KidneyUS\imagesTr"
labels_out_dir = r"C:\Users\umakirk\nnUNet_raw\Dataset100_KidneyUS\labelsTr"
os.makedirs(images_out_dir, exist_ok=True)
os.makedirs(labels_out_dir, exist_ok=True)

# ---- Save each frame ----
for i, (img, seg) in enumerate(zip(sampled_imgs, sampled_segs)):
    # Normalize image and convert to float32
    img = img.astype(np.float32)
    seg = (seg > 0).astype(np.uint8)

    # Add channel axis to image (shape: H, W, 1)
    img_3d = img[..., np.newaxis]
    seg_3d = seg[..., np.newaxis]

    # Save with matching names (no _0000)
    images_base_name = f"kidney_{i+1:03d}_0000.nii.gz"
    labels_base_name = f"kidney_{i+1:03d}.nii.gz"
    
    nib.save(nib.Nifti1Image(img_3d, affine=np.eye(4)), os.path.join(images_out_dir, images_base_name))
    nib.save(nib.Nifti1Image(seg_3d, affine=np.eye(4)), os.path.join(labels_out_dir, labels_base_name))

print(f"Saved {len(sampled_imgs)} images and labels as NIfTI files (no _0000 suffix)")
