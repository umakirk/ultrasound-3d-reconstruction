import numpy as np
from add_artifacts import motion_blur_frames, add_fuzzy_shadow
from UNet_setup_code.load_testing_dataset import load_inference_data
import subprocess
from UNet_setup_code.edit_predict_script import write_predict_script
from UNet_setup_code.nifti_predictions_to_array import nifti_segmentations_to_array
from filter_segmentations import filter_small_segments
from multipass_PNN import PNN_reconstruct
from generate_mesh import mesh_from_voxel_grid
import pyvista as pv


from unified_check_helper_funcs.metadata_removal import remove_metadata
from unified_check_helper_funcs.blur_detection_per_frame import detect_blur_in_frame
from unified_check_helper_funcs.depth_varying_noncontact_detection import detect_noncontact_realtime
from unified_check_helper_funcs.shadow_detection import (
    find_sector_corners, generate_rays, ray_intensity_profiles,
    shadow_segments, ray_shadow_mask, filter_standalone_shadow_rays,
    smooth_and_fill_shadow_mask, extract_largest_shadows
)

##### Default artifact detection parameters #####
noncontact_params = dict(threshold_factor=0.75, top_percent=2, min_valid_ratio=0.15, width_change_thresh=10)
blur_params = dict(tenengrad_threshold=11000, wavelet_threshold=300)
shadow_params = {
    "dark": dict(shadow_threshold=20, min_shadow_length=10, shadow_min_threshold=0),
    "medium": dict(shadow_threshold=60, min_shadow_length=33, shadow_min_threshold=30),
    "PAE": dict(shadow_threshold=255, min_shadow_length=10, shadow_min_threshold=200)
}

noncontact_detect = True
blur_detect = True
ray_detections = ["dark", "medium", "PAE"]



###################################################################################################################

##### USER INPUTS #####

# 1) Define a name for this mesh (pass number, type of artifact added if any)
# DON'T specify if detection is being run, will automatically be added to the name later
#name = "pass5_motionblur50_100"
#name = "pass3,4,5,6_overlapfiltered"

name = "poster_artifact_added_pass5"


# 2) Pick what indices from multipass scan to use
pass1_indices = np.arange(175,331)
pass2_indices = np.arange(395,603)
pass3_indices = np.arange(683,874)
pass4_indices = np.arange(929,1114)
pass5_indices = np.arange(1183, 1383)
pass6_indices = np.arange(1445,1640)


#indices = np.concatenate((pass1_indices, pass2_indices))

#indices = np.concatenate((pass3_indices, pass4_indices, pass5_indices, pass6_indices))

indices = pass5_indices

# (Slicing the multipass 2375 frames scan to get frames at those specific indices)
#unartifact_frames = np.load("multipass_givens/multipass_phantom(2375, 649, 850).npy")[indices]

# 3) Optional: Add artifacts using add_artifacts.py helper funcs #####
#frames = motion_blur_frames(frames, range(50, 100), kernel_size=100, direction='horizontal')

# frames = add_fuzzy_shadow(
#     scan_array=unartifact_frames,
#     rect_size=(250, 175),      # height=60, width=40
#     top_mid=(150, 425),       # top edge midpoint at row=80, col=128
#     frame_range=(unartifact_frames.shape[0]//3, 2*unartifact_frames.shape[0]//3),
#     fuzz_sigma=8
# )


### OR LOAD FRAMES DIRECTLY HERE

#frames = np.load(r"artifact_added_arrays/pass1and2_shadowed_frames.npy")

#frames = np.load(r"artifact_added_arrays/pass3,4_shadowed_frames.npy")
#frames = np.load(r"artifact_added_arrays/pass3,4,5_shadowed_frames.npy")

#frames = unartifact_frames

frames = np.load("poster_artifact_added_pass5_frames.npy")


# Save artifact-added numpy array in folder artifact_added_arrays
#np.save(f"artifact_added_arrays/{name}_frames.npy", frames)
#print("Saved artifact-added frames to numpy array.")


# 5) Toggle artifact detection on or off
artifact_detection = False
if artifact_detection:
    name = "detected_" + name

### Change artifact detection parameters here if needed
noncontact_detect = False
blur_detect = True
ray_detections = []


###################################################################################################################

##### Run artifact detection BEFORE SEGMENTATION (if doing artifact detection)
if artifact_detection:

    num_frames = frames.shape[0]
    h, w = frames.shape[1:]
    artifact_maps = np.zeros((num_frames, h, w), dtype=np.uint8)

    # Step 1: Noncontact detection (precomputed)
    if noncontact_detect:
        noncontact_flags, _ = detect_noncontact_realtime(frames, **noncontact_params)

    # Step 2: Blur + Ray-based artifact detection (per frame)
    keep_indices = []

    for i in range(num_frames):
        frame = frames[i]
        unclean_frame = frame.copy()

        # Skip frame if noncontact
        if noncontact_detect and noncontact_flags[i]:
            continue

        # Skip frame if blur
        if blur_detect:
            blur_flag, _ = detect_blur_in_frame(unclean_frame, **blur_params)
            if blur_flag:
                continue

        keep_indices.append(i)

        # Run ray-based artifact detection
        artifact_map = np.zeros_like(frame, dtype=np.uint8)

        if ray_detections:
            cleaned_frame = remove_metadata(frame)

            corners = find_sector_corners(cleaned_frame, transducer="CURVED LINEAR")
            rays = generate_rays(corners, num_rays=200, points_per_ray=100, transducer="CURVED LINEAR")
            profiles = ray_intensity_profiles(cleaned_frame, rays)

            for idx, key in enumerate(ray_detections):
                segs = shadow_segments(profiles, **shadow_params[key])
                raw_mask = ray_shadow_mask(rays, segs, frame.shape)
                filtered = filter_standalone_shadow_rays(raw_mask, rays, segs, neighbor_range=1)
                filled = smooth_and_fill_shadow_mask(filtered, dilation_iter=4, closing_iter=3)
                final_mask = extract_largest_shadows(filled, min_area=500)

                artifact_map[final_mask > 0] |= (1 << idx)

            artifact_maps[i] = artifact_map

    # Step 3: Filter frames and artifact_maps
    keep_indices = np.array(keep_indices)
    frames = frames[keep_indices]
    artifact_maps = artifact_maps[keep_indices]

    # Save updated frames and artifact_maps for later use
    np.save(f"artifact_detected_data/{name}_filtered_frames.npy", frames)
    np.save(f"artifact_detected_data/{name}_artifact_maps.npy", artifact_maps)

    print(f"Artifact detection done! Kept {len(frames)} frames out of {num_frames}.")
    print(f"Discarded {num_frames - len(keep_indices)} frames due to artifact detection.")
    print("Saved filtered frames and artifact maps in artifact_detected_data folder.")


###################################################################################################################

##### Get segmentations using UNet #####

# Create inference data nifti files, save to properly named folder in inference_data
load_inference_data(frames, name)
print("Inference data nifti files created on local computer.")

scp_upload_data_command = [
    "scp",
    "-r",
    f"C:/Users/umakirk/Desktop/nnUnet_all/inference_data/{name}",
    f"umakirk@orcd-login001.mit.edu:/home/umakirk/inference_data/"
]

try:
    # Run the command and wait for it to complete
    print("Uploading inference data to Engaging Home Directory...")

    result = subprocess.run(scp_upload_data_command, check=True, text=True, capture_output=True)
    print("Inference data upload to Engaging Home Directory completed successfully!")
    #print("Output:", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error during inference data upload to Engaging Home Directory:")
    print(e.stderr)


# Update predict.sh to use proper paths, then upload updated script to Engaging Home Directory
write_predict_script(name, output_path="engaging_sh_scripts/predict.sh")
scp_upload_script_command = [
    "scp",
    f"C:/Users/umakirk/Desktop/nnUnet_all/engaging_sh_scripts/predict.sh",
    f"umakirk@orcd-login001.mit.edu:/home/umakirk/predict.sh"
]

try:
    subprocess.run(scp_upload_script_command, check=True, text=True)
    print("predict.sh uploaded to Engaging Home Directory successfully!")
except subprocess.CalledProcessError as e:
    print("Error uploading predict.sh to Engaging Home Directory:")
    print(e.stderr)


### NOW MANUALLY LOG INTO SSH AND SUBMIT sbatch predict.sh

### WAIT FOR SBATCH TO FINISH RUNNING BEFORE TRYING TO DOWNLOAD
input("Manually submit predict.sh script. Press Enter to download predictions after the job is done...")


# Download prediction segmentation nifti files from Engaging Home Directory to my computer folder
scp_download_data_command = [
    "scp",
    "-r",
    f"umakirk@orcd-login001.mit.edu:/home/umakirk/inference_predictions/{name}",
    f"C:/Users/umakirk/Desktop/nnUnet_all/inference_predictions/" 
]

try:
    # Run the command and wait for it to complete
    result = subprocess.run(scp_download_data_command, check=True, text=True, capture_output=True)
    print("Inference predictions download to local computer completed successfully!")
    #print("Output:", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error during inference predictions download to local computer:")
    print(e.stderr)


# Run nifti_predictions_to_array.py

nifti_segmentations_to_array(name)
print("Inference prediction nifti files saved as 3D array on local computer.")

###################################################################################################################

# Run PNN recon

poses = np.load("multipass_givens/multipass_poses_centered(3, 4, 2375).npy")[:, :, indices]

if artifact_detection:
    poses = poses[:, :, keep_indices]

segmentations = np.load(f"UNet_seg_arrays/{name}_UNet_segmentations.npy")
#segmentations = np.load(r"UNet_seg_arrays\multipass_all2375_UNet_segmentations.npy")[indices]
segmentations = filter_small_segments(segmentations, min_area=10000)

segmented_frames = np.where(segmentations != 0, frames, 0)
voxel_grid, min_coords, elapsed_time, point_cloud = PNN_reconstruct(segmented_frames, poses)

np.save(f"voxel_grid_reconstructions/{name}_reconstruction", voxel_grid)

print("PNN reconstruction completed!")
print("Voxel grid shape:", voxel_grid.shape)
print("Origin/min_coords:", min_coords)
print("Elapsed time:", elapsed_time)



# Save and mesh
np.save(f"voxel_grid_reconstructions/{name}_common_reconstruction", voxel_grid)

###################################################################################################################

##### Get surface mesh from generate_mesh.py #####


mesh = mesh_from_voxel_grid(voxel_grid, voxel_size = 0.001)
print("Generated surface mesh from voxel grid!")

# Compute volume
volume_m3 = mesh.volume
volume_ml = volume_m3 * 0.001  # mm³ to mL
print(f"Estimated volume: {volume_ml:.2f} mL")

# Save .ply file
#mesh.save(f"surfaces/{name}_surface.ply")
mesh.save(f"surfaces/{name}_common_surface.ply")


# Visualize
plotter = pv.Plotter()
plotter.add_mesh(mesh, color='lightblue', show_edges=False)
plotter.add_axes()
plotter.show_grid()
plotter.show()
