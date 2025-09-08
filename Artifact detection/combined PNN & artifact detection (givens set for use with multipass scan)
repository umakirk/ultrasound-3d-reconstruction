import numpy as np
from collections import defaultdict
from pydicom import dcmread

# Importing my helper functions
from unified_check_helper_funcs.load_dicom import get_dicom_array
from unified_check_helper_funcs.metadata_removal import remove_metadata
from unified_check_helper_funcs.blur_detection_per_frame import detect_blur_in_frame
from unified_check_helper_funcs.depth_varying_noncontact_detection import detect_noncontact_realtime
from unified_check_helper_funcs.shadow_detection import (find_sector_corners, generate_rays, ray_intensity_profiles,
    shadow_segments, ray_shadow_mask, filter_standalone_shadow_rays,
    smooth_and_fill_shadow_mask, extract_largest_shadows)

from add_artifacts import blur_dicom_array


# Givens/constants
template = np.load("unified_check_helper_funcs/LOGIC_E9.npy")
UStoCam = np.array([
[0,       0,     -0.0398],
[-1.5813e-04, 0,     -0.0175],
[0,      -1.5813e-04, -0.1065],
[0,       0,      1.0000]
])
xoffset = 425
yoffset = 71


def real_time_weighted_pnn(
    dicom_array, segmentation_masks, poses, 
    grid_size, min_coords, voxel_size, 
    artifact_weights={0: 0.0, 1: 0.3, 2: 0.5}, noncontact_detect = True, blur_detect = True, dark_shadow_detect = True, 
                medium_shadow_detect = True, PAE_detect = True,
    noncontact_params=None, blur_params=None, shadow_params=None):

    ##### Set default detection parameters if not already user-inputted #####
    if noncontact_params is None:
        noncontact_params = dict(threshold_factor=0.75, top_percent=2, min_valid_ratio=0.15, width_change_thresh=10)

    if blur_params is None:
        blur_params = dict(tenengrad_threshold=11000, wavelet_threshold=300)
    
    if shadow_params is None:
        shadow_params = {
            "dark": dict(shadow_threshold=20, min_shadow_length=10, shadow_min_threshold=0),
            "medium": dict(shadow_threshold=60, min_shadow_length=33, shadow_min_threshold=30),
            "PAE": dict(shadow_threshold=255, min_shadow_length=10, shadow_min_threshold=200)
        }
    
    
    ### Create list of ray detections that are toggled on, to loop through easily
    ray_detections = []
    if dark_shadow_detect:
        ray_detections.append("dark")
    if medium_shadow_detect:
        ray_detections.append("medium")
    if PAE_detect:
        ray_detections.append("PAE")


    ##### Read in DICOM, segmentation mask, poses matrix from paths #####
    # dicom = dcmread(dicom_path)
    # frames = get_dicom_array(dicom)

    frames = dicom_array
    num_frames = frames.shape[0]
    #transducer = dicom.TransducerType

    # segmentation_masks = np.load(segmentation_masks_path)
    # poses = np.load(poses_path)
    poses = poses.transpose(2, 0, 1) # (#scans, 3, 4)


    ##### Initialize empty voxel grid using user-inputted grid_size #####
    voxel_grid = np.zeros(grid_size, dtype=np.uint8)
    voxel_sums = defaultdict(float)
    voxel_weights = defaultdict(float)

 
    ##### Run artifact detection #####

    # TEMPORARY: precomputing noncontact flags, change to real-time later
    if noncontact_detect:
        noncontact_flags, _ = detect_noncontact_realtime(frames, **noncontact_params)
        print("Noncontact frames flagged:", (np.count_nonzero(noncontact_flags)))

    # Iterate through each frame (mimicks flow of real-time data acquisition and reconstruction)
    for i in range(num_frames):
        unclean_frame = frames[i]

        # Remove metadata from frame
        frame = remove_metadata(unclean_frame, template)

        # Skip this frame if flagged for noncontact or blur
        if noncontact_detect and noncontact_flags[i]:
            continue
        # if detect_noncontact_realtime(frame):
        #     continue

        # RUNNING BLUR DETECTION ON UNCLEAN FRAME BC THATS WHAT I CALCULATED THRESHOLDS ON
        if blur_detect:
            blur_flag, _ = detect_blur_in_frame(unclean_frame, **blur_params)
            if blur_flag:
                print("blur flagged")
                continue
        

        artifact_map = np.zeros_like(frame, dtype=np.uint8)
        if ray_detections:
            # Shadow + PAE detection on full frame --> build artifact_map
            corners = find_sector_corners(frame, transducer = "CURVED LINEAR")
            rays = generate_rays(corners, num_rays=200, points_per_ray=100, transducer= "CURVED LINEAR")
            profiles = ray_intensity_profiles(frame, rays)

            #artifact_map = np.zeros_like(frame, dtype=np.uint8)

            for idx, key in enumerate(ray_detections):
                segs = shadow_segments(profiles, **shadow_params[key])
                raw_mask = ray_shadow_mask(rays, segs, frame.shape)
                filtered = filter_standalone_shadow_rays(raw_mask, rays, segs, neighbor_range=1)
                filled = smooth_and_fill_shadow_mask(filtered, dilation_iter=4, closing_iter=3)
                final_mask = extract_largest_shadows(filled, min_area=500)

                artifact_map[final_mask > 0] |= (1 << idx)


        # Apply segmentation mask to frame and artifact_map
        seg_mask = segmentation_masks[i] > 0
        masked_frame = np.where(seg_mask, frame, 0)
        masked_artifact_map = np.where(seg_mask, artifact_map, 0)

        


        ##### Incorporate this frame into weighted PNN reconstruction #####
            # Use poses to find real world coordinates of every pixel in a frame
            # Compute cumulative weight w for each pixel

        # Starting info: frame (contains row, col, intensity data for each pixel), pose, artifact_map
        # Resulting info (for each non-zero pixel in frame): X, Y, Z, intensity, cumulative weight

        # Get pose
        pose = poses[i]
        
        # Compute world coordinates
        rows, cols = masked_frame.shape
        x_coords = np.arange(cols) - xoffset
        y_coords = np.arange(rows) - yoffset
        xx, yy = np.meshgrid(x_coords, y_coords)
        flat_x = xx.ravel()
        flat_y = yy.ravel()
        flat_ones = np.ones_like(flat_x)

        flat_intensity = masked_frame.ravel()
        flat_artifacts = masked_artifact_map.ravel()

        nonzero_mask = flat_intensity != 0
        pixel_coords = np.vstack([
            flat_x[nonzero_mask],
            flat_y[nonzero_mask],
            flat_ones[nonzero_mask]
        ])  # shape (3, N)

        coords = (pose @ UStoCam @ pixel_coords).T



        # Compute weight per pixel

        intensities = flat_intensity[nonzero_mask]
        artifact_bits = flat_artifacts[nonzero_mask]
        weights = np.ones_like(intensities, dtype=np.float32)
        for bit in artifact_weights:
            mask = (artifact_bits & (1 << bit)) != 0
            weights[mask] *= artifact_weights[bit]


        #Filter out any pixels with total weight 0
        valid = weights > 0
        valid_coords = coords[valid]
        valid_intensities = intensities[valid]
        valid_weights = weights[valid]

            
        # Map each pixel to nearest voxel
        voxel_indices = np.floor((valid_coords - min_coords) / voxel_size).astype(int)

        # For each voxel: keep track of sum of weighted intensities and sum of weights
        for idx, val, w in zip(voxel_indices, valid_intensities, valid_weights):
            key = tuple(idx)
            voxel_sums[key] += val * w
            voxel_weights[key] += w

        # Update only voxels touched by the current frame (rather than re-writing every voxel seen so far)
        updated_keys = {tuple(idx) for idx in voxel_indices}
        for key in updated_keys:
            w = voxel_weights[key]
            if w > 0:
                avg = voxel_sums[key] / w
                voxel_grid[key] = np.clip(int(round(avg)), 0, 255)


    return voxel_grid



if __name__ == "__main__":   

    ### Setting all artifact detections toggled on
    noncontact_detect = True
    blur_detect = True
    dark_shadow_detect = True
    medium_shadow_detect = True
    PAE_detect = True

    ### Setting default parameters
    shadow_params = {
        "dark": dict(shadow_threshold=20, min_shadow_length=10, shadow_min_threshold=0),
        "medium": dict(shadow_threshold=60, min_shadow_length=33, shadow_min_threshold=30),
        "PAE": dict(shadow_threshold=255, min_shadow_length=10, shadow_min_threshold=200)
    }
    blur_params = dict(tenengrad_threshold=11000, wavelet_threshold=300)
    noncontact_params = dict(threshold_factor=0.75, top_percent=2, min_valid_ratio=0.15, width_change_thresh=10)




    dicom_array = np.load(r"multipass_phantom\multipass_phantom(2375, 649, 850).npy")[918:1138]
    segmentation_masks = np.load(r"multipass_phantom\multipass_segmentations(2375, 649, 850).npy")[918:1138]
    poses = np.load(r"multipass_phantom\multipass_poses_centered(3, 4, 2375).npy")[:, :, 918:1138]
    noncontact_detect = False
    blur_detect = False
    dark_shadow_detect = False
    medium_shadow_detect = False
    shadow_params["PAE"] = dict(shadow_threshold=255, min_shadow_length=25, shadow_min_threshold=75)


    grid_size = (147, 107, 57)
    min_coords = np.array([-0.07118221, -0.04709152, -0.01415395])



    ### ARTIFICALLY ADDING ARTIFACTS
    #dicom_array = blur_dicom_array(dicom_array, range(180, 210), sigma_value = 5)

    voxel_grid= real_time_weighted_pnn(
        dicom_array, segmentation_masks, poses, 
        grid_size, min_coords, voxel_size=0.001, 
        artifact_weights={0: 0.0, 1: 0.3, 2: 0.5},
        noncontact_params=noncontact_params, blur_params=blur_params, shadow_params=shadow_params,
        noncontact_detect = noncontact_detect, blur_detect = blur_detect, dark_shadow_detect = dark_shadow_detect, 
        medium_shadow_detect = medium_shadow_detect, PAE_detect = PAE_detect
        )

    np.save("combined_PAE_pass4.npy", voxel_grid)
