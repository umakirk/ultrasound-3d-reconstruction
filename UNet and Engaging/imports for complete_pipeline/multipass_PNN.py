import numpy as np
from collections import defaultdict
import time

def PNN_reconstruct(array_2D_scans, allPoses, voxel_size = 0.001, multiple_inputs = 'average',
                    scan_min = 0, scan_max = None, row_min = 0, row_max = None, 
                    col_min = 0, col_max = None):
    """
    Inputs:
        - array_2D_scans: a 3D array of shape (# 2D scans, # rows, # cols) storing 2D scans of size row x col.
        - allPoses: a 3D array of shape (3, 4, #2D scans) containing the pose of each scan.
            Each 3x4 matrix is the rotation matrix augmented with the transpose vector of a given scan.
        - voxel_size: size of square voxel, measured in meters
        - multiple_inputs: 'average', 'max', or 'recent', to indicate how to deal with multiple pixel inputs
            to the same voxel.
        - Optional: min and max values for scans, rows, and cols, to crop the data
    
    Output:
        - voxel_grid: a 3D NumPy array of type uint8 containing the reconstructed voxel intensities.
          Each voxel stores the average intensity (0-255) of all projected pixels that fall within it.
          The shape of the array is determined by the spatial bounds of the point cloud and the voxel size.

    """

    start_time = time.time()

    # Replace any None max values so that cropping works below
    if scan_max == None:
        scan_max = array_2D_scans.shape[0]
    if row_max == None:
        row_max = array_2D_scans.shape[1]
    if col_max == None:
        col_max = array_2D_scans.shape[2]
    
    # Cropping of data
    data = array_2D_scans[scan_min:scan_max, row_min:row_max, col_min:col_max]

    # Adjust offsets for cropped image
        # xoffset and yoffset get subtracted from col_location and row_location later in the XYZ formula.
        # So by subtracting x_min from xoffset, we are adding x_min to our col_location.
        # This is because after cropping our column numbers and row numbers get shift down to zero, this compensates.
    xoffset = 425 - col_min
    yoffset = 71 - row_min

    UStoCam = np.array([
[0,       0,     -0.0398],
[-1.5813e-04, 0,     -0.0175],
[0,      -1.5813e-04, -0.1065],
[0,       0,      1.0000]
])

    allPoses = allPoses[:, :, scan_min:scan_max] # getting rid of poses for any scans that were cropped out

    all_points = []

    # Iterate through every scan
    for scan_index in range(data.shape[0]):
        scan_2D = data[scan_index, :, :]
        framePose = allPoses[:, :, scan_index]

        rows, cols = scan_2D.shape

        # Meshgrid for pixel positions
        x_coords = np.arange(cols) - xoffset
        y_coords = np.arange(rows) - yoffset
        xx, yy = np.meshgrid(x_coords, y_coords)
        # xx: each row is a copy of x_coords
        # yy: each col is a copy of y_coords
        # Creates two coordinate matrices for easy indexing to find x,y coordinate of each pixel
            # xx[i, j] and yy[i, j] give the (x, y) pixel coordinate of the pixel at image location (i, j)

        # Flatten and prepare coordinates
        flat_x = xx.ravel()
        flat_y = yy.ravel()
        flat_ones = np.ones_like(flat_x)
        flat_intensity = scan_2D.ravel()
        # flat_x[i], flat_y[i] = (x, y) coordinate of the i-th pixel
        # flat_intensity[i] = intensity value at that pixel

        # Filter nonzero intensities
        nonzero_mask = flat_intensity != 0 #boolean mask

        pixel_coords = np.vstack([ # vertically stacks/concatenates to make matrix
            flat_x[nonzero_mask],
            flat_y[nonzero_mask],
            flat_ones[nonzero_mask]
        ])  # shape (3, N)

        intensities = flat_intensity[nonzero_mask]

        # Apply formula to get 3D point coordinates of all pixels in space
            # use @ to do numpy matrix multiplication
        coords = framePose @ UStoCam @ pixel_coords 

        points_from_scan = np.vstack([coords, intensities]).T  # shape (N, 4)
        all_points.append(points_from_scan)

    # Final point cloud
    point_cloud = np.vstack(all_points)  # shape (total_N, 4), where the 4 channels are x, y, z, intensity



    ### Voxelization from point cloud
    coords = point_cloud[:, :3]
    intensities = point_cloud[:, 3]

    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)

    # number of voxels needed in each dimension
    grid_shape = np.ceil((max_coords - min_coords) / voxel_size).astype(int) + 1
    
    # Mapping each point to a voxel index
    voxel_indices = np.floor((coords - min_coords) / voxel_size).astype(int)
    # Each point shifted so that min_coords is treated as the origin
    # Dividing by voxel_size gives its location in voxel space, round down to lower voxel


    if multiple_inputs == 'average':
        input_sums = defaultdict(float)
        input_counts = defaultdict(int)
        # Automatically adds unexisting keys when referenced, initializes to value 0 (int) or 0.0 (float) 

        for idx, intensity in zip(voxel_indices, intensities):
            key = tuple(idx)
            input_sums[key] += intensity
            input_counts[key] += 1

        voxel_grid = np.zeros(grid_shape, dtype=np.uint8)

        for key in input_sums:
            avg_intensity = input_sums[key] / input_counts[key]
            voxel_grid[key] = np.clip(int(round(avg_intensity)), 0, 255)
    
    elif multiple_inputs == 'max':
        input_lists = defaultdict(list) #initializes empty list for referencing keys that don't exist yet
        
        for idx, intensity in zip(voxel_indices, intensities):
            key = tuple(idx)
            input_lists[key].append(intensity)
        
        voxel_grid = np.zeros(grid_shape, dtype=np.uint8)

        for key in input_lists:
            max_intensity = max(input_lists[key])
            voxel_grid[key] = np.clip(int(round(max_intensity)), 0, 255)

    elif multiple_inputs == 'recent':
        input_recent = defaultdict(float)
        
        for idx, intensity in zip(voxel_indices, intensities):
            key = tuple(idx)
            input_recent[key] = intensity # gets overwritten
        
        voxel_grid = np.zeros(grid_shape, dtype=np.uint8)

        for key in input_recent:
            recent_intensity = input_recent[key]
            voxel_grid[key] = np.clip(int(round(recent_intensity)), 0, 255)

    end_time = time.time()
    elapsed_time = end_time - start_time

    return voxel_grid, min_coords, elapsed_time, point_cloud



def PNN_reconstruct_unused(array_2D_scans, allPoses, indices, voxel_size = 0.01, multiple_inputs = 'average',
                    scan_min = 0, scan_max = None, row_min = 0, row_max = None, 
                    col_min = 0, col_max = None):
    """
    Inputs:
        - array_2D_scans: a 3D array of shape (# 2D scans, # rows, # cols) storing 2D scans of size row x col.
        - allPoses: a 3D array of shape (3, 4, #2D scans) containing the pose of each scan.
            Each 3x4 matrix is the rotation matrix augmented with the transpose vector of a given scan.
        - voxel_size: size of square voxel, measured in meters
        - multiple_inputs: 'average', 'max', or 'recent', to indicate how to deal with multiple pixel inputs
            to the same voxel.
        - Optional: min and max values for scans, rows, and cols, to crop the data
    
    Output:
        - voxel_grid: a 3D NumPy array of type uint8 containing the reconstructed voxel intensities.
          Each voxel stores the average intensity (0-255) of all projected pixels that fall within it.
          The shape of the array is determined by the spatial bounds of the point cloud and the voxel size.

    """

    array_2D_scans = array_2D_scans[np.concatenate(indices)]
    allPoses = array_2D_scans[np.concatenate(indices)]
    

    start_time = time.time()

    # Replace any None max values so that cropping works below
    if scan_max == None:
        scan_max = array_2D_scans.shape[0]
    if row_max == None:
        row_max = array_2D_scans.shape[1]
    if col_max == None:
        col_max = array_2D_scans.shape[2]
    
    # Cropping of data
    data = array_2D_scans[scan_min:scan_max, row_min:row_max, col_min:col_max]

    # Adjust offsets for cropped image
        # xoffset and yoffset get subtracted from col_location and row_location later in the XYZ formula.
        # So by subtracting x_min from xoffset, we are adding x_min to our col_location.
        # This is because after cropping our column numbers and row numbers get shift down to zero, this compensates.
    xoffset = 425 - col_min
    yoffset = 71 - row_min

    UStoCam = np.array([
[0,       0,     -0.0398],
[-1.5813e-04, 0,     -0.0175],
[0,      -1.5813e-04, -0.1065],
[0,       0,      1.0000]
])
    
    

    allPoses = allPoses[:, :, scan_min:scan_max] # getting rid of poses for any scans that were cropped out

    all_points = []

    

    # Iterate through every scan
    for scan_index in range(data.shape[0]):
        scan_2D = data[scan_index, :, :]
        framePose = allPoses[:, :, scan_index]

        rows, cols = scan_2D.shape

        # Meshgrid for pixel positions
        x_coords = np.arange(cols) - xoffset
        y_coords = np.arange(rows) - yoffset
        xx, yy = np.meshgrid(x_coords, y_coords)
        # xx: each row is a copy of x_coords
        # yy: each col is a copy of y_coords
        # Creates two coordinate matrices for easy indexing to find x,y coordinate of each pixel
            # xx[i, j] and yy[i, j] give the (x, y) pixel coordinate of the pixel at image location (i, j)

        # Flatten and prepare coordinates
        flat_x = xx.ravel()
        flat_y = yy.ravel()
        flat_ones = np.ones_like(flat_x)
        flat_intensity = scan_2D.ravel()
        # flat_x[i], flat_y[i] = (x, y) coordinate of the i-th pixel
        # flat_intensity[i] = intensity value at that pixel

        # Filter nonzero intensities
        nonzero_mask = flat_intensity != 0 #boolean mask

        pixel_coords = np.vstack([ # vertically stacks/concatenates to make matrix
            flat_x[nonzero_mask],
            flat_y[nonzero_mask],
            flat_ones[nonzero_mask]
        ])  # shape (3, N)

        intensities = flat_intensity[nonzero_mask]

        # Apply formula to get 3D point coordinates of all pixels in space
            # use @ to do numpy matrix multiplication
        coords = framePose @ UStoCam @ pixel_coords 

        points_from_scan = np.vstack([coords, intensities]).T  # shape (N, 4)
        all_points.append(points_from_scan)

    # Final point cloud
    point_cloud = np.vstack(all_points)  # shape (total_N, 4), where the 4 channels are x, y, z, intensity


### Voxelization from point cloud
    coords = point_cloud[:, :3]
    intensities = point_cloud[:, 3]
    
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    grid_shape = np.ceil((max_coords - min_coords) / voxel_size).astype(int) + 1
    voxel_indices = np.floor((coords - min_coords) / voxel_size).astype(int)
    
    # Initialize grids
    voxel_grid = np.zeros(grid_shape, dtype=np.uint8)
    frame_counts_grid = np.zeros(grid_shape, dtype=np.uint16)  # new grid to track frame contributions

    if multiple_inputs == 'average':
        input_sums = defaultdict(float)
        input_counts = defaultdict(int)
        for idx, intensity in zip(voxel_indices, intensities):
            key = tuple(idx)
            input_sums[key] += intensity
            input_counts[key] += 1
        for key in input_sums:
            avg_intensity = input_sums[key] / input_counts[key]
            voxel_grid[key] = np.clip(int(round(avg_intensity)), 0, 255)
            frame_counts_grid[key] = input_counts[key]  # each voxel gets count of contributing pixels


    elif multiple_inputs == 'max':
        input_lists = defaultdict(list) #initializes empty list for referencing keys that don't exist yet
        
        for idx, intensity in zip(voxel_indices, intensities):
            key = tuple(idx)
            input_lists[key].append(intensity)
        
        voxel_grid = np.zeros(grid_shape, dtype=np.uint8)

        for key in input_lists:
            max_intensity = max(input_lists[key])
            voxel_grid[key] = np.clip(int(round(max_intensity)), 0, 255)

    elif multiple_inputs == 'recent':
        input_recent = defaultdict(float)
        
        for idx, intensity in zip(voxel_indices, intensities):
            key = tuple(idx)
            input_recent[key] = intensity # gets overwritten
        
        voxel_grid = np.zeros(grid_shape, dtype=np.uint8)

        for key in input_recent:
            recent_intensity = input_recent[key]
            voxel_grid[key] = np.clip(int(round(recent_intensity)), 0, 255)

    end_time = time.time()
    elapsed_time = end_time - start_time

    return voxel_grid, min_coords, elapsed_time, point_cloud, frame_counts_grid


if __name__ == "__main__":

    from filter_segmentations import filter_small_segments

    pass1_indices = np.arange(175,331)
    pass2_indices = np.arange(395,603)
    pass3_indices = np.arange(683,874)
    pass4_indices = np.arange(929,1114)
    pass5_indices = np.arange(1183, 1383)
    pass_6_indices = np.arange(1445,1640)
    
    # Pass 4
    #indices = np.arange(929, 1113)

    # Pass 5
    indices = np.arange(1183, 1382)

    # Pass 4&5
    #indices = np.concatenate((np.arange(929, 1114), np.arange(1183, 1383)))


    dicom_array = np.load(r"artifact_added_arrays\pass5_motionblur60_130_frames.npy")
    print(f"Number of frames: {dicom_array.shape[0]}")

    segmentation_masks = np.load(r'UNet_seg_arrays\pass5_motionblur60_130_UNet_segmentations.npy')
    segmentation_masks = filter_small_segments(segmentation_masks, min_area=10000)
    poses = np.load(r"multipass_givens\multipass_poses_centered(3, 4, 2375).npy")[:, :, indices]

 

    masked_dicom_array = np.where(segmentation_masks !=0 , dicom_array, 0)
    voxel_grid, origin, elapsed_time, point_cloud, frame_counts_grid = PNN_reconstruct(masked_dicom_array, poses, 0.001, multiple_inputs='average')

    np.save(r"voxel_grid_reconstructions/pass5_motionblur60_130_reconstruction", voxel_grid)





    print("Voxel grid shape:", voxel_grid.shape)
    print("Origin/min_coords:", origin)
    print("Elapsed time:", elapsed_time)
