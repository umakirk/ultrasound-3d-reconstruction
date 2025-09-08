import numpy as np
from collections import defaultdict
import time

def weighted_reconstruct(array_2D_scans, allPoses, voxel_size=0.01, method='gaussian',
                          kernel_radius=1, sigma=0.01, epsilon=1e-5,
                          scan_min=0, scan_max=None, row_min=0, row_max=None,
                          col_min=0, col_max=None):
    """
    Inputs:
        - array_2D_scans: (N_scans, rows, cols) ultrasound slices
        - allPoses: (3, 4, N_scans) pose matrix for each scan
        - voxel_size: voxel edge length (in meters)
        - method: 'gaussian', 'inverse', or 'solus'
        - kernel_radius: neighborhood size around pixel in voxel space
        - sigma: standard deviation for Gaussian OR thickness (for Solus)
        - epsilon: small constant to avoid divide-by-zero in inverse
        - *_min/_max: optional cropping bounds

    Outputs:
        - voxel_grid: (X, Y, Z) array of reconstructed volume
        - origin: real-world XYZ coordinate of voxel (0, 0, 0)
        - elapsed_time: in seconds
    """

    start_time = time.time()

    # Replace any None max values for cropping
    if scan_max is None:
        scan_max = array_2D_scans.shape[0]
    if row_max is None:
        row_max = array_2D_scans.shape[1]
    if col_max is None:
        col_max = array_2D_scans.shape[2]

    data = array_2D_scans[scan_min:scan_max, row_min:row_max, col_min:col_max]
    allPoses = allPoses[:, :, scan_min:scan_max]

    xoffset = 425 - col_min
    yoffset = 90 - row_min

    UStoCam = np.array([
        [0,       0,     -0.0398],
        [-0.0002, 0,     -0.0175],
        [0,      -0.0002, -0.1065],
        [0,       0,      1.0000]])

    all_points = []

    # First pass to get spatial bounds
    for scan_index in range(data.shape[0]):
        scan_2D = data[scan_index]
        framePose = allPoses[:, :, scan_index]
        rows, cols = scan_2D.shape

        x_coords = np.arange(cols) - xoffset
        y_coords = np.arange(rows) - yoffset
        xx, yy = np.meshgrid(x_coords, y_coords)

        flat_x = xx.ravel()
        flat_y = yy.ravel()
        flat_ones = np.ones_like(flat_x)
        flat_intensity = scan_2D.ravel()
        nonzero_mask = flat_intensity != 0

        pixel_coords = np.vstack([
            flat_x[nonzero_mask],
            flat_y[nonzero_mask],
            flat_ones[nonzero_mask]
        ])
        coords_3d = framePose @ UStoCam @ pixel_coords
        coords_3d = coords_3d.T
        all_points.append(coords_3d)

    all_points = np.vstack(all_points)
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)
    grid_shape = np.ceil((max_coords - min_coords) / voxel_size).astype(int) + 1

    value_sums = defaultdict(float)
    weight_sums = defaultdict(float)

    # Second pass to apply contributions
    for scan_index in range(data.shape[0]):
        scan_2D = data[scan_index]
        framePose = allPoses[:, :, scan_index]
        rows, cols = scan_2D.shape

        x_coords = np.arange(cols) - xoffset
        y_coords = np.arange(rows) - yoffset
        xx, yy = np.meshgrid(x_coords, y_coords)

        flat_x = xx.ravel()
        flat_y = yy.ravel()
        flat_ones = np.ones_like(flat_x)
        flat_intensity = scan_2D.ravel()
        nonzero_mask = flat_intensity != 0

        pixel_coords = np.vstack([
            flat_x[nonzero_mask],
            flat_y[nonzero_mask],
            flat_ones[nonzero_mask]
        ])
        intensities = flat_intensity[nonzero_mask]
        coords_3d = framePose @ UStoCam @ pixel_coords
        coords_3d = coords_3d.T

        z_axis_cam = np.array([0, 0, 1, 0])
        normal_world = (framePose @ z_axis_cam)[:3]
        normal_world /= np.linalg.norm(normal_world)

        for pt, intensity in zip(coords_3d, intensities):
            voxel_idx = np.floor((pt - min_coords) / voxel_size).astype(int)

            for dx in range(-kernel_radius, kernel_radius + 1):
                for dy in range(-kernel_radius, kernel_radius + 1):
                    for dz in range(-kernel_radius, kernel_radius + 1):
                        vi = voxel_idx + np.array([dx, dy, dz])
                        if np.any(vi < 0) or np.any(vi >= grid_shape):
                            continue
                        voxel_center = (vi + 0.5) * voxel_size + min_coords
                        diff = voxel_center - pt
                        dist = np.linalg.norm(diff)

                        # Compute weight
                        if method == 'solus':
                            thickness = sigma
                            normal_dist = np.abs(np.dot(diff, normal_world))
                            if normal_dist > thickness:
                                continue
                            weight = 1.0
                        elif method == 'inverse':
                            weight = 1.0 / (dist + epsilon)
                        elif method == 'gaussian':
                            weight = np.exp(-dist**2 / (2 * sigma**2))
                       
                        key = tuple(vi)
                        value_sums[key] += intensity * weight
                        weight_sums[key] += weight

    voxel_grid = np.zeros(grid_shape, dtype=np.uint8)
    for key in value_sums:
        avg = value_sums[key] / weight_sums[key]
        voxel_grid[key] = np.clip(int(round(avg)), 0, 255)

    elapsed = time.time() - start_time
    return voxel_grid, min_coords, elapsed



if __name__ == "__main__":
    LL1 = np.load("givens/LL1(279, 649, 850).npy")
    masked_LL1 = np.load("givens/masked_LL1(279, 649, 850).npy")
    allPoses = np.load('givens/poses_downsampled(3, 4, 279).npy')

    # Gaussian kernel
    voxel_grid, origin, elapsed_time = weighted_reconstruct(
        masked_LL1, allPoses, voxel_size=0.001,
        method='gaussian', kernel_radius=1, sigma=0.01)
    np.save("gaussian.npy", voxel_grid)
    print("Gaussian")
    print("Grid shape:", voxel_grid.shape)
    print("Elapsed time:", elapsed_time)
    print()

    # Solus method
    voxel_grid, origin, elapsed_time = weighted_reconstruct(
        masked_LL1, allPoses, voxel_size=0.001,
        method='solus', kernel_radius=1, sigma=0.01)
    np.save("solus.npy", voxel_grid)
    print("Solus")
    print("Grid shape:", voxel_grid.shape)
    print("Elapsed time:", elapsed_time)
    print()

    # Inverse method
    voxel_grid, origin, elapsed_time = weighted_reconstruct(
        masked_LL1, allPoses, voxel_size=0.001,
        method='inverse', kernel_radius=1, epsilon=1e-5)
    np.save("inverse.npy", voxel_grid)
    print("Inverse")
    print("Grid shape:", voxel_grid.shape)
    print("Elapsed time:", elapsed_time)
    print()
