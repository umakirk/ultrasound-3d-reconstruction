import numpy as np
from collections import defaultdict
import time

def kernel_reconstruct(array_2D_scans, allPoses, voxel_size=0.01,
                       kernel_type='gaussian', kernel_radius=1, sigma=0.01,
                       scan_min=0, scan_max=None, row_min=0, row_max=None,
                       col_min=0, col_max=None):
    """
    Inputs:
        - array_2D_scans: (N_scans, rows, cols) ultrasound slices
        - allPoses: (3, 4, N_scans) pose matrix for each scan
        - voxel_size: edge length of voxel (meters)
        - kernel_type: 'spherical', 'cubic', 'gaussian', 'ellipsoid', 'truncated_gaussian'
        - kernel_radius: size of kernel (voxels)
        - sigma: std dev for Gaussian kernels
        - *_min/_max: cropping bounds (inclusive)

    Outputs:
        - voxel_grid: 3D uint8 grid of intensities
        - origin: real-world XYZ coordinate of voxel (0,0,0)
        - elapsed_time: time to reconstruct (s)
    """
    start_time = time.time()

    if scan_max is None: scan_max = array_2D_scans.shape[0]
    if row_max is None: row_max = array_2D_scans.shape[1]
    if col_max is None: col_max = array_2D_scans.shape[2]

    data = array_2D_scans[scan_min:scan_max, row_min:row_max, col_min:col_max]
    allPoses = allPoses[:, :, scan_min:scan_max]

    xoffset = 425 - col_min
    yoffset = 90 - row_min

    UStoCam = np.array([
        [0,       0,     -0.0398],
        [-0.0002, 0,     -0.0175],
        [0,      -0.0002, -0.1065],
        [0,       0,      1.0000]])

    all_coords = []

    for scan_index in range(data.shape[0]):
        scan = data[scan_index]
        pose = allPoses[:, :, scan_index]

        rows, cols = scan.shape
        xx, yy = np.meshgrid(np.arange(cols) - xoffset, np.arange(rows) - yoffset)
        flat_x = xx.ravel()
        flat_y = yy.ravel()
        flat_intensity = scan.ravel()
        mask = flat_intensity != 0

        pixels = np.vstack([flat_x[mask], flat_y[mask], np.ones(np.sum(mask))])
        points = pose @ UStoCam @ pixels
        points = points.T
        intensities = flat_intensity[mask]
        all_coords.append((points, intensities))

    all_pts = np.vstack([pts for pts, _ in all_coords])
    min_coords = all_pts.min(axis=0)
    max_coords = all_pts.max(axis=0)
    grid_shape = np.ceil((max_coords - min_coords) / voxel_size).astype(int) + 1

    value_sums = defaultdict(float)
    weight_sums = defaultdict(float)

    for points, intensities in all_coords:
        for pt, intensity in zip(points, intensities):
            voxel_idx = np.floor((pt - min_coords) / voxel_size).astype(int)

            for dx in range(-kernel_radius, kernel_radius + 1):
                for dy in range(-kernel_radius, kernel_radius + 1):
                    for dz in range(-kernel_radius, kernel_radius + 1):
                        vi = voxel_idx + np.array([dx, dy, dz])
                        if np.any(vi < 0) or np.any(vi >= grid_shape):
                            continue

                        center = (vi + 0.5) * voxel_size + min_coords
                        diff = center - pt
                        dist = np.linalg.norm(diff)

                        if kernel_type == 'spherical':
                            if dist <= kernel_radius * voxel_size:
                                weight = 1.0
                            else:
                                continue
                        elif kernel_type == 'cubic':
                            weight = 1.0
                        elif kernel_type == 'gaussian':
                            weight = np.exp(-dist**2 / (2 * sigma**2))
                        elif kernel_type == 'truncated_gaussian':
                            if dist > kernel_radius * voxel_size:
                                continue
                            weight = np.exp(-dist**2 / (2 * sigma**2))
                        elif kernel_type == 'ellipsoid':
                            dxm = diff[0] / (kernel_radius * voxel_size)
                            dym = diff[1] / (kernel_radius * voxel_size)
                            dzm = diff[2] / (kernel_radius * voxel_size)
                            if dxm**2 + dym**2 + dzm**2 <= 1:
                                weight = 1.0
                            else:
                                continue
                        else:
                            raise ValueError(f"Unknown kernel type '{kernel_type}'")

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
    
    voxel_grid, origin, time_taken = kernel_reconstruct(
        masked_LL1, allPoses, voxel_size=0.001,
        kernel_type='truncated_gaussian', kernel_radius=1, sigma=0.01)

    print("Grid shape:", voxel_grid.shape)
    print("Time taken:", time_taken)
    np.save("trunc_gaussian_voxel.npy", voxel_grid)
