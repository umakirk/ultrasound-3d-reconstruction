def check_alignment(voxel_indices, voxel_size, origin, num_nearest_points):
        voxel_indices = np.array(voxel_indices)
        voxel_size = np.array([voxel_size] * 3)

        origin = np.array(origin)

        voxel_xyz_center =  origin + (voxel_indices + 0.5) * voxel_size

        from scipy.spatial import cKDTree

        # spatial index for fast neighbor lookup
        tree = cKDTree(point_cloud[:, :3])  # Use only XYZ, ignore intensity

        # find 10 nearest neighbors to the voxel center
        _, indices = tree.query(voxel_xyz_center, k=num_nearest_points)

        nearest_points = point_cloud[indices]  # shape (num_nearest_points, 4), last col is intensity

        avg_intensity = np.mean(nearest_points[:, 3])

        voxel_intensity = voxel_grid[*voxel_indices]

        return voxel_xyz_center, num_nearest_points, avg_intensity, voxel_intensity

for voxel_indices in [(10, 10, 10), (15, 15, 15), (20, 20, 20), (30, 30, 30), (25, 30, 10)]:
        voxel_xyz_center, num_nearest_points, avg_intensity, voxel_intensity = check_alignment(voxel_indices, 0.001, origin, 40)
        print("Voxel indices:", voxel_indices)
        print("Voxel center in xyz world coordinates:", voxel_xyz_center)
        print(f"Average intensity of {num_nearest_points} nearest points:", avg_intensity)
        print("Voxel grid intensity:", voxel_intensity)
        print("\n")
