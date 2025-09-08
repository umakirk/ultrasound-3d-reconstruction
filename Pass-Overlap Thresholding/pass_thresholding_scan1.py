
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import vtk
from generate_mesh import mesh_from_voxel_grid
from scipy.spatial import cKDTree

# Load PNN recon results
voxel_grid = np.load("VOXELGRID_fromsegmentations.npy")
voxel_pass_count = np.load("VOXELPASSCOUNT_fromsegmentations.npy")

# Load ground truth mesh
gt_mesh_path = "surfaces/kidney_shape_model.ply"
gt_mesh = pv.read(gt_mesh_path).clean()
gt_centroid = np.array(gt_mesh.center)
gt_mesh.translate(-gt_centroid, inplace=True)
gt_mesh.rotate_y(180, inplace=True)
gt_points = gt_mesh.points

# Thresholds to display
thresholds = range(1, 9)

# ICP helper function
def run_icp(source_mesh, target_mesh, max_iterations=200):
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source_mesh)
    icp.SetTarget(target_mesh)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(max_iterations)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(source_mesh)
    transform_filter.SetTransform(icp)
    transform_filter.Update()
    return pv.wrap(transform_filter.GetOutput())

# Distance metric functions
def chamfer_distance(source_points, target_points):
    tree_target = cKDTree(target_points)
    tree_source = cKDTree(source_points)
    dist_src_to_tgt, _ = tree_target.query(source_points)
    dist_tgt_to_src, _ = tree_source.query(target_points)
    return (np.mean(dist_src_to_tgt) + np.mean(dist_tgt_to_src)) / 2

def hausdorff_distance(source_points, target_points):
    tree_target = cKDTree(target_points)
    tree_source = cKDTree(source_points)
    dist_src_to_tgt, _ = tree_target.query(source_points)
    dist_tgt_to_src, _ = tree_source.query(target_points)
    return max(np.max(dist_src_to_tgt), np.max(dist_tgt_to_src))

# Prepare PyVista plotter
plotter = pv.Plotter(shape=(2, 4), window_size=(2400, 1200))
shared_camera = pv.Camera()

# Store volumes and distances
volumes = []
chamfer_distances = []
hausdorff_distances = []
aligned_meshes = []

# Generate thresholded, ICP-aligned meshes
for t in thresholds:
    # Filter voxels below threshold
    common_voxel_grid = np.where(voxel_pass_count >= t, voxel_grid, 0)

    # Generate mesh
    mesh = mesh_from_voxel_grid(common_voxel_grid, voxel_size=0.001)
    mesh.translate(-np.array(mesh.center), inplace=True)
    mesh.rotate_y(180, inplace=True)

    # ICP alignment
    mesh_icp = run_icp(mesh, gt_mesh)
    aligned_meshes.append(mesh_icp)

    # Record metrics
    volumes.append(mesh_icp.volume * 0.001)  # volume in mL
    chamfer_distances.append(chamfer_distance(mesh_icp.points, gt_points))
    hausdorff_distances.append(hausdorff_distance(mesh_icp.points, gt_points))

    print(f"Threshold {t}: Volume = {volumes[-1]:.2f} mL, Chamfer = {chamfer_distances[-1]:.2f} mm, Hausdorff = {hausdorff_distances[-1]:.2f} mm")

# Plot Volume vs Threshold
gt_volume_ml = gt_mesh.volume * 0.001
plt.figure(figsize=(6,4))
plt.plot(thresholds, volumes, marker='o', linestyle='-')
plt.axhline(y=gt_volume_ml, color='red', linestyle='--', label=f"GT volume: {gt_volume_ml:.2f} mL")
plt.xlabel("Pass-overlap threshold")
plt.ylabel("Estimated volume (mL)")
plt.title("Volume vs. Pass-overlap Threshold")
plt.grid(True)
plt.legend()
plt.show()

# Plot Chamfer Distance vs Threshold
plt.figure(figsize=(6,4))
plt.plot(thresholds, chamfer_distances, marker='o', linestyle='-')
plt.xlabel("Pass-overlap threshold")
plt.ylabel("Chamfer Distance (mm)")
plt.title("Chamfer Distance vs. Threshold")
plt.grid(True)
plt.show()

# Plot Hausdorff Distance vs Threshold
plt.figure(figsize=(6,4))
plt.plot(thresholds, hausdorff_distances, marker='o', linestyle='-')
plt.xlabel("Pass-overlap threshold")
plt.ylabel("Hausdorff Distance (mm)")
plt.title("Hausdorff Distance vs. Threshold")
plt.grid(True)
plt.show()

# 2x4 Grid of thresholded ICP-aligned meshes
for idx, mesh_icp in enumerate(aligned_meshes):
    row = idx // 4
    col = idx % 4
    plotter.subplot(row, col)

    # Semi-transparent GT mesh
    plotter.add_mesh(gt_mesh, color="lightgray", opacity=0.3)

    # Thresholded ICP-aligned reconstruction
    plotter.add_mesh(mesh_icp, color="orange", opacity=0.5)

    plotter.add_text(f"Threshold {thresholds[idx]}", font_size=12)
    plotter.show_bounds(grid='front', location='outer', all_edges=True)
    plotter.add_axes()
    plotter.camera = shared_camera

plotter.link_views()
plotter.show()
