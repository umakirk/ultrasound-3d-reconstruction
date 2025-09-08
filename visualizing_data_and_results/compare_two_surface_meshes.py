
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import vtk


# File paths

#Scale of millimeters
recon_mesh_path = r"surfaces\poster_artifact_added_pass5_common_surface.ply"
gt_mesh_path = "surfaces/kidney_shape_model.ply"


# recon_mesh_path = "surfaces/pass5_motionblur50_100_surface.ply"
# gt_mesh_path = "surfaces/pass5_groundtruth_surface.ply"

recon_name = recon_mesh_path[9:-4]
gt_name = gt_mesh_path[9:-4]


# Load and preprocess meshes
recon_mesh = pv.read(recon_mesh_path).clean()
gt_mesh = pv.read(gt_mesh_path).clean()

# Align centroids
recon_centroid = np.array(recon_mesh.center)
gt_centroid = np.array(gt_mesh.center)
recon_mesh.translate(-recon_centroid, inplace=True)
gt_mesh.translate(-gt_centroid, inplace=True)

# Rotate meshes 180 degrees about y-axis in place
recon_mesh.rotate_y(180, inplace=True)
gt_mesh.rotate_y(180, inplace=True)


# Perform ICP
def run_icp(source_mesh, target_mesh, max_iterations=200):
    # Get the underlying vtkPolyData from PyVista PolyData
    source_vtk = source_mesh  # already vtkPolyData
    target_vtk = target_mesh  # already vtkPolyData

    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source_vtk)
    icp.SetTarget(target_vtk)
    icp.GetLandmarkTransform().SetModeToRigidBody()  # Rigid alignment
    icp.SetMaximumNumberOfIterations(max_iterations)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    # Apply transformation
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(source_vtk)
    transform_filter.SetTransform(icp)
    transform_filter.Update()

    return pv.wrap(transform_filter.GetOutput()), icp


# Align reconstruction mesh to GT
recon_mesh_icp, icp_transform = run_icp(recon_mesh, gt_mesh)

# Distance calculations
def surface_distance_metrics(source_points, target_points):
    tree = cKDTree(target_points)
    dists, _ = tree.query(source_points)
    return dists

d_recon_to_gt = surface_distance_metrics(recon_mesh_icp.points, gt_mesh.points)
d_gt_to_recon = surface_distance_metrics(gt_mesh.points, recon_mesh_icp.points)

# Metrics
recon_volume = recon_mesh_icp.volume  * 0.001  # 1 mL = 1000 mm³
gt_volume = gt_mesh.volume * 0.001  # 1 mL = 1000 mm³

print(f"Ground truth volume: {gt_volume:.2f} mL")
print(f"Reconstruction volume: {recon_volume:.2f} mL")

chamfer_dist_mm = (np.mean(d_recon_to_gt) + np.mean(d_gt_to_recon)) / 2
hausdorff_dist_mm = max(np.max(d_recon_to_gt), np.max(d_gt_to_recon))

print(f"Chamfer distance (ICP-aligned):  {chamfer_dist_mm:.2f} mm")
print(f"Hausdorff distance (ICP-aligned): {hausdorff_dist_mm:.2f} mm")

# Add distances to mesh
recon_mesh_icp["DistToGT"] = d_recon_to_gt
# ---- Setup Plotter with 4 views ----
plotter = pv.Plotter(shape=(2, 2), border=True)

# Shared camera
shared_camera = pv.Camera()

# Function to prepare each subplot
def setup_subplot(row, col, title):
    plotter.subplot(row, col)
    plotter.add_text(title, font_size=10)
    plotter.show_bounds(grid='front', location='outer', all_edges=True)
    plotter.add_axes()
    plotter.camera = shared_camera


# View 1: GT Mesh
setup_subplot(0, 0, gt_name)
plotter.add_mesh(gt_mesh, color="lightblue", opacity=1.0)

# View 2: Reconstruction
setup_subplot(0, 1, recon_name)
plotter.add_mesh(recon_mesh_icp, color="lightcoral", opacity=1.0)

# View 3: Overlay (Transparent, Centered)
setup_subplot(1, 0, "Overlaid (ICP)")
plotter.add_mesh(gt_mesh, color="blue", opacity=0.3)
plotter.add_mesh(recon_mesh_icp, color="red", opacity=0.3)
plotter.add_mesh(pv.Sphere(radius=1, center=[0, 0, 0]), color="blue", name="GT ICP")

# View 4: Error Heatmap
setup_subplot(1, 1, "Error Heatmap (ICP)")
plotter.add_mesh(recon_mesh_icp, scalars="DistToGT", cmap="viridis", show_scalar_bar=True, 
                 scalar_bar_args={"title": f"Distance to {gt_name} (mm) \n", "title_font_size": 13, "label_font_size": 10, "height": 0.1})

# Add button to reset all views
def reset_camera():
    for i in range(4):
        plotter.subplot(i // 2, i % 2)
        plotter.reset_camera()

# Finalize view
plotter.link_views()
reset_camera()  # auto-reset on start
plotter.show()
