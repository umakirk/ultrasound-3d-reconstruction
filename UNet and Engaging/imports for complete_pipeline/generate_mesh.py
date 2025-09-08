import numpy as np
from skimage import measure
from scipy.ndimage import binary_closing, binary_erosion, binary_dilation, label, generate_binary_structure
from skimage.morphology import remove_small_objects
import pyvista as pv


def mesh_from_voxel_grid(voxel_grid, voxel_size = 0.001):
    # mild cleanup
    binary_volume = voxel_grid > 0
    binary_volume = remove_small_objects(binary_volume, min_size=2000)
    binary_volume = binary_closing(binary_volume, structure=np.ones((3, 3, 3)))
    binary_volume = binary_erosion(binary_volume, iterations=1)
    binary_volume = binary_dilation(binary_volume, iterations=1)


    # Keep only largest structure
    structure3d = generate_binary_structure(3, 1)
    labeled, _ = label(binary_volume, structure=structure3d)
    labels, counts = np.unique(labeled[labeled > 0], return_counts=True)
    largest_label = labels[np.argmax(counts)]
    binary_volume = (labeled == largest_label)

    # Run Marching Cubes
    verts, faces, normals, _ = measure.marching_cubes(binary_volume.astype(np.uint8), level=0.5)
    verts_world = verts * voxel_size  # Convert to physical units (meters)
    faces_pv = np.hstack([[3, *face] for face in faces])
    mesh = pv.PolyData(verts_world, faces_pv)

    # Clean surface
    mesh = mesh.connectivity(largest=True)
    mesh = mesh.fill_holes(hole_size=1e-3)
    mesh = mesh.clean()
    mesh = mesh.smooth(n_iter=20, relaxation_factor=0.05)

    # Recenter so min corner is at (0, 0, 0) in meters
    bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax) in meters
    min_bounds = np.array([bounds[0], bounds[2], bounds[4]])
    mesh.translate(-min_bounds, inplace=True)

    # Conver to scale of millimeters
    mesh.scale([1000, 1000, 1000], inplace=True)

    return mesh


if __name__ == "__main__":

    pass_num = "4&5"

    # Load volume

    voxel_grid = np.load("voxel_grid_reconstructions/pass5_motionblur60_130_reconstruction.npy")
    mesh = mesh_from_voxel_grid(voxel_grid, voxel_size = 0.001)
    
    # Compute volume
    volume_m3 = mesh.volume
    volume_ml = volume_m3 * 0.001  # mm³ to mL
    print(f"Estimated kidney volume: {volume_ml:.2f} mL")

    # Save .ply file    
    mesh.save(f"surfaces/pass5_motionblur60_130_surface.ply")

    # Visualize
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue', show_edges=False)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()
