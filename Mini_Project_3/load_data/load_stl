import numpy as np
import trimesh

def stl_to_voxel_array(stl_path, voxel_resolution=100):
    # Load the STL file as a mesh
    mesh = trimesh.load_mesh(stl_path)

    # Normalize the mesh to fit inside a unit cube (optional but common)
    mesh.apply_translation(-mesh.bounds[0])
    mesh.apply_scale(1.0 / mesh.extents.max())

    # Create a voxelized version of the mesh
    voxelized = mesh.voxelized(pitch=1.0 / voxel_resolution)

    # Convert to a dense boolean array: True=inside, False=outside
    voxel_matrix = voxelized.matrix

    # Convert boolean to uint8: 1=inside, 0=outside
    voxel_array = voxel_matrix.astype(np.uint8)

    return voxel_array

# Example usage

# voxel_array = stl_to_voxel_array("kidney.stl", voxel_resolution=300)
# print(voxel_array.shape)  # (100, 100, 100) or similar
# np.save('kidney', voxel_array)
