import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

mesh = o3d.io.read_triangle_mesh("Goat skull.ply")
# o3d.visualization.draw_geometries([mesh])
mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:len(mesh.triangles) // 3, :])
mesh.triangle_normals = o3d.utility.Vector3dVector(np.asarray(mesh.triangle_normals)[:len(mesh.triangle_normals) // 3, :])
mesh.paint_uniform_color([1, 0.706, 0])
# o3d.visualization.draw_geometries([mesh])


mesh_copy = o3d.io.read_triangle_mesh("Goat skull.ply")
# vertices = np.asarray(mesh_copy.vertices)
# noise = 15
# vertices += np.random.uniform(0, noise, size=vertices.shape)
# mesh_copy.vertices = o3d.utility.Vector3dVector(vertices)
# mesh_copy.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh_copy])

# mesh_out = mesh_copy.filter_smooth_simple(number_of_iterations=20)
# mesh_out.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh_out])

# mesh_out = mesh_copy.filter_smooth_laplacian(number_of_iterations=150)
# mesh_out.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh_out])

mesh_copy.compute_vertex_normals()
pcd = mesh_copy.sample_points_uniformly(number_of_points=750)

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh1, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

densities = np.asarray(densities)

density_colors = plt.get_cmap('plasma')((densities - densities.min()) / (densities.max() - densities.min()))
density_colors = density_colors[:, :3]
density_mesh = o3d.geometry.TriangleMesh()
density_mesh.vertices = mesh1.vertices
density_mesh.triangles = mesh1.triangles
density_mesh.triangle_normals = mesh1.triangle_normals
density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
o3d.visualization.draw_geometries([density_mesh])

vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh1.remove_vertices_by_mask(vertices_to_remove)

o3d.visualization.draw_geometries([mesh1])