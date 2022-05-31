# Самый первый взгляд, песочница
import open3d as o3d
import numpy as np

# print("Testing IO for point cloud ...")
# sample_pcd_data = o3d.data.PCDPointCloud()
# print(1)
# pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
# print(2)
# print(pcd)
# print(3)
# o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)


pcd = o3d.io.read_point_cloud("raptor.ply")
# pcd = o3d.io.read_triangle_mesh("raptor.ply")

# pcd = mesh.sample_points_poisson_disk(750)
# o3d.visualization.draw_geometries([pcd])
# alpha = 0.03
# print(f"alpha={alpha:.3f}")
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# o3d.visualization.draw_geometries([pcd])


# print(pcd)
# print(np.asarray(pcd.points))

# Downsample the point cloud with a voxel of 0.05
downpcd = pcd.voxel_down_sample(voxel_size=0.07)

# Окраска в один цвет
# downpcd.paint_uniform_color([1, 0.706, 0])

#  Recompute the normal of the downsampled point cloud
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=10))

aabb = downpcd.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
obb = downpcd.get_oriented_bounding_box()
obb.color = (0, 1, 0)

# hull, _ = downpcd.compute_convex_hull()
# hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
# hull_ls.paint_uniform_color((1, 0, 0))
diameter = np.linalg.norm(np.asarray(downpcd.get_max_bound()) - np.asarray(downpcd.get_min_bound()))

camera = [0, 0, diameter]
radius = diameter * 100

_, pt_map = downpcd.hidden_point_removal(camera, radius)

downpcd = downpcd.select_by_index(pt_map)

o3d.visualization.draw_geometries([downpcd, aabb, obb])