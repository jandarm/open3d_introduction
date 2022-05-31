import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


print("Загружаем тестовый сегмент")
pcd = o3d.io.read_point_cloud("copy_of_fragment.pcd")
# Читаем массив точек, это интересно тем,
# Что в таком случае сформировать файл можно и в ручную
# Даже в текстовом редакторе
print(pcd)
print(np.asarray(pcd.points))
# Вообще все параметры после массива не обязательны
# - они для центровки и камеры
# Визуализация спокойно отрисует если передать только массив облак
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

# Доказательство что можно вручную формировать облако
# Переиспользую pcd только потому, что каждый раз пускаем новый рендер.
# Так сохраняем понимание, что работаем с одним и тем же типом
# - открываемым файлом pcd
pcd = o3d.io.read_point_cloud("test.pcd")
print(np.asarray(pcd.points))

# Опровергаю своё предыдущее утверждение:
# Хоть эти параметры и не обязательны, но очень рекомендуются
# Т.к. лучше центровать камеру на своём объекте и не терять его
o3d.visualization.draw_geometries([pcd],zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[9.37730000e-001, 3.37630000e-001, 0.00000000e+000],
                                  up=[-0.0694, -0.9768, 0.2024])


# Можно было бы использовать тетсовый файл, но я хочу убедиться, что
# вызывающий на этапе реконструкции пример работает на предыдущих этапах.
# Разница в ply и pcd выглядит незначительной, хотя ply может всё,
# что должен уметь pcd, но не наоборот
print("Децимация точек по вокселю 0.05")
pcd = o3d.io.read_point_cloud("raptor.ply")
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
o3d.visualization.draw_geometries([downpcd])

# Нажать N чтобы отобразить нормали
print("Пересчитываем нормали точек после децимации")
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([downpcd])

print("Получаем доступ к вектору нулевой точки")
print(downpcd.normals[0])
print("К первым 10")
print(np.asarray(downpcd.normals)[:10, :])


# Не до конца понимаю как работает обрезка
# Если это сохранение ТОЛЬКО точек попавших в полигон,
# То от динозавра должна была остаться хотя бы одна, но он исключается полностью
# Возможно он слишком далеко...
print("Обрезка по полигону")
pcd = o3d.io.read_point_cloud("fragment.pcd")
vol = o3d.visualization.read_selection_polygon_volume("crop.json")
cropped_wall = vol.crop_point_cloud(pcd)
o3d.visualization.draw_geometries([cropped_wall])


print("Покраска")
# pcd = o3d.io.read_point_cloud("raptor.ply")
downpcd.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([downpcd])


print("Вычисление расстояния")
dists = downpcd.compute_point_cloud_distance(downpcd)
dists = np.asarray(dists)
ind = np.where(dists > 0.01)[0]
downpcd_without_pcd = downpcd.select_by_index(ind)
o3d.visualization.draw_geometries([downpcd_without_pcd])


aabb = downpcd.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
obb = downpcd.get_oriented_bounding_box()
obb.color = (0, 1, 0)
o3d.visualization.draw_geometries([downpcd, aabb, obb])


print("Наименьшее множество всех точек")
hull, _ = downpcd.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((1, 0, 0))
o3d.visualization.draw_geometries([downpcd, hull_ls])


print("Расскраска кластеров, на удивление динозавр очень равномерный")
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(downpcd.cluster_dbscan(eps=0.065, min_points=10, print_progress=True))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
downpcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([downpcd])


print("Находим плоскость с большинством соседствующих точек")
downpcd = o3d.io.read_point_cloud("raptor.ply")
downpcd = downpcd.voxel_down_sample(voxel_size=0.05)
plane_model, inliers = downpcd.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = downpcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = downpcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


# Должно убирать пересекающиеся вершины относительно точки обзора
# Поэтому сверху должно выглядеть будто бы ничего не изменилось
diameter = np.linalg.norm(np.asarray(downpcd.get_max_bound()) - np.asarray(downpcd.get_min_bound()))
camera = [10, 10, diameter]
radius = diameter * 1000
_, pt_map = downpcd.hidden_point_removal(camera, radius)
downpcd = downpcd.select_by_index(pt_map)
o3d.visualization.draw_geometries([downpcd])