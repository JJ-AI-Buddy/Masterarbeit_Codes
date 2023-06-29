# -*- coding: utf-8 -*-
"""
Created on Wed May  3 21:17:04 2023

@author: Johanna
"""

import open3d as o3d
import copy
import numpy as np


test_icp_pc = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud(test_icp_pc.paths[0])
target = o3d.io.read_point_cloud(test_icp_pc.paths[1])


source_temp = copy.deepcopy(source)
target_temp = copy.deepcopy(target)
source_temp.paint_uniform_color([1,0.706,1])
o3d.visualization.draw_geometries([source_temp], zoom=0.4459, front=[0.9288, -0.2951, -0.2242], lookat=[1.6784, 2.0612, 1.4451], up=[-0.3402, -0.9189, -0.1996])


source_points = np.asarray(source.points)
print(source_points)

source_down = source.voxel_down_sample(0.05)
o3d.visualization.draw_geometries([source_down], zoom=0.4459, front=[0.9288, -0.2951, -0.2242], lookat=[1.6784, 2.0612, 1.4451], up=[-0.3402, -0.9189, -0.1996])


source_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
source_down.paint_uniform_color([1,0.706,1])
o3d.visualization.draw_geometries([source_down], zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)

demo_crop_data = o3d.data.DemoCropPointCloud()
pcd = o3d.io.read_point_cloud(demo_crop_data.point_cloud_path)
vol = o3d.visualization.read_selection_polygon_volume(demo_crop_data.cropped_json_path)
chair = vol.crop_point_cloud(pcd)
o3d.visualization.draw_geometries([chair],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])