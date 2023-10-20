# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 21:17:23 2023

@author: Johanna
"""

import open3d as o3d
import copy
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import math

path_map = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\Moriyama_Map.pcd"
path_pc = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\PointClouds_Moriyama_140\1427157790778626.pcd"
path_csv = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\14_Local_Pose.csv"


pc_1 = o3d.io.read_point_cloud(path_pc)
source_pc = copy.deepcopy(pc_1)

source_pc.points

bbox = o3d.geometry.AxisAlignedBoundingBox([-100,-100,0.5],[100,100,3])
source_pc = o3d.geometry.PointCloud.crop(source_pc,bbox)

o3d.visualization.draw_geometries([source_pc],
                                 zoom=0.8, front=[0, 0, 1],
                                 lookat=[0, 0, 0],
                                 up=[0, 1, 0],
                                 point_show_normal=False)


points = np.asarray(source_pc.points)

points_ring = np.zeros((len(points),3))
r_min = 10
r_max = 15

j = 0
for i in range(0,len(points)):
    r = math.sqrt(pow(points[i,0],2) + pow(points[i,1],2))
    
    if (r >= r_min) and (r <= r_max):
        points_ring [j] = points[i]
        j += 1

points_ring = points_ring[~np.all(points_ring == 0, axis=1)]

#source_pc = o3d.geometry.PointCloud()
source_pc.points = o3d.utility.Vector3dVector(points_ring)