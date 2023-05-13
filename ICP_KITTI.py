# -*- coding: utf-8 -*-
"""
Created on Sat May  6 19:09:59 2023

@author: Johanna
"""
import open3d as o3d
import copy
import numpy as np
import os
from datetime import datetime
from SupportFunctions import *

#### Function for getting cropped pc (with rectangular bbox)
def crop_pc_bbox (pc, min_bd,max_bd):
    min_bound=min_bd
    max_bound=max_bd
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped_pc = o3d.geometry.PointCloud.crop(pc,bbox)
    
    return cropped_pc

def downsampling_pc (pc, voxel_size):
    pc_down = pc.voxel_down_sample(voxel_size)
    
    return pc_down
    


timestamp_src = 1
timestamp_trg = 1

name_txt = "TEST_05102023.txt"


path_to_data = r"C:\Users\Johanna\Downloads\2011_09_26\2011_09_26_drive_0001_sync\velodyne_points\data"
path_to_file = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline"

path_src = os.path.join(path_to_data, str(timestamp_src).zfill(10) + ".bin")
path_trg  = os.path.join(path_to_data, str(timestamp_trg).zfill(10) + ".bin")

path_txt = os.path.join(path_to_file, name_txt)


pc_1 = KITTI_Bin2PCD(path_src)
pc_2 = KITTI_Bin2PCD(path_trg)

source_pc = copy.deepcopy(pc_1)
target_pc = copy.deepcopy(pc_2)


##### DOWNSAMPLING

source_pc = downsampling_pc(source_pc,0.5)
target_pc = downsampling_pc(target_pc,0.5)


### CROPPING

target_pc = crop_pc_bbox(target_pc, (-50,-50,-50),(-10,50,50))

transform_matrix = np.asarray([[1.0, 0.0, 0.0, 5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
#source_pc = source_pc.transform(transform_matrix)

#source_pc.paint_uniform_color([1, 0.706, 0])
#target_pc.paint_uniform_color([0, 0.651, 0.929])
#o3d.visualization.draw_geometries([source_pc, target_pc],
 #                                 zoom=0.4459, front=[0.9288, -0.2951, -0.2242],
#                                  lookat=[1.6784, 2.0612, 1.4451],
#                                  up=[-0.3402, -0.9189, -0.1996])



with open(path_txt, 'w') as f:
    f.write('Evaluation of ICP algorithm for pc scan matching on KITTI dataset "2011_09_26_drive_0001_sync"\n ' + str(datetime.now()) + "\n\n")
    f.write('Source point cloud:\nTimestamp = %s\nNumber of points = %s\nCropped (Y/N)? = N\n\n' 
            %((str(timestamp_src).zfill(10)), str(len(source_pc.points))))
    f.write('Target point cloud:\nTimestamp = %s\nNumber of points = %s\nCropped (Y/N)? = Y\n\n'
            %((str(timestamp_trg).zfill(10)), str(len(target_pc.points))))


#### Scan matching with rectangualr bbox cropping
transform = icp(source_pc, target_pc, path_txt, transform_matrix)

print(transform)

with open(path_txt, 'a') as f:
    f.write('Transformation matrix:\n%s ' %str(transform) + "\n\n")    
    



##### Polygon volume

corners = np.array([[ 10, 50,  0.30217625],
 [ 30, 40,  0.29917539],
 [ 50, 30,  0.30329364],
 [ 70, 20,  0.3062945 ],
 [ 90, 0,  1.03551451],
 [ 70, -20,  1.03251366],
 [ 50, -30,  1.03663191],
 [ 30, -40,  1.03963277],
 [ 10, -50,  1.03963277]])

#corners = np.array(...)

# Convert the corners array to have type float64
bounding_polygon = corners.astype("float64")

# Create a SelectionPolygonVolume
vol = o3d.visualization.SelectionPolygonVolume()

# You need to specify what axis to orient the polygon to.
# I choose the "Y" axis. I made the max value the maximum Y of
# the polygon vertices and the min value the minimum Y of the
# polygon vertices.
vol.orthogonal_axis = "Z"
vol.axis_max = 10
vol.axis_min = -10

# Set all the Y values to 0 (they aren't needed since we specified what they
# should be using just vol.axis_max and vol.axis_min).
bounding_polygon[:, 2] = 0

# Convert the np.array to a Vector3dVector
vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)

# Crop the point cloud using the Vector3dVector
cropped_pcd = vol.crop_point_cloud(source_pc)

# Get a nice looking bounding box to display around the newly cropped point cloud
# (This part is optional and just for display purposes)
bounding_box = cropped_pcd.get_axis_aligned_bounding_box()
bounding_box.color = (1, 0, 0)

# Draw the newly cropped PCD and bounding box
o3d.visualization.draw_geometries([cropped_pcd, bounding_box],
                                  zoom=2,
                                  front=[5, -2, 0.5],
                                  lookat=[7.67473496, -3.24231903,  0.3062945],
                                  up=[1.0, 0.0, 0.0])