# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:34:08 2023

@author: Johanna
"""

import open3d as o3d 
import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path

save_scan = True
route_name = "Suburban_Scan_"
validation_set = True

if validation_set == False:
    str_pc_scans = input("Please provide the path to the folder of the original point clouds collected.\n")
    path_pc_scans = Path(str_pc_scans)

    if os.path.lexists(path_pc_scans) == True:
        print("The given path is valid!\n\n")
    else: print("Please retry. The path does not seem to exist.\n\n")
    

str_csv_select = input("Please input the path to the csv-file containing information of the selected points cloud of each route.\n")
path_csv_select = Path(str_csv_select)

if os.path.lexists(path_csv_select) == True:
    print("The given path is valid!\n\n")
else: print("Please retry. The path does not seem to exist.\n\n")

str_prepro_scans = input("Please also provide the path to the folder where the preprocessed point clouds should be saved.\n")
path_prepro_scans = Path(str_prepro_scans)

if os.path.lexists(path_csv_select) == True:
    print("The given path is valid!\n\n")
else: print("Please retry. The path does not seem to exist.\n\n")


df_selected = pd.read_csv(path_csv_select, delimiter = ';', header = 0, index_col = False)

if validation_set == False:
    timestamps = df_selected["%time"].values.tolist()
list_path_pc = df_selected["pc.timestamp.path"].values.tolist()

x = 0
for x in range(0,len(list_path_pc)):
    
    if validation_set == False:
        pc_file = os.path.basename(str(list_path_pc[x]))
        path_pcd = os.path.join(path_pc_scans,pc_file)
    else:
        path_pcd = list_path_pc[x]
    
    pcd = o3d.io.read_point_cloud(str(path_pcd))
    
    new_name = route_name + str(x).zfill(2) + '.pcd'
    
    print("Loaded point cloud contains %i points.\n\n" %len(pcd.points))
    
    #### Downsampling
    
    pcd = pcd.voxel_down_sample(voxel_size=0.1)
    
    
    #### Ground removal
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.5,
                             ransac_n=6,
                             num_iterations=1000)
    
    pcd = pcd.select_by_index(inliers,invert = True)
    
    
    print("Preprocessed point cloud contains now %i points.\n\n" %len(pcd.points))
    
    path_pcd_prepro = os.path.join(path_prepro_scans,new_name)
    
    check = o3d.io.write_point_cloud(str(path_pcd_prepro), pcd)
    
    if check == True:
        print("Preprocessed point cloud has been saved successfully!\n\n")
    else:
        print("Saving of new point cloud failed!\n\n")
    
#Visualization
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=1.34,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat= [0.0,0.0,0.0], 
#                                   up=[-0.0694, -0.9768, 0.2024])



