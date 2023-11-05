# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:02:49 2023

@author: Johanna Hable

Notes:
    - Always input the directories without quotation marks
    - Set 'save_map' to 'True' if you want to save the result of any mapping as a point cloud in pcd-file format
    - Set 'ground_removal' to 'True' if you want the ground of the scans to be removed for the maps
    - Set 'create_runs' to 'True' if you want to create the map for a specific run of a specific route and follow the input instructions; else set the variable to 'False'
    - Set 'create_final' to 'True' if you want to combine the maps of several runs in one combined map and follow the input instructions; else set the variable to 'False'
    - Set 'load_online_scans' to 'True' if you want to load online scans from a run not used to build the map; these scans will then be seen as the online scans for applying the point cloud registration (matching online scan and map)
    - The result of 'load_online_scans' will be a csv-file listing the selected scans with their corresponding path and the GT poses from odometry
    - Do not change the variable 'hz_diff'
    - The variable 'step' is adjustable; depending on how many scans you want to use to build your map
    - The variable 'step_online' defines in which intervall the online scans will be selected if 'load_online_scans' is 'True'
    - The variable 'static_offset' is used to adjust the origin of the map point cloud; it is the same for Route_1, Route_2 and Route_3 and different for Route_4 because the starting point of data collection was not at CAR
"""

import open3d as o3d 
import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path

save_map = False
create_runs =False
create_final = False
load_online_scans = True
ground_removal = False


num_runs = 2

hz_diff = 5 #Odometry messages are published at 15 Hz (LiDAR scans at 10 Hz)
step = 10 #Every xth timestamp is used to build the map
step_online = 1000
static_offset = (-328106, -4424795, -216) #(-326500,-4429495, -230) #for Route1, 2 and 3 #(-328106, -4424795, -216) #only for Route 4


######################################################
################### Novatel ODOM #####################
######################################################

if create_runs == True:
    
    str_path_gps = input("Please enter the path to the odometry or GPS pose file which should be used to arrange the LiDAR scans as one map ending in '.csv' and without ' "" '.\n")
    path_gps = Path(str_path_gps)
    
    if os.path.lexists(path_gps) == True:
        print("The given path is valid!\n\n")
    else: print("Please retry. The path does not seem to exist.\n\n")
    
    str_path_scans = input("Please also provide the directory of the LiDAR scans (.pcd-files) ending with a backslash and '*.pcd'.\n")
    # = str_path_scans + "*.pcd"
    path_scans = Path(str_path_scans)
    
    if os.path.lexists(os.path.basename(path_scans)) == True:
        print("The given path is valid!\n\n")
    else: print("Please retry. The path does not seem to exist.\n\n")
    
    # Reading the odometry data (GPS+RTK+IMU) from csv-file
    df_gps = pd.read_csv(path_gps, delimiter = ',', header = 0, engine = 'python', encoding = 'utf-8')#unicode_escape')
    #df_gps[['field.pose.pose.position.x', 'field.pose.pose.position.y', 'field.pose.pose.position.z']]
    #df_gps.dtypes
    
    #Saving the position + orientation for each timestamp in a list
    list_gps = []
    for ind in df_gps.index:
        transl_gps = np.array([df_gps['field.pose.pose.position.x'][ind],
                               df_gps['field.pose.pose.position.y'][ind],
                               df_gps['field.pose.pose.position.z'][ind],
                               df_gps['field.pose.pose.orientation.x'][ind],
                               df_gps['field.pose.pose.orientation.y'][ind],
                               df_gps['field.pose.pose.orientation.z'][ind],
                               df_gps['field.pose.pose.orientation.w'][ind]])
        #(transl_gps)
        list_gps.append(transl_gps)
    
   
    counter = 0
    k = hz_diff               #Odometry messages are published at 15 Hz (LiDAR scans at 10 Hz)
    point_cloud = np.zeros((1,3))
    for i in glob.glob(str_path_scans):  # * means inner directory
        if counter == 0 and k < len(list_gps):
            print(i)
            print(k)
            print(df_gps.loc[k, '%time'])
            #print(counter)
            pcd = o3d.io.read_point_cloud(i)
            #pcd = pcd.voxel_down_sample(voxel_size=0.1)
            
            T = np.eye(4) #Initialize transformation matrix
            T[:3, :3] = pcd.get_rotation_matrix_from_quaternion((list_gps[k][6],list_gps[k][3], list_gps[k][4], list_gps[k][5]))
            #T[:3, :3] = pcd.get_rotation_matrix_from_xyz((0,0,-60*(np.pi/180)))
            T[0, 3] = list_gps[k][0]
            T[1, 3] = list_gps[k][1]
            T[2,3] = list_gps[k][2]
            
            #R = pcd.get_rotation_matrix_from_quaternion((list_gps[k][3],list_gps[k][4], list_gps[k][5], list_gps[k][6]))
            #pcd.rotate(R, center=pcd.get_center())
            #pcd.translate((list_gps[k][0], list_gps[k][1], list_gps[k][2]), relative = False)
            
            pcd = pcd.transform(T)
            
            if ground_removal == True:
                
                plane_model, inliers = pcd.segment_plane(distance_threshold=0.5,
                                         ransac_n=6,
                                         num_iterations=1000)
                
                pcd = pcd.select_by_index(inliers,invert = True)
            
            a = np.asarray(pcd.points)
            #point_cloud.append(a)
            point_cloud = np.concatenate((point_cloud,a),axis = 0 )
        
        counter += 1
        if counter == step:
            counter = 0
        else: pass
        
        k += hz_diff
    
    point_cloud = np.delete(point_cloud,0,0)  #Delete the first row that only contains zeros  
    
    pcd_map = o3d.geometry.PointCloud()
    pcd_map.points = o3d.utility.Vector3dVector(point_cloud)
    
    print("The aggregation of the selected LiDAR scans contains %i points.\n" %len(pcd_map.points))
    
    pcd_map = pcd_map.remove_duplicated_points()
    
    print("The map contains %i points after removing duplicated points.\n" %len(pcd_map.points))
    
    pcd_map_down = pcd_map.voxel_down_sample(voxel_size=0.1)
    
    print("Created map contains %i points after downsampling.\n\n" %len(pcd_map_down.points))
    
    #pcd_map_down.paint_uniform_color([1.0,0.0,0.0])
    
    #mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #R = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi ))
    #pcd_map_down.rotate(R, center=(0, 0, 0))
    
    
    # Saving the map
    if save_map == True:
        
        str_path_map = input("To save the map you need to input its file path also defining the filename ending in '.pcd'.\n")
        #path_map = Path(str_path_map)
        
        check = o3d.io.write_point_cloud(str_path_map, pcd_map_down)
        
        if check == True:
            print("New map has been saved successfully!\n\n")
        else:
            print("Saving of new map failed!\n\n")

    #Visualization
    o3d.visualization.draw_geometries([pcd_map_down],
                                      zoom=1.34,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat= [-static_offset[0],-static_offset[1], -static_offset[2]], 
                                      up=[-0.0694, -0.9768, 0.2024])



#######################################
######## Creating the final map ######
######################################

if create_final ==True:
    
    list_pc_map = []
    print("You defined to use the maps of %i run(s) to build the final map.\n\n" %num_runs)
    
    for i in range(0,num_runs):
        
        str_map_run = input("Please provide the file directory of map %i to use for the final map.\n" %i)
        #path_map_run = Path(str_map_run)

        pc_scan = o3d.io.read_point_cloud(str_map_run)
        list_pc_map.append(pc_scan)

    print ("Final map gets created now.\n\n")
    
    #Aggregate the points to one big map
    map_point_cloud = np.zeros((1,3))
    for el in list_pc_map:
        arr = np.asarray(el.points)
        map_point_cloud = np.concatenate((map_point_cloud,arr),axis = 0 )
    
    
    map_point_cloud = np.delete(map_point_cloud,0,0) 
    
    pcd_map = o3d.geometry.PointCloud()
    pcd_map.points = o3d.utility.Vector3dVector(map_point_cloud)
    
    print("The aggregation of the selected maps contains %i points.\n" %len(pcd_map.points))
    
    pcd_map = pcd_map.remove_duplicated_points()
    
    print("The final map contains %i points after removing duplicated points.\n" %len(pcd_map.points))
    
    pcd_map = pcd_map.voxel_down_sample(voxel_size=0.1)
    
    print("Final map out of %i runs with a total of %i points after downsampling has been created successfully.\n" %(len(list_pc_map),len(pcd_map.points)))
    
    #Set the origin of the final map by relatively translating the map with fixed values
    #center = np.negative(pcd_map.get_center())
    pcd_map = pcd_map.translate(static_offset, relative = True) 
    
    if save_map == True:
        
        str_path_map_final = input("To save the final map you need to input its file path also defining the filename ending in '.pcd'.\n")
        #path_map_final = Path(str_path_map_final)
        
        check = o3d.io.write_point_cloud(str_path_map_final, pcd_map)
        
        if check == True:
            print("New map has been saved successfully!\n\n")
        else:
            print("Saving of new map failed!\n\n")
    
    #Visualize
    o3d.visualization.draw_geometries([pcd_map],
                                      zoom=1,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[0,0, 0],
                                      up=[-0.0694, -0.9768, 0.2024])
  
    
    
######################################################################
######### Load the online scans from runs not used for mapping #######
######################################################################

if load_online_scans == True:
    
    str_path_map_final = input("Please provide the file path of the final map to be loaded which is also defining the filename ending in '.pcd'.\n")
    
    final_map = o3d.io.read_point_cloud(str_path_map_final)
    
    str_path_gps = input("Please enter the path to the odometry or GPS pose file which defines the vehicle pose at every timestamp ending in '.csv' and without ' "" '.\n")
    path_gps = Path(str_path_gps)
    
    if os.path.lexists(path_gps) == True:
        print("The given path is valid!\n\n")
    else: print("Please retry. The path does not seem to exist.\n\n")
    
    str_path_scans = input("Please also provide the directory of the LiDAR scans (.pcd-files) ending with a backslash and '*.pcd'.\n")
    # = str_path_scans + "*.pcd"
    path_scans = Path(str_path_scans)
    
    if os.path.lexists(os.path.basename(path_scans)) == True:
        print("The given path is valid!\n\n")
    else: print("Please retry. The path does not seem to exist.\n\n")
    
    
    df_gps = pd.read_csv(path_gps, delimiter = ',', header = 0, engine = 'python', encoding = 'utf-8')#unicode_escape')
    #df_gps[['field.pose.pose.position.x', 'field.pose.pose.position.y', 'field.pose.pose.position.z']]
    #df_gps.dtypes
    
    list_gps = []
    for ind in df_gps.index:
        transl_gps = np.array([df_gps['field.pose.pose.position.x'][ind],
                               df_gps['field.pose.pose.position.y'][ind],
                               df_gps['field.pose.pose.position.z'][ind],
                               df_gps['field.pose.pose.orientation.x'][ind],
                               df_gps['field.pose.pose.orientation.y'][ind],
                               df_gps['field.pose.pose.orientation.z'][ind],
                               df_gps['field.pose.pose.orientation.w'][ind]])
        #(transl_gps)
        list_gps.append(transl_gps)
    
    
    #pc_test = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Run1PC_GPS\1692975298.153691530.pcd")
    #R = pc_test.get_rotation_matrix_from_quaternion((list_gps[5][6],list_gps[5][3], list_gps[5][4], list_gps[5][5]))
    
    
    online_scans = []
    list_timestamps = []
    list_df_gps_index = []
    counter = 0
    k = hz_diff
    point_cloud = np.zeros((1,3))
    for i in glob.glob(str_path_scans):  # * means inner directory
        if counter == 0 and k < len(list_gps):
            print(i)
            print(k)
            print(df_gps.loc[k, '%time'])
            #print(counter)
            pcd = o3d.io.read_point_cloud(i)
            #pcd = pcd.voxel_down_sample(voxel_size=0.1)
    
            
            T = np.eye(4)
            T[:3, :3] = pcd.get_rotation_matrix_from_quaternion((list_gps[k][6],list_gps[k][3], list_gps[k][4], list_gps[k][5]))
            #T[:3, :3] = pcd.get_rotation_matrix_from_xyz((0,0,-60*(np.pi/180)))
            T[0, 3] = list_gps[k][0]
            T[1, 3] = list_gps[k][1]
            T[2,3] = list_gps[k][2]
            
            #R = pcd.get_rotation_matrix_from_quaternion((list_gps[k][3],list_gps[k][4], list_gps[k][5], list_gps[k][6]))
            #pcd.rotate(R, center=pcd.get_center())
            #pcd.translate((list_gps[k][0], list_gps[k][1], list_gps[k][2]), relative = False)
            
            if ground_removal == True:
                
                plane_model, inliers = pcd.segment_plane(distance_threshold=0.5,
                                         ransac_n=6,
                                         num_iterations=1000)
                
                pcd = pcd.select_by_index(inliers,invert = True)
            
            
            pcd = pcd.transform(T)
            pcd = pcd.translate(static_offset, relative = True)
            
            online_scans.append(pcd)
            list_timestamps.append(i)
            list_df_gps_index.append(k)

        
        counter += 1
        if counter == step_online:
            counter = 0
        else: pass
        
        k += hz_diff
    
    print("%i Online LiDAR scans have been loaded" %len(online_scans))
    
    df_gps_GT = df_gps[['%time',
                        'field.pose.pose.position.x',
                        'field.pose.pose.position.y',
                        'field.pose.pose.position.z',
                        'field.pose.pose.orientation.x',
                        'field.pose.pose.orientation.y',
                        'field.pose.pose.orientation.z',
                        'field.pose.pose.orientation.w']]
    
    df_gps_GT [['field.pose.pose.position.x']] = df_gps_GT [['field.pose.pose.position.x']] + float(static_offset[0])
    df_gps_GT [['field.pose.pose.position.y']] = df_gps_GT [['field.pose.pose.position.y']] + float(static_offset[1])
    df_gps_GT [['field.pose.pose.position.z']] = df_gps_GT [['field.pose.pose.position.z']] + float(static_offset[2])
    
    
    df_gps_GT = df_gps_GT.iloc[list_df_gps_index]
    df_gps_GT.insert(0, "pc.timestamp.path", list_timestamps, True)
    
    # Save GT positions of the selected online scans in a csv file
    #str_path_gps_GT = input("Please provide a file directory also including the file name for saving the GT GPS poses of the selected online scans with step size %i and ending with '.csv' and without ' "" '.\n" %step_online)
    #path_gps_GT = Path(str_path_gps_GT)

    #df_gps_GT.to_csv(path_gps_GT,sep = ';', index = False, header = True)
    
    
    for el in online_scans:
        el.paint_uniform_color([0.5,0.0,0.5])
    
    online_scans.append(final_map)

    
    o3d.visualization.draw_geometries(online_scans,
                                      zoom=1,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[0,0, 0],
                                      up=[-0.0694, -0.9768, 0.2024])
    