# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:09:37 2023

@author: Johanna
"""

import open3d as o3d 
import numpy as np
import pandas as pd
import glob
import os
import random as rd


## INPUTS

test_load = False
path_to_scans = "D:\\Johanna\\bags\\Route_1\\withoutCamera\\Run_3\\pcd_static\\*.pcd"
#"D:\\Johanna\\bags\TRC-Skidpad\\Map_with_GPS\\Run6PC_GPS\\*.pcd"
path_to_map = "D:\\Johanna\\bags\\Route_1\\withoutCamera\\Run_3\\map\\Route_1_Run_3_map3_groundRemoval.pcd"
#"D:\\Johanna\\bags\TRC-Skidpad\\Map_with_GPS\\Map_GPS_Run6.pcd"
path_to_final_map = "D:\\Johanna\\bags\Route_1\\FinalMap_Route1.pcd"
file_gps = "ODOM_Pose_Route1_Run3.csv"
#"ODOM_GPS_Run6.csv"
path_origin = r"D:\Johanna\bags\Route_1\withoutCamera\Run_3\odometry"
#r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Run6PC_GPS"
save_map = True
create_runs = False
create_final = True
load_online_scans = False
ground_removal = False

static_offset = (-326500,-4429495, -230) #(-283960,-4465142,-334)

path_to_data = r"D:\Johanna\bags\Route_1"
# r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Evaluations\01_TRC_Skidpad_Data"
file_GT_csv = "01_TRC-Skidpad_GT_Poses_Run6.csv"

maps = []         # empty list to collect the selected lidar scans

##########################################
###### Testing of LiDAR scan loading #####
#########################################

if test_load == True:

    counter = 0
    point_cloud = np.zeros((1,3))    #numpy array for aggregation of all the points of the selected lidar scans
    for i in glob.glob(path_to_scans):  # * means inner directory
        if counter == 0:
            #print(i)
            pcd = o3d.io.read_point_cloud(i) #load point cloud from pcd-file
            pcd = pcd.voxel_down_sample(voxel_size=0.1) #Downsampling
            a = np.array(pcd.points) #Transfer the points to numpy array
            #point_cloud.append(a)
            point_cloud = np.concatenate((point_cloud,a),axis = 0 ) #Add the new points to existing array
        
        counter += 1
        if counter == 10:
            counter = 0
        else: pass
        
    
    #len(point_cloud)
    
    pcd_map = o3d.geometry.PointCloud() #Initialize new point cloud object
    pcd_map.points = o3d.utility.Vector3dVector(point_cloud) #Set the points from numpy array
    
    print("The aggregation of the selected LiDAR scans contains %i points" %len(pcd_map.points))
    
    pcd_map = pcd_map.remove_duplicate_points()
    
    print("The map contains %i points after removing duplicaate points" %len(pcd_map.points))
    
    pcd_map_down = pcd_map.voxel_down_sample(voxel_size=0.1) #Downsample the new aggregated point cloud
    
    
    print("The final map after downsampling contains %i points" %len(pcd_map_down.points))
    
    if save_map == True:
        check = o3d.io.write_point_cloud(path_to_map, pcd_map_down)
        
        if check == True:
            print("New map has been saved successfully")
        else:
            print("Saving of new map failed")
        
    
    maps.append(pcd_map_down)
    
    print("%i maps have been created" %len(maps))
    
    # Paint each of the created maps with a uniform random color for visualization
    for el in maps:
        el.paint_uniform_color([rd.random(),rd.random(),rd.random()])

    
    #Visualization
    o3d.visualization.draw_geometries(maps,
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])


#pcd = o3d.io.read_point_cloud("D:\\Johanna\\bags\TRC-Skidpad\\Run1TRCSkidpad\\1692975390.033923149.pcd")

#pcd = pcd.voxel_down_sample(voxel_size=0.1)

#o3d.visualization.draw_geometries([pcd],
#                                  zoom=0.3412,
#                                  front=[0.4257, -0.2125, -0.8795],
#                                  lookat=[2.6172, 2.0475, 1.532],
#                                  up=[-0.0694, -0.9768, 0.2024])

#len(pcd.points)

###############################################
############### Novatel BESTPOS ###############
##############################################

# Using only the BESTPOS messages is not enough for aggregating the point clouds to a consistent map

# maps = []

# file_gps = "GNSSRTK_Pose_Run1.csv"
# path_origin = r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Run1PC_GPS"

# path_gps = os.path.join(path_origin,file_gps)

# df_gps = pd.read_csv(path_gps, delimiter = ',', header = 0, engine = 'python', encoding = 'utf-8')#unicode_escape')
# df_gps[['field.lat', 'field.lon', 'field.hgt']]

# df_gps[['field.lon']].max()
# df_gps[['field.lon']].min()

# df_gps.dtypes

# list_gps = []
# for ind in df_gps.index:
#     transl_gps = np.array([df_gps['field.lat'][ind], df_gps['field.lon'][ind], df_gps['field.hgt'][ind]])
#     print(transl_gps)
#     #(transl_gps)
#     list_gps.append(transl_gps)


# counter = 0
# k = 0
# point_cloud = np.zeros((1,3))
# for i in glob.glob("D:\\Johanna\\bags\TRC-Skidpad\\Map_with_GPS\\Run1PC_GPS\\*.pcd"):  # * means inner directory
#     if counter == 0 and k < len(list_gps):
#         print(i)
#         print(k)
#         print(counter)
#         pcd = o3d.io.read_point_cloud(i)
#         #pcd = pcd.voxel_down_sample(voxel_size=0.1)
#         pcd.translate((list_gps[k][0]*10, list_gps[k][1]*10, 0), relative = False)
#         a = np.asarray(pcd.points)
#         #point_cloud.append(a)
#         point_cloud = np.concatenate((point_cloud,a),axis = 0 )
    
#     counter += 1
#     if counter == 10:
#         counter = 0
#     else: pass
    
#     k += 1 

# point_cloud = np.delete(point_cloud,0,0)    
    
# pcd_map = o3d.geometry.PointCloud()
# pcd_map.points = o3d.utility.Vector3dVector(point_cloud)

# pcd_map_down = pcd_map.voxel_down_sample(voxel_size=0.1)

# len(pcd_map_down.points)

# pcd_map_down.paint_uniform_color([1.0,0.0,0.0])

# #mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
# #R = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi ))
# #pcd_map_down.rotate(R, center=(0, 0, 0))

# maps.append(pcd_map_down)

# #################################################################################

# pcd_lidar = o3d.io.read_point_cloud("D:\\Johanna\\bags\TRC-Skidpad\\Run1Map.pcd")
# pcd_lidar.paint_uniform_color([0.0,0.0,1.0])


######################################################
################### Novatel ODOM #####################
######################################################

if create_runs == True:
    
    maps = []
    
    # Reading the odometry data (GPS+RTK+IMU) from csv-file
    path_gps = os.path.join(path_origin,file_gps)
    
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
    
    
    #pc_test = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Run1PC_GPS\1692975298.153691530.pcd")
    #R = pc_test.get_rotation_matrix_from_quaternion((list_gps[5][6],list_gps[5][3], list_gps[5][4], list_gps[5][5]))
    
    
    counter = 0
    k = 5               #Odometry messages are published at 15 Hz (LiDAR scans at 10 Hz)
    point_cloud = np.zeros((1,3))
    for i in glob.glob(path_to_scans):  # * means inner directory
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
        if counter == 10:
            counter = 0
        else: pass
        
        k += 5
    
    point_cloud = np.delete(point_cloud,0,0)  #Delete the first row that only contains zeros  
    
    pcd_map = o3d.geometry.PointCloud()
    pcd_map.points = o3d.utility.Vector3dVector(point_cloud)
    
    print("The aggregation of the selected LiDAR scans contains %i points" %len(pcd_map.points))
    
    pcd_map = pcd_map.remove_duplicated_points()
    
    print("The map contains %i points after removing duplicated points" %len(pcd_map.points))
    
    pcd_map_down = pcd_map.voxel_down_sample(voxel_size=0.1)
    
    print("Created map contains %i points after downsampling" %len(pcd_map_down.points))
    
    #pcd_map_down.paint_uniform_color([1.0,0.0,0.0])
    
    #mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #R = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi ))
    #pcd_map_down.rotate(R, center=(0, 0, 0))
    
    maps.append(pcd_map_down)
    
    
    ########################
    ##### TEST #############
    
    # pc_1 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Run1PC_GPS\1692975298.153691530.pcd")
    # pc_2 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Run1PC_GPS\1692975390.033923149.pcd")
    
    # mesh_1 = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # mesh_2 = o3d.geometry.TriangleMesh.create_coordinate_frame()
    
    # #pc_1.translate((-1.5,0,2))
    # #pc_2.translate((-1.5,0,2))
    
    # pc_1.translate((283737.7408 ,4465501.1162,0), relative = False)   #336.6636+2.0
    # pc_2.translate((284287.9961 -1.5,4464550.3370,2), relative = False)   #333.2362+2.0
    
    # mesh_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 5, origin = (283737.7408 ,4465501.1162,0))
    
    # mesh_1.translate((283737.7408,4465501.1162 -10,0), relative = False)
    # mesh_2.translate((284287.9961,4464550.3370,0), relative = False) 
    
    # R1 = pc_1.get_rotation_matrix_from_quaternion((-0.0070716048161214105,-0.009820895726918398,-0.49714285313124085,0.8675843048332392))
    # pc_1.rotate(R1, center=pc_1.get_center())
    
    # R2 = pc_2.get_rotation_matrix_from_quaternion((-0.0019136898290919519,1.1853186742495961e-05,-0.4974765819227137,0.8674752953768963))
    # pc_2.rotate(R2, center=pc_2.get_center())
    
    
    # mesh_1.rotate(R1, center = mesh_1.get_center())
    # mesh_2.rotate(R2, center = mesh_2.get_center())
    # mesh_1.scale(5, center=mesh_1.get_center())
    # mesh_2.scale(5, center=mesh_2.get_center())
    
    
    # pc_1.paint_uniform_color([1.0,0.0,0.0])
    # pc_2.paint_uniform_color([0.0,1.0,0.0])
    
    #######################################
    #######################################
    
    print("%i maps have been created" %len(maps))
    
    
    if save_map == True:
        check = o3d.io.write_point_cloud(path_to_map, pcd_map_down)
        
        if check == True:
            print("New map has been saved successfully")
        else:
            print("Saving of new map failed")
            
    
    # Paint each of the created maps with a uniform random color for visualization
    #for el in maps:
    #    el.paint_uniform_color([rd.random(),rd.random(),rd.random()])
    
    #Visualization
    o3d.visualization.draw_geometries(maps,
                                      zoom=1.34,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat= [326500,4429495, 230],  # [283737,4465501, 336],
                                      up=[-0.0694, -0.9768, 0.2024])
    

#######################################
######## Creating the final map ######
######################################

if create_final ==True:

    pc_1 = o3d.io.read_point_cloud(r"D:\Johanna\bags\Route_1\withoutCamera\Run_2\map\Route_1_Run_2_map2_groundRemoval.pcd")
    pc_2 = o3d.io.read_point_cloud(r"D:\Johanna\bags\Route_1\withoutCamera\Run_3\map\Route_1_Run_3_map3_groundRemoval.pcd")
    pc_3 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Map_GPS_Run3.pcd")
    pc_4 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Map_GPS_Run4.pcd")
    pc_5 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Map_GPS_Run5.pcd")
    pc_6 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Map_GPS_Run6.pcd")
    
    #Relative alignment (if needed) - values from manual alignment with Blender
    #pc_1 = pc_1.translate((-66,113.5,0), relative = True)
    #pc_2 none
    #pc_3 none
    #pc_4 = pc_4.translate((38,-67,0), relative = True)
    #pc_5 = pc_5.translate((3,-3,0), relative = True)
    #pc_6 none
    
    
    #Which runs should be used to create the final map?
    list_pc_map = [pc_1,pc_2]
    
    #Aggregate the points to one big map
    map_point_cloud = np.zeros((1,3))
    for el in list_pc_map:
        arr = np.asarray(el.points)
        map_point_cloud = np.concatenate((map_point_cloud,arr),axis = 0 )
    
    
    map_point_cloud = np.delete(map_point_cloud,0,0) 
    
    pcd_map = o3d.geometry.PointCloud()
    pcd_map.points = o3d.utility.Vector3dVector(map_point_cloud)
    
    print("The aggregation of the selected maps contains %i points" %len(pcd_map.points))
    
    pcd_map = pcd_map.remove_duplicated_points()
    
    print("The map contains %i points after removing duplicated points" %len(pcd_map.points))
    
    pcd_map = pcd_map.voxel_down_sample(voxel_size=0.1)
    
    print("Final map out of %i runs with a total of %i points after downsampling has been created" %(len(list_pc_map),len(pcd_map.points)))
    
    #Set the origin of the final map by relatively translating the map with fixed values
    #center = np.negative(pcd_map.get_center())
    pcd_map = pcd_map.translate(static_offset, relative = True) 
    
    if save_map == True:
        check = o3d.io.write_point_cloud(path_to_final_map, pcd_map)
        
        if check == True:
            print("New map has been saved successfully")
        else:
            print("Saving of new map failed")
    
    #Visualize
    o3d.visualization.draw_geometries([pcd_map],
                                      zoom=1,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[0,0, 0],
                                      up=[-0.0694, -0.9768, 0.2024])
    
    
    ############################################
    ############# Testing ######################
    ############################################
    
    #Comparison with KISS-ICP map
    # map_1 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Map_GPS_35.pcd")
    # map_2 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Map_GPS_26.pcd")
    # map_kiss = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_KISSICP\Run2Map.pcd")
    
    # R = map_2.get_rotation_matrix_from_xyz((0,0,np.pi))
    # map_2.rotate(R, center=map_2.get_center())
    
    # #Map 26 & Map 35 - notup to date anymore
    # map_1 = map_1.translate((17,-28,0), relative = True)
    # R = map_1.get_rotation_matrix_from_xyz((-0.62*(np.pi/180),0,0))
    # map_1.rotate(R, center=map_1.get_center())
    
    # final_map_list = [map_1,map_2]
    
    # map_point_cloud = np.zeros((1,3))
    # for el in final_map_list:
    #     arr = np.asarray(el.points)
    #     map_point_cloud = np.concatenate((map_point_cloud,arr),axis = 0 )
    
    
    # map_point_cloud = np.delete(map_point_cloud,0,0) 
    
    # pcd_map = o3d.geometry.PointCloud()
    # pcd_map.points = o3d.utility.Vector3dVector(map_point_cloud)   
    
    # pcd_map.points 
    # pcd_map = pcd_map.voxel_down_sample(voxel_size=0.1)
    
    # pcd_map.points
    
    # R = map_kiss.get_rotation_matrix_from_xyz((0,0,-230*(np.pi/180)))
    # map_kiss.rotate(R, center=map_kiss.get_center())
    

######################################################################
######### Load the online scans from runs not used for mapping #######
######################################################################

if load_online_scans == True:
    
    final_map = o3d.io.read_point_cloud(path_to_final_map)
    
    path_gps = os.path.join(path_origin,file_gps)
    
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
    k = 5
    point_cloud = np.zeros((1,3))
    for i in glob.glob(path_to_scans):  # * means inner directory
        if counter == 0 and k < len(list_gps):
            print(i)
            print(k)
            print(df_gps.loc[k, '%time'])
            #print(counter)
            pcd = o3d.io.read_point_cloud(i)
            #pcd = pcd.voxel_down_sample(voxel_size=0.1)
            
            #new_file_name = 'PointClouds/' + os.path.basename(path_origin) + '_TS-' + os.path.basename(i)
            #new_file_path = os.path.join(path_to_data,new_file_name)
            #o3d.io.write_point_cloud(new_file_path,pcd)
            
            #print(new_file_path)
            
            T = np.eye(4)
            T[:3, :3] = pcd.get_rotation_matrix_from_quaternion((list_gps[k][6],list_gps[k][3], list_gps[k][4], list_gps[k][5]))
            #T[:3, :3] = pcd.get_rotation_matrix_from_xyz((0,0,-60*(np.pi/180)))
            T[0, 3] = list_gps[k][0]
            T[1, 3] = list_gps[k][1]
            T[2,3] = list_gps[k][2]
            
            #R = pcd.get_rotation_matrix_from_quaternion((list_gps[k][3],list_gps[k][4], list_gps[k][5], list_gps[k][6]))
            #pcd.rotate(R, center=pcd.get_center())
            #pcd.translate((list_gps[k][0], list_gps[k][1], list_gps[k][2]), relative = False)
            
            pcd = pcd.transform(T)
            pcd = pcd.translate(static_offset, relative = True)
            
            online_scans.append(pcd)
            list_timestamps.append(i)
            list_df_gps_index.append(k)

        
        counter += 1
        if counter == 300:
            counter = 0
        else: pass
        
        k += 5
    
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
    
    path_GT_csv = os.path.join(path_to_data ,file_GT_csv)
    df_gps_GT.to_csv(path_GT_csv,sep = ',', index = False, header = True)
    
    
    for el in online_scans:
        el.paint_uniform_color([0.5,0.0,0.5])
    
    online_scans.append(final_map)

    
    o3d.visualization.draw_geometries(online_scans,
                                      zoom=1,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[0,0, 0],
                                      up=[-0.0694, -0.9768, 0.2024])
    
    
    ###############################
    
    pc_scan = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Run5PC_GPS\1692975926.319923401.pcd")
    pc_map = o3d.io.read_point_cloud(r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Evaluations\01_TRC_Skidpad_Data\01_Map_GPS_TRC_1234.pcd")
    
    pc_scan.translate([-220.98,358.12,4.18])
    
    
    o3d.visualization.draw_geometries([pc_scan,pc_map],
                                      zoom=1,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[0,0, 0],
                                      up=[-0.0694, -0.9768, 0.2024])