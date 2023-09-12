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




maps = []

counter = 0
point_cloud = np.zeros((1,3))
for i in glob.glob("D:\\Johanna\\bags\TRC-Skidpad\\Run6TRCSkidpad\\*.pcd"):  # * means inner directory
    if counter == 0:
        print(i)
        pcd = o3d.io.read_point_cloud(i)
        pcd = pcd.voxel_down_sample(voxel_size=0.1)
        a = np.array(pcd.points)
        #point_cloud.append(a)
        point_cloud = np.concatenate((point_cloud,a),axis = 0 )
    
    counter += 1
    if counter == 10:
        counter = 0
    else: pass
    


len(point_cloud)

pcd_map = o3d.geometry.PointCloud()
pcd_map.points = o3d.utility.Vector3dVector(point_cloud)

pcd_map_down = pcd_map.voxel_down_sample(voxel_size=0.1)

len(pcd_map_down.points)

o3d.io.write_point_cloud("D:\\Johanna\\bags\TRC-Skidpad\\Run6Map.pcd", pcd_map_down)

maps.append(pcd_map_down)

len(maps)

maps[0].paint_uniform_color([1.0,0.0,0.0])
maps[2].paint_uniform_color([0.0,1.0,0.0])
maps[4].paint_uniform_color([0.0,0.0,1.0])


o3d.visualization.draw_geometries([maps[0],maps[2],maps[4]],
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


########################################################### Novatel BESTPOS ###############
maps = []

file_gps = "GNSSRTK_Pose_Run1.csv"
path_origin = r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Run1PC_GPS"

path_gps = os.path.join(path_origin,file_gps)

df_gps = pd.read_csv(path_gps, delimiter = ',', header = 0, engine = 'python', encoding = 'utf-8')#unicode_escape')
df_gps[['field.lat', 'field.lon', 'field.hgt']]

df_gps[['field.lon']].max()
df_gps[['field.lon']].min()

df_gps.dtypes

list_gps = []
for ind in df_gps.index:
    transl_gps = np.array([df_gps['field.lat'][ind], df_gps['field.lon'][ind], df_gps['field.hgt'][ind]])
    print(transl_gps)
    #(transl_gps)
    list_gps.append(transl_gps)


counter = 0
k = 0
point_cloud = np.zeros((1,3))
for i in glob.glob("D:\\Johanna\\bags\TRC-Skidpad\\Map_with_GPS\\Run1PC_GPS\\*.pcd"):  # * means inner directory
    if counter == 0 and k < len(list_gps):
        print(i)
        print(k)
        print(counter)
        pcd = o3d.io.read_point_cloud(i)
        #pcd = pcd.voxel_down_sample(voxel_size=0.1)
        pcd.translate((list_gps[k][0]*10, list_gps[k][1]*10, 0), relative = False)
        a = np.asarray(pcd.points)
        #point_cloud.append(a)
        point_cloud = np.concatenate((point_cloud,a),axis = 0 )
    
    counter += 1
    if counter == 10:
        counter = 0
    else: pass
    
    k += 1 

point_cloud = np.delete(point_cloud,0,0)    
    
pcd_map = o3d.geometry.PointCloud()
pcd_map.points = o3d.utility.Vector3dVector(point_cloud)

pcd_map_down = pcd_map.voxel_down_sample(voxel_size=0.1)

len(pcd_map_down.points)

pcd_map_down.paint_uniform_color([1.0,0.0,0.0])

#mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
#R = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi ))
#pcd_map_down.rotate(R, center=(0, 0, 0))

maps.append(pcd_map_down)

#################################################################################

pcd_lidar = o3d.io.read_point_cloud("D:\\Johanna\\bags\TRC-Skidpad\\Run1Map.pcd")
pcd_lidar.paint_uniform_color([0.0,0.0,1.0])


###################### Novatel ODOM ################################################
maps = []

file_gps = "ODOM_GPS_Run6.csv"
path_origin = r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Run6PC_GPS"

path_gps = os.path.join(path_origin,file_gps)

df_gps = pd.read_csv(path_gps, delimiter = ',', header = 0, engine = 'python', encoding = 'utf-8')#unicode_escape')
#df_gps[['field.pose.pose.position.x', 'field.pose.pose.position.y', 'field.pose.pose.position.z']]


df_gps.dtypes

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
k = 5
point_cloud = np.zeros((1,3))
for i in glob.glob("D:\\Johanna\\bags\TRC-Skidpad\\Map_with_GPS\\Run6PC_GPS\\*.pcd"):  # * means inner directory
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
        
        pcd = pcd.transform(T)
        
        a = np.asarray(pcd.points)
        #point_cloud.append(a)
        point_cloud = np.concatenate((point_cloud,a),axis = 0 )
    
    counter += 1
    if counter == 10:
        counter = 0
    else: pass
    
    k += 5

point_cloud = np.delete(point_cloud,0,0)    

pcd_map = o3d.geometry.PointCloud()
pcd_map.points = o3d.utility.Vector3dVector(point_cloud)

pcd_map_down = pcd_map.voxel_down_sample(voxel_size=0.1)

len(pcd_map_down.points)

#pcd_map_down.paint_uniform_color([1.0,0.0,0.0])

#mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
#R = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi ))
#pcd_map_down.rotate(R, center=(0, 0, 0))

maps.append(pcd_map_down)


#####
#####
pc_1 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Run1PC_GPS\1692975298.153691530.pcd")
pc_2 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Run1PC_GPS\1692975390.033923149.pcd")

mesh_1 = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh_2 = o3d.geometry.TriangleMesh.create_coordinate_frame()

#pc_1.translate((-1.5,0,2))
#pc_2.translate((-1.5,0,2))

pc_1.translate((283737.7408 ,4465501.1162,0), relative = False)   #336.6636+2.0
pc_2.translate((284287.9961 -1.5,4464550.3370,2), relative = False)   #333.2362+2.0

mesh_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 5, origin = (283737.7408 ,4465501.1162,0))

mesh_1.translate((283737.7408,4465501.1162 -10,0), relative = False)
mesh_2.translate((284287.9961,4464550.3370,0), relative = False) 

R1 = pc_1.get_rotation_matrix_from_quaternion((-0.0070716048161214105,-0.009820895726918398,-0.49714285313124085,0.8675843048332392))
pc_1.rotate(R1, center=pc_1.get_center())

R2 = pc_2.get_rotation_matrix_from_quaternion((-0.0019136898290919519,1.1853186742495961e-05,-0.4974765819227137,0.8674752953768963))
pc_2.rotate(R2, center=pc_2.get_center())


mesh_1.rotate(R1, center = mesh_1.get_center())
mesh_2.rotate(R2, center = mesh_2.get_center())
mesh_1.scale(5, center=mesh_1.get_center())
mesh_2.scale(5, center=mesh_2.get_center())


pc_1.paint_uniform_color([1.0,0.0,0.0])
pc_2.paint_uniform_color([0.0,1.0,0.0])
#######
######

maps[0].paint_uniform_color([1.0,0.0,0.0])
maps[1].paint_uniform_color([0.0,0.5,0.5])
maps[2].paint_uniform_color([0.0,1.0,0.0])
maps[3].paint_uniform_color([0.5,0.0,0.5])
maps[4].paint_uniform_color([0.0,0.0,1.0])
maps[5].paint_uniform_color([0.5,0.5,0.0])

o3d.visualization.draw_geometries([maps[0],maps[1],maps[2], maps[3], maps[4], maps[5]],
                                  zoom=10.34,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[283737,4465501, 336],
                                  up=[-0.0694, -0.9768, 0.2024])

o3d.io.write_point_cloud("D:\\Johanna\\bags\\TRC-Skidpad\\Map_with_GPS\\Map_GPS_Run6.pcd", maps[5])


#######################################
pc_1 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Map_GPS_Run1.pcd")
pc_2 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Map_GPS_Run2.pcd")
pc_3 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Map_GPS_Run3.pcd")
pc_4 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Map_GPS_Run4.pcd")
pc_5 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Map_GPS_Run5.pcd")
pc_6 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Map_GPS_Run6.pcd")

pc_1 = pc_1.translate((-66,113.5,0), relative = True)
#pc_2 none
#pc_3 none
pc_4 = pc_4.translate((38,-67,0), relative = True)
pc_5 = pc_5.translate((3,-3,0), relative = True)
#pc_6 none

list_pc_map = [pc_1,pc_2,pc_3,pc_4]


map_point_cloud = np.zeros((1,3))
for el in list_pc_map:
    arr = np.asarray(el.points)
    map_point_cloud = np.concatenate((map_point_cloud,arr),axis = 0 )


map_point_cloud = np.delete(map_point_cloud,0,0) 

pcd_map = o3d.geometry.PointCloud()
pcd_map.points = o3d.utility.Vector3dVector(map_point_cloud)   

pcd_map.points 
pcd_map = pcd_map.voxel_down_sample(voxel_size=0.1)

pcd_map.points

#center = np.negative(pcd_map.get_center())
pcd_map = pcd_map.translate((-283960,-4465143,-334), relative = True) 


o3d.visualization.draw_geometries([pcd_map],
                                  zoom=1,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[0,0, 0],
                                  up=[-0.0694, -0.9768, 0.2024])



o3d.io.write_point_cloud("D:\\Johanna\\bags\\TRC-Skidpad\\Map_with_GPS\\Map_GPS_26.pcd", pcd_map)


map_1 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Map_GPS_35.pcd")
map_2 = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Map_GPS_26.pcd")
map_kiss = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_KISSICP\Run2Map.pcd")

R = map_2.get_rotation_matrix_from_xyz((0,0,np.pi))
map_2.rotate(R, center=map_2.get_center())

#Map 26 & Map 35
map_1 = map_1.translate((17,-28,0), relative = True)
R = map_1.get_rotation_matrix_from_xyz((-0.62*(np.pi/180),0,0))
map_1.rotate(R, center=map_1.get_center())

final_map_list = [map_1,map_2]

map_point_cloud = np.zeros((1,3))
for el in final_map_list:
    arr = np.asarray(el.points)
    map_point_cloud = np.concatenate((map_point_cloud,arr),axis = 0 )


map_point_cloud = np.delete(map_point_cloud,0,0) 

pcd_map = o3d.geometry.PointCloud()
pcd_map.points = o3d.utility.Vector3dVector(map_point_cloud)   

pcd_map.points 
pcd_map = pcd_map.voxel_down_sample(voxel_size=0.1)

pcd_map.points

R = map_kiss.get_rotation_matrix_from_xyz((0,0,-230*(np.pi/180)))
map_kiss.rotate(R, center=map_kiss.get_center())

#Online scans from run1 and run4
file_gps = "ODOM_GPS_Run1.csv"
path_origin = r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Run1PC_GPS"

path_gps = os.path.join(path_origin,file_gps)

df_gps = pd.read_csv(path_gps, delimiter = ',', header = 0, engine = 'python', encoding = 'utf-8')#unicode_escape')
#df_gps[['field.pose.pose.position.x', 'field.pose.pose.position.y', 'field.pose.pose.position.z']]


df_gps.dtypes

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
counter = 0
k = 5
point_cloud = np.zeros((1,3))
for i in glob.glob("D:\\Johanna\\bags\TRC-Skidpad\\Map_with_GPS\\Run1PC_GPS\\*.pcd"):  # * means inner directory
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
        
        pcd = pcd.transform(T)
        pcd = pcd.translate((-283960,-4465142,-334), relative = True)
        
        online_scans.append(pcd)
        
     
    
    counter += 1
    if counter == 10:
        counter = 0
    else: pass
    
    k += 5

len(online_scans)





pc_scan = o3d.io.read_point_cloud(r"D:\Johanna\bags\TRC-Skidpad\Map_with_GPS\Run1PC_GPS\1692975318.693958521.pcd")
pc_scan.paint_uniform_color([0.5,0.0,0.5])

o3d.visualization.draw_geometries(online_scans,
                                  zoom=1,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[0,0, 0],
                                  up=[-0.0694, -0.9768, 0.2024])