# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 14:33:39 2023

@author: Johanna Hable

Notes:
    - This script provides a visualization of the iteration steps during alignment of source and target point cloud of the evaluation results on the own OSU dataset
    - Compared to the visualization used during baseline studies on Moriyama the initial pose is perturbated in x,y and yaw angle at the same time
    - Most of the code is similar to the visualization code of the baseline studies 'Visualization_Modified'
"""

import open3d as o3d
import numpy as np
import pandas as pd
import glob
import os
import copy
from scipy.spatial.transform import Rotation as R
import time
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from PIL import Image, ImageDraw, ImageFont

def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()
    


path_map = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Evaluations\02_Route_1_Data\FinalMap_Route1.pcd"
path_scans = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Evaluations\02_Route_1_Data\pcd_prepro"
path_GT_csv = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Evaluations\02_Route_1_Data\GT_Poses_OnlineScans_500.csv"
path_to_file = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Evaluations\02_Route_1_Evaluation"

path_to_ITcsv = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Evaluations\02_Route_1_Evaluation\D000_Iter_P2Point-ICP_Route01.csv"

Idx_timestamp = 0
#Idx_axis = 0  # 0 - x, 1-y, 2-z, 3-alpha, 4-beta, 5-gamma
#Idx_init = 1 # Which one of the 17 evaluation points between -2 and 2 or -pi/4 and pi/4
Value_init = ' -45 °' #If known please add the value as string (value + unit)

#Load GT poses from csv
df_GT = pd.read_csv(path_GT_csv, delimiter = ',', header = 0)


#Save GT position (x,y,z) and orientation as quaternions (x,y,z,w) in numpy array
arr_GT_poses = df_GT.iloc[:,2:9].to_numpy(dtype = 'float')
timestamps = df_GT["%time"].values.tolist()

# Choose the GT of the timestamp to be evaluated
#sample_step = 100
#arr_GT_poses = np.zeros((5,7))

#i = 3
#timestamps = []
#for j in range(0, len(arr_GT_poses)):
  
#    arr_GT_poses[j,:] = np.asarray(df_GT.iloc[i, 4:11])
#    timestamps.append(df_GT.iloc[i,2])
#    i += sample_step

list_path_pc = [] 
for i in list(glob.glob(path_scans)):
    path_scan = os.path.join(path_scans,i)
    list_path_pc.append(str(path_scan))

#list_path_pc = df_GT["pc.timestamp.path"].values.tolist()
#list_path_pc = [path_pc_1, path_pc_2, path_pc_3, path_pc_4, path_pc_5]

pc_map = o3d.io.read_point_cloud(path_map)  #map point cloud
target_pc = copy.deepcopy(pc_map)

path_pc = list_path_pc[Idx_timestamp]
init_pose = arr_GT_poses[Idx_timestamp]
timestamp = timestamps[Idx_timestamp]

pc_scan = o3d.io.read_point_cloud(path_pc)  #online scan point cloud from above declared timestamp
source_pc = copy.deepcopy(pc_scan)

r = R.from_quat(init_pose[3:7])
R_matrix = r.as_matrix()
R_vec = r.as_rotvec()
R_Euler = r.as_euler('xyz')

t_vec = np.zeros((3,1))

for k in range(0,len(t_vec)):
    t_vec[k] = init_pose[k]

tranform_raw = np.hstack((R_matrix, t_vec)) # 3x3 matrix for rotation and translation
transform_GT = np.vstack((tranform_raw,np.array([0,0,0,1]))) # 4x4 homography transformation matrix


# No dynamic point cloud loader for the map like in Autoware
# (Loads only the part of the map needed for search space of localization algorithm)
# We crop the map point cloud around the GT pose; limits can be set individually
min_bound = np.zeros((3,1))
max_bound = np.zeros((3,1))

delta = np.array([[180],             # in x-direction +/- 200 m
                  [180],             # in y-direction +/- 200 m
                  [100]])            # in z-direction +/- 100 m

for i in range(0,len(t_vec)):
    min_bound[i,0]= t_vec[i,0] - delta[i,0]
    max_bound[i,0] = t_vec[i,0] + delta[i,0]

bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
target_pc = o3d.geometry.PointCloud.crop(target_pc,bbox)

#target_pc.paint_uniform_color([0.5,0.5,0.5])

df = pd.read_csv(path_to_ITcsv, delimiter = ';', header = 0)

## Axis 0,1,2 are for translation
## Axis 3,4,5 are for rotation

timestamps = df.groupby('Timestamp').groups.keys()
#axes = df.groupby('Axis').groups.keys()
#init_values = df.groupby('Init Error (Trans or Rot)').groups.keys()

timestamp_1 = list(timestamps)[Idx_timestamp]
#axis_1 = list(axes)[Idx_axis]
#init_value_1 = list(init_values) [Idx_init]

df = df.loc[df['Timestamp'] == timestamp_1]
initial_values = str(df.iloc[0,4])
initial_values = initial_values.split(',')


### Initial transformation matrix
transl_values = [0,0,0]
rot_values = [0,0,0]

transl_values[0] = float(initial_values[0])   # x
transl_values[1] = float(initial_values[1]) # y
rot_values[2] = float(initial_values[2]) # yaw

transform_init = copy.deepcopy(transform_GT)
R_Euler_init = copy.deepcopy(R_Euler)

for i in range(0,3):
    transform_init [i,3] = transform_init [i,3] + transl_values[i]
    R_Euler_init[i] = R_Euler_init[i] + rot_values[i]

r = R.from_euler('xyz',R_Euler_init)
R_matrix_init = r.as_matrix()

transform_init[0:3,0:3] = R_matrix_init

print('Found %s iteration steps for selected timestamp' %(str(len(df))))


i = 0

mesh_map = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=20, origin=[t_vec.T[0][0]-2, t_vec.T[0][1]-45, t_vec.T[0][2]]) #[t_vec.T[0][0]-20, t_vec.T[0][1]-20, t_vec.T[0][2]]

mesh_lidar = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 15, origin = [0,0,0])
mesh_lidar.transform(transform_init)
source_temp = copy.deepcopy(source_pc)

source_temp.paint_uniform_color([0.0,0.0,0.0])


WINDOW_WIDTH=1920 
WINDOW_HEIGHT=1080 

vis = o3d.visualization.Visualizer()           #Visualizer
vis.create_window(width = WINDOW_WIDTH, height = WINDOW_HEIGHT, left = 10, top = 10)
ctr = vis.get_view_control()

#Add text

text = "Own OSU dataset - Route 1 \nTimestamp: " + str(Idx_timestamp) + "\n\nIterations: " + str(len(df))
img = Image.new('RGB', (WINDOW_WIDTH, WINDOW_HEIGHT), color = (255,255,255,0))
font = ImageFont.truetype(r'arial.ttf', 25)
d = ImageDraw.Draw(img)
d.text((800,10), text, font=font, fill=(0,0,0), align = 'center')
img.save('pil_text.png')

im = o3d.io.read_image("./pil_text.png")
vis.add_geometry(im)

vis.add_geometry(source_temp, reset_bounding_box = True)
vis.add_geometry(target_pc, reset_bounding_box = True)
vis.add_geometry(mesh_map)
vis.add_geometry(mesh_lidar)


ctr.set_zoom(0.8)

source_temp.transform(transform_init)

vis.reset_view_point(True)
vis.update_geometry(source_temp)
vis.poll_events()
vis.update_renderer()

#vis.run()
#vis.run()  # user changes the view and press "q" to terminate
#param = vis.get_view_control().convert_to_pinhole_camera_parameters()
#o3d.io.write_pinhole_camera_parameters('Test.json', param)
#vis.destroy_window()


time.sleep(5)

transform_init_inv = np.linalg.inv(transform_init)
source_temp.transform(transform_init_inv)
mesh_lidar.transform(transform_init_inv)

t_matrix = np.zeros((4,4))
t_matrix_inv = np.identity(4)

save_image = False

#vis.remove_geometry(target_pc)
#vis.remove_geometry(mesh_map)
#vis.remove_geometry(source_temp)
#vis.remove_geometry(mesh_lidar)
#time.sleep(0.1)

#text_position = np.zeros((3,1))
#text_position [0][0] = t_vec.T[0][0] - 2 
#text_position [1][0] = t_vec.T[0][1] - 45 
#text_position [2][0] = t_vec.T[0][2]
#text = o3d.visualization.gui.Label3D ("Test",text_position )
#vis.add_geometry(text)
#text.text = "Test"
#text.position = np.zeros((3,1)) 
#np.array([t_vec.T[0][0]-2],
#                 [t_vec.T[0][1]-45],
#                 [t_vec.T[0][2]], dtype = np.float32)



for i in range(0,len(df)):
    
    t_matrix[0,0] = df.iloc[i,7]       # maybe start at 8 ???
    t_matrix[0,1] = df.iloc[i,8]
    t_matrix[0,2] = df.iloc[i,9]
    t_matrix[0,3] = df.iloc[i,10]
    t_matrix[1,0] = df.iloc[i,11]
    t_matrix[1,1] = df.iloc[i,12]
    t_matrix[1,2] = df.iloc[i,13]
    t_matrix[1,3] = df.iloc[i,14]
    t_matrix[2,0] = df.iloc[i,15]
    t_matrix[2,1] = df.iloc[i,16]
    t_matrix[2,2] = df.iloc[i,17]
    t_matrix[2,3] = df.iloc[i,18]
    t_matrix[3,0] = df.iloc[i,19]
    t_matrix[3,1] = df.iloc[i,20]
    t_matrix[3,2] = df.iloc[i,21]
    t_matrix[3,3] = df.iloc[i,22]
    
    #vis.remove_geometry(im)
    #vis.poll_events()
    #vis.update_renderer()
    #vis.remove_geometry(target_pc)
    #vis.remove_geometry(mesh_map)
    #vis.remove_geometry(source_temp)
    #vis.remove_geometry(mesh_lidar)
    time.sleep(0.1)


    #vis = o3d.visualization.Visualizer()
    #vis.create_window()
    #vis.add_geometry(source_pc, reset_bounding_box = True)
    #vis.add_geometry(target_pc, reset_bounding_box = True)
    
    text = "\n\nIteration: " + str(i)
    img = Image.new('RGB', (WINDOW_WIDTH, WINDOW_HEIGHT), color = (255,255,255,0))
    font = ImageFont.truetype(r'arial.ttf', 25)
    d = ImageDraw.Draw(img)
    d.text((1200,10), text, font=font, fill=(0,0,0), align = 'center')
    img.save('pil_text_2.png')

    im = o3d.io.read_image("./pil_text_2.png")
    

    
    source_temp.transform(t_matrix_inv)
    mesh_lidar.transform(t_matrix_inv)
    
    time.sleep(0.05)

    
    source_temp.transform(t_matrix)
    mesh_lidar.transform(t_matrix)
    

    #vis.reset_view_point(True)
    #vis.add_geometry(im_2)
    #vis.poll_events()
    #vis.update_renderer()
    #time.sleep(0.1)
    #vis.reset_view_point(True)
    #vis.add_geometry(im)
    #vis.poll_events()
   # vis.update_renderer()
    #time.sleep(0.5)
    vis.reset_view_point(True)
    vis.update_geometry(target_pc)
    vis.poll_events()
    vis.update_renderer()
    vis.reset_view_point(True)
    vis.update_geometry(mesh_map)
    vis.poll_events()
    vis.update_renderer()
    vis.reset_view_point(True)
    vis.update_geometry(mesh_lidar)
    vis.poll_events()
    vis.update_renderer()
    vis.reset_view_point(True)
    vis.update_geometry(source_temp)
    vis.poll_events()
    vis.update_renderer()
    

    
    #time.sleep(1)
    #vis.remove_geometry(im_2)
    #vis.poll_events()
    #vis.update_renderer()
    time.sleep(1)
    
    if save_image:
        vis.capture_screen_image("temp_%04d.jpg" % i)
    
    t_matrix_inv = np.linalg.inv(t_matrix)


source_GT = copy.deepcopy(source_pc)
source_GT.paint_uniform_color([1.0, 0, 0])
source_GT.transform(transform_GT)

dists = source_GT.compute_point_cloud_distance(source_temp)
dists = np.asarray(dists)
ind = np.where(dists > 0.01)[0]
#pcd_without_chair = pcd.select_by_index(ind)
print("\nAveraged distance over all points betweeen final pose and Ground Truth in m: %f\nStandard deviation: %f" %(dists.mean(), dists.std()))


vis.reset_view_point(True)
vis.add_geometry(source_GT)
vis.poll_events()
vis.update_renderer()
vis.run()
vis.clear_geometries()
vis.destroy_window() 
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)   