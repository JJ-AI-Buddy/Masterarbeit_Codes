# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 22:57:59 2023

@author: Johanna
"""

#Use open3d.t.geometry to yield results of each iteration step when using ICP
#Animated visualization of the registration process



import open3d as o3d
import copy
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0] 
    q1 = Q[1] 
    q2 = Q[2] 
    q3 = Q[3] 
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1)  -1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2)  -1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

path_map = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\Moriyama_Map.pcd"
path_pc = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\PointClouds_Moriyama_140\1427157790778626.pcd"
path_csv = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\14_Local_Pose.csv"

device = 'CPU:0'
dtype = 'float32'


df_GT = pd.read_csv(path_csv, delimiter = ',', header = 0)

pc_1 = o3d.io.read_point_cloud(path_pc)
pc_2 = o3d.io.read_point_cloud(path_map)

pcd_1 = o3d.t.io.read_point_cloud(path_pc)
pcd_2 = o3d.t.io.read_point_cloud(path_map)

source_pcd = copy.deepcopy(pcd_1)
source_pc = copy.deepcopy(pc_1)
target_pcd = copy.deepcopy(pcd_2)
target_pc = copy.deepcopy(pc_2)


sample_step = 50
arr_GT_poses = np.zeros((5,7))

i = 3
for j in range(0, len(arr_GT_poses)):
  
    arr_GT_poses[j,:] = np.asarray(df_GT.iloc[i, 4:11])
    i += sample_step

arr_GT_poses

init_pose = arr_GT_poses[0]
init_pose

#R_matrix = quaternion_rotation_matrix(init_pose[3:7])
#R_matrix = -R_matrix


#R = source_pc.get_rotation_matrix_from_quaternion(init_pose[3:7])


r = R.from_quat(init_pose[3:7])
R_matrix = r.as_matrix()
#R = np.asarray([[1, 0, 0],
#                         [0, 1, 0],
#                         [0, 0, 1]])

t_vec = np.zeros((3,1))

for k in range(0,len(t_vec)):
    t_vec[k] = init_pose[k]

tranform_raw = np.hstack((R_matrix, t_vec))
tranform_raw

transform_GT = np.vstack((tranform_raw,np.array([0,0,0,1])))
transform_GT



min_bound = np.zeros((3,1))
max_bound = np.zeros((3,1))

delta = np.array([[200],
                  [200],
                  [100]])

for i in range(0,len(t_vec)):
    min_bound[i,0]= t_vec[i,0] - delta[i,0]
    max_bound[i,0] = t_vec[i,0] + delta[i,0]
    
min_bound[0][0]


bbox = o3d.t.geometry.AxisAlignedBoundingBox(o3d.core.Tensor([min_bound[0][0], min_bound[1][0], min_bound[2][0]]),
                                                             o3d.core.Tensor([max_bound[0][0], max_bound[1][0], max_bound[2][0]]))
target_pcd_cropped = o3d.t.geometry.PointCloud.crop(target_pcd,bbox)

bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound,max_bound)
target_pc_cropped = o3d.geometry.PointCloud.crop(target_pc,bbox)



#source_pc.paint_uniform_color([1, 0.0, 0.5])
#target_pc.paint_uniform_color([0, 1, 0.5])

#threshold = 0.05
#trans_offset = np.asarray([[1, 0, 0, 0.0],
#                         [0, 1, 0, 0.0],
#                         [0, 0, 1, 0],
#                         [0.0, 0.0, 0.0, 1.0]])

#target_pc.transform(trans_offset)

source_pcd.transform(transform_GT)
source_pc.transform(transform_GT)

vis = o3d.visualization.Visualizer()
vis.create_window('Point Cloud')
vis.add_geometry(source_pc)
vis.destroy_window()


vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(source_pc)
vis.add_geometry(target_pc_cropped)
threshold = 0.05
icp_iteration = 5
save_image = False

for i in range(icp_iteration):
   reg_p2p = o3d.pipelines.registration.registration_icp(
       source_pc, target_pc, threshold, trans_init,
       o3d.pipelines.registration.TransformationEstimationPointToPoint(),
       o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
   source_pc.transform(reg_p2p.transformation)
   vis.update_geometry(source_pc)
   vis.poll_events()
   vis.update_renderer()
   if save_image:
       vis.capture_screen_image("temp_%04d.jpg" % i)
vis.destroy_window()

o3d.visualization.draw([target_pcd_cropped,source_pcd],
                                 zoom=0.8, front=[0, 0, 1],
                                 lookat=[-14700, -85000, 44],
                                 up=[0, 1, 0],
                                 point_show_normal=False)

o3d.visualization.draw([target_pcd_cropped, source_pcd])

threshold = 0.5
trans_init = transform_GT.copy()
trans_init [0,3] += 0.5

o3d.core.Tensor(trans_init)


evaluation = o3d.t.pipelines.registration.evaluate_registration(
    source_pcd, target_pcd_cropped, threshold, o3d.core.Tensor(trans_init))
print(evaluation)

def registration_callback(curr_transform):
    print(curr_transform)

print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source_pc, target_pc, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 30),registration_callback)
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)

source_pc.transform(reg_p2p.transformation)
