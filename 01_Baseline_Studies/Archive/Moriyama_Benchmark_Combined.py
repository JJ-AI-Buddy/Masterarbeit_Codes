# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 22:57:59 2023

@author: Johanna
"""

#Use open3d.t.geometry to yield results of each iteration step when using ICP
#Yield iteration, convergence time
#Animated visualization of the registration process

# Write results in .csv file for data evaluation & graphic representation
# Problem: Timestamps not synchronized!?!?


import open3d as o3d
import copy
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os
from datetime import datetime


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


# Input: Define the path direction of the map to use, the specific point cloud from a defined timestamp and the path direction to the csv file containing the GT poses (from NDT localization in Autoware) of the Localizer
path_map = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\Moriyama_Map.pcd"
path_GT_csv = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\14_Local_Pose.csv"
path_to_file = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\03_Moriyama_Evaluation"

ID = '013'

name_txt = str(ID) + '_RotBaselineICPMoriyama.txt'
path_txt = os.path.join(path_to_file, name_txt)
name_csv = str(ID) + '_RotBaselineICPMoriyama.csv'
path_csv = os.path.join(path_to_file,name_csv)

#Prepare CSV file
with open(path_csv, 'w') as f:
    f.write('ID; Timestamp GT Pose; Axis; Initial Transl alpha; Initial Transl beta; Initial Transl gamma; Initial fitness; Initial RMSE Inliers; Initial Correspondence Set Size; Initial Transl. Error [m]; Initial Rot. Error 1 [°]; Initial Rot. Error 2 [°]; Fitness; RMSE Inliers; Correspondence Set Size; Transl. Error [m]; Rot. Error 1 [°]; Rot. Error 2 [°]\n')

#Prepare txt file
with open(path_txt, 'w') as f:
    f.write('Evaluation of ICP algorithm for map matching on Moriyama dataset \n' + str(datetime.now()) + "\n\n")

bool_trans = False
bool_rot = True

#device = 'CPU:0'
#dtype = 'float32'

#Load GT poses from csv
df_GT = pd.read_csv(path_GT_csv, delimiter = ',', header = 0)

# Choose the GT of the timestamp to be evaluated
sample_step = 100

arr_GT_poses = np.zeros((5,7))

i = 3

timestamps = []
for j in range(0, len(arr_GT_poses)):
  
    arr_GT_poses[j,:] = np.asarray(df_GT.iloc[i, 4:11])
    timestamps.append(df_GT.iloc[i,2])
    i += sample_step


#Point Clouds
path_map = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\Moriyama_Map.pcd"
path_pc_1 = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\PointClouds_Moriyama_140\1427157790678528.pcd"
path_pc_2 = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\PointClouds_Moriyama_140\1427157800687781.pcd"
path_pc_3 = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\PointClouds_Moriyama_140\1427157810697012.pcd"
path_pc_4 = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\PointClouds_Moriyama_140\1427157820706337.pcd"
path_pc_5 = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\PointClouds_Moriyama_140\1427157830815641.pcd"


list_path_pc = [path_pc_1, path_pc_2, path_pc_3, path_pc_4, path_pc_5]


#Load map
pc_map = o3d.io.read_point_cloud(path_map)  #map point cloud
target_pc = copy.deepcopy(pc_map)

for x in range(0, len(arr_GT_poses)):
    #Choose which timestamp should be evaluated
    number_pose = x

    path_pc = list_path_pc[number_pose]
    init_pose = arr_GT_poses[number_pose]
    timestamp = timestamps[number_pose]

    #Load online scans
    pc_scan = o3d.io.read_point_cloud(path_pc)  #online scan point cloud from above declared timestamp
    source_pc = copy.deepcopy(pc_scan)

    #pcd_1 = o3d.t.io.read_point_cloud(path_pc)
    #pcd_2 = o3d.t.io.read_point_cloud(path_map)

    # Deep copy the point clouds to keep the original
    #source_pcd = copy.deepcopy(pcd_1)
    #target_pcd = copy.deepcopy(pcd_2)

    #Downsampling
    #source_pc = source_pc.voxel_down_sample(0.5)


    #R_matrix = quaternion_rotation_matrix(init_pose[3:7])
    #R_matrix = -R_matrix
    #R = source_pc.get_rotation_matrix_from_quaternion(init_pose[3:7])

    # Derive the GT transformation matrix from GT pose (3D translation + 4 quaternions for rotation)
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

    delta = np.array([[200],             # in x-direction +/- 200 m
                      [200],             # in y-direction +/- 200 m
                      [100]])            # in z-direction +/- 100 m

    for i in range(0,len(t_vec)):
        min_bound[i,0]= t_vec[i,0] - delta[i,0]
        max_bound[i,0] = t_vec[i,0] + delta[i,0]
    

    #bbox = o3d.t.geometry.AxisAlignedBoundingBox(o3d.core.Tensor([min_bound[0][0], min_bound[1][0], min_bound[2][0]]),
    #                                                            o3d.core.Tensor([max_bound[0][0], max_bound[1][0], max_bound[2][0]]))
    #target_pcd_cropped = o3d.t.geometry.PointCloud.crop(target_pcd,bbox)

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound,max_bound)
    target_pc_cropped = o3d.geometry.PointCloud.crop(target_pc,bbox)



    #source_pc.paint_uniform_color([1, 0.0, 0.5])
    #target_pc.paint_uniform_color([0, 1, 0.5])


    #source_pcd.transform(transform_GT)
    #source_pc.transform(transform_GT)

    #vis = o3d.visualization.Visualizer()
    #vis.create_window('Point Cloud')
    #vis.add_geometry(source_pc)
    #vis.destroy_window()


    #vis = o3d.visualization.VisualizerWithEditing()
    #vis.create_window('TEST')
    #vis.clear_geometries()
    #vis.add_geometry(source_pc, reset_bounding_box= True)
    #vis.run()
    #vis.add_geometry(target_pc_cropped)
    #vis.update_geometry()
    #vis.clear_geometry(source_pc)
    #vis.destroy_window()
    #threshold = 0.05
    #icp_iteration = 5
    #save_image = False

    #for i in range(icp_iteration):
        #   reg_p2p = o3d.pipelines.registration.registration_icp(
        #       source_pc, target_pc, threshold, trans_init,
#       o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#       o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
#   source_pc.transform(reg_p2p.transformation)
#   vis.update_geometry(source_pc)
#   vis.poll_events()
#   vis.update_renderer()
#   if save_image:
#       vis.capture_screen_image("temp_%04d.jpg" % i)
#vis.destroy_window()


    #source_pc.transform(transform_GT)
    #o3d.visualization.draw_geometries([target_pc_cropped,source_pc],
    #                             zoom=0.8, front=[0, 0, 1],
    #                             lookat= init_pose [0:3] ,
    #                             up=[0, 1, 0],
     #                            point_show_normal=False)


    with open(path_txt, 'a') as f:
        f.write('\n\n' + '#'*30 + '\n\n')
        f.write('Source point cloud:\nTimestamp = %s\nNumber of points = %s\nCropped (Y/N)? = N\n\n' 
                %((str(timestamp).zfill(10)), str(len(source_pc.points))))
        f.write('Target point cloud: Moriyama map,\nNumber of points after cropping (RoI): %s\nCropped (Y/N)? = Y\n\n' %str(len(target_pc_cropped.points)))

    with open(path_txt, 'a') as f:
        f.write('GT Transformation matrix:\n%s ' %str(transform_GT) + "\n\n")  

    #o3d.visualization.draw([target_pcd_cropped, source_pcd])

    # Set parameter for localization algorithm

    #### ICP ###############################################################
    threshold = 0.5     # max. distance between two points to be seen as correct correspondence (=: inlier)
    trans_init = transform_GT.copy()

    #steps_init = [0.5, 0.5,0.5]   #step size in x,y,z direction
    lower_limits = [-2,-2,-2,-np.pi/4,-np.pi/4,-np.pi/4]    #x,y,z, alpha, beta, gamma
    upper_limits = [2,2,2, np.pi/4,np.pi/4,np.pi/4]       #x,y,z, alpha, beta, gamma

    number_eval_points = [17,17,17,10,10,10]
    
    axis2eval = [1,1,0,0,0,0]

    #def animation_callback(viz):
        #   global final_transform, vis_pc,pc_1

        #    source_pc.transform(transform_GT)


    with open(path_txt, 'a') as f:
        f.write('Parameter Set:\n\n')
        f.write('Threshold: %s\nLower limits: alpha = %s, beta = %s, gamma = %s\nUpper limits: x = %s, y = %s, z = %s\nNumber of evaluation points: x = %s, y = %s, z = %s\n\n' 
                %(str(threshold), lower_limits[0], lower_limits[1], lower_limits[2], upper_limits[0], upper_limits[1], upper_limits[2],
                  number_eval_points[0], number_eval_points[1], number_eval_points[2]))
    
    for k in range(3,3):
    
        eval_points = np.linspace(lower_limits[k],upper_limits[k],number_eval_points[k])
        print(eval_points)
    
        list_axes = [0,0,0]
        arr_euler = np.zeros((1,3))

        with open(path_txt, 'a') as f:
            f.write('\n\nEvaluation Steps for AXIS: ' + str(k))
            f.write('\n\n' +str(eval_points))
    
        for n in eval_points:
        
            list_axes [k] = n
            arr_euler [0][k] = n
            
        
            trans_init_updated = trans_init.copy()
            R_Euler_updated = R_Euler.copy()
            
            if bool_trans == True:
                trans_init_updated [k,3] = trans_init_updated [k,3] + n
                
            if bool_rot == True:
                R_Euler_updated = R_Euler_updated + arr_euler [0]
                
                r = R.from_euler('xyz',R_Euler_updated, degrees = False)
                R_matrix_updated = r.as_matrix()
                
                trans_init_updated [0:3,0:3] = R_matrix_updated
        
            print ("New initial transformation matrix is:\n", trans_init_updated, "\n ------------------- \n")
        
            #with open(path_txt, 'a') as f:
                #    f.write('\n\nEVAL POSITION:' + str(n))
                #    f.write('\n\nFitness or Overlapping area (inliers/target points); RMSE of inliers; Correspondence set size; Relative Transl. Error in m; Relative Rotation Error in deg\n')
        
            #o3d.core.Tensor(trans_init)

            print("ICP Registration for variation in direction %f; value of delta = %f" %(k,n))
            evaluation = o3d.pipelines.registration.evaluate_registration(
                            source_pc, target_pc_cropped, threshold, trans_init_updated)
            print(evaluation)
        
            l2 = np.sum(np.power((trans_init_updated[:,3]-transform_GT[:,3]),2))
            rse_transl = np.sqrt(l2)      #Euclidian distance between GT pose and estimated pose
        
            ### Get relative rotation error of predicted rotation vector compared to GT rotation vector - Angle in deg between those vectors
            final_rot_matrix = trans_init_updated [0:3,0:3]
            r = R.from_matrix(final_rot_matrix)
            R_vec_final = r.as_rotvec()


            unit_vector_1 = R_vec / np.linalg.norm(R_vec)
            unit_vector_2 = R_vec_final / np.linalg.norm(R_vec_final)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle = abs(np.arccos(dot_product))
            rso_deg_1 = np.rad2deg(angle)
        
            rso_deg_2 = np.rad2deg(np.arccos((np.trace(R_matrix * np.transpose(final_rot_matrix)) - 1)/2)) #from Literature
        
            #with open(path_txt, 'a') as f:
                #   f.write('\nInitial:\n')
                #    f.write(str(evaluation.fitness) +'; ' + str(evaluation.inlier_rmse) 
                #           + '; ' + str(len(evaluation.correspondence_set)) + '; '
                #           + str(rse_transl) + '; ' + str(rso_deg) + '\n\n')
         
            with open(path_csv, 'a') as f: 
                 f.write(str(ID) + '; ' + str(timestamp) + '; ' + str(int(k)) + '; ' + str(arr_euler[0][0]) + '; ' + str(arr_euler[0][1]) + '; ' + str(arr_euler[0][2]) + 
                         '; ' + str(evaluation.fitness) + '; ' + str(evaluation.inlier_rmse) 
                           + '; ' + str(float(len(evaluation.correspondence_set))) + '; '
                           + str(rse_transl) + '; ' + str(rso_deg_1) + '; ' + str(rso_deg_2) + '; ')

            #def registration_callback(curr_transform):
                #print(curr_transform)

            reg_p2p = o3d.pipelines.registration.registration_icp(
                                source_pc, target_pc_cropped, threshold, trans_init_updated,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 30))
            print("Results:\n", reg_p2p)
            print("Transformation is:")
            print(reg_p2p.transformation)
            print("\n ------------------------------------ \n")
            #For visualization transform the online scan with the predictet transformation matrix
        
            final_transform = reg_p2p.transformation
        
        
            ### Get relative translation error of predicted pose compared to GT - Euclidean distance (L2 norm)
            l2 = np.sum(np.power((final_transform[:,3]-transform_GT[:,3]),2))
            rse_transl = np.sqrt(l2)      #Euclidian distance between GT pose and estimated pose
        
            ### Get relative rotation error of predicted rotation vector compared to GT rotation vector - Angle in deg between those vectors
            final_rot_matrix = copy.deepcopy(final_transform [0:3,0:3])
            r = R.from_matrix(final_rot_matrix)
            R_vec_final = r.as_rotvec()


            unit_vector_1 = R_vec / np.linalg.norm(R_vec)
            unit_vector_2 = R_vec_final / np.linalg.norm(R_vec_final)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle = abs(np.arccos(dot_product))
            rso_deg_1 = np.rad2deg(angle)
        
            rso_deg_2 = np.rad2deg(np.arccos((np.trace(R_matrix * np.transpose(final_rot_matrix)) - 1)/2)) #from Literature

            #source_pc.transform(reg_p2p.transformation)
        
            #vis.add_geometry(source_pc.deepcopy.transform(reg_p2p.transformation))
            #vis.capture_screen_image('Test' + str(n) + '.jpg', do_render = True)
            #vis.remove_geometry(source_pc.deepcopy.transform(reg_p2p.transformation))

            #with open(path_txt, 'a') as f:
                #   f.write('Output:\n')
                #   f.write(str(reg_p2p.fitness) +'; ' + str(reg_p2p.inlier_rmse) +
                #           '; ' + str(len(reg_p2p.correspondence_set)) + '; '
                #           + str(rse_transl) + '; ' + str(rso_deg) + '\n\n')
                #f.write(str(reg_p2p.transformation) + '\n')

            with open(path_csv, 'a') as f: 
                f.write(str(reg_p2p.fitness) + '; ' + str(reg_p2p.inlier_rmse) 
                       + '; ' + str(float(len(reg_p2p.correspondence_set))) + '; '
                       + str(rse_transl) + '; ' + str(rso_deg_1) + '; ' + str(rso_deg_2) + '\n')




#reg_p2p = o3d.pipelines.registration.registration_icp(
#                        source_pc, target_pc_cropped, threshold, transform_GT,
#                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 30))  

#source_pc.transform(reg_p2p.transformation)      

#final_transform = reg_p2p.transformation

#l2 = np.sum(np.power((final_transform[:,3]-transform_GT[:,3]),2))
#rse_transl = np.sqrt(l2)      #Euclidian distance between GT pose and estimated pose

#final_rot_matrix = copy.deepcopy(final_transform [0:3, 0:3])  
#r = R.from_matrix(final_rot_matrix)
#R_vec_final = r.as_rotvec()
#R_vec_final

#unit_vector_1 = R_vec / np.linalg.norm(R_vec)
#unit_vector_2 = R_vec_final / np.linalg.norm(R_vec_final)
#dot_product = np.dot(unit_vector_1, unit_vector_2)
#angle = np.arccos(dot_product)
#angle_deg = np.rad2deg(angle)