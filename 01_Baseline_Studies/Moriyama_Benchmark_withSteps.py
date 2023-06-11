# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:50:17 2023

@author: Johanna
"""
import open3d as o3d
import copy
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os
from datetime import datetime
import time



def draw_registration_result(source, target, transformation):
    source_temp = source.clone()
    target_temp = target.clone()

    source_temp.transform(transformation)
    o3d.visualization.draw(
        [source_temp.to_legacy(),
        target_temp.to_legacy()])

#############################################
##### SET INPUTS    
bool_trans = False
bool_rot = True
bool_1D = True

ID = 'B011'

# ICP parameters
max_correspondence_distance = 0.35   # max. distance between two points to be seen as correct correspondence (=: inlier)
    
estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
   
#Convergence criteria
m_iters = 30
rel_rmse = 0.0001
rel_fit = 0.0001

voxel_size = -1   # -1 no downsampling
save_loss_log = True
    
lower_limits = [-2,-2,-2,-np.pi/4,-np.pi/4,-np.pi/4]    #x,y,z, alpha, beta, gamma
upper_limits = [2,2,2, np.pi/4,np.pi/4,np.pi/4]       #x,y,z, alpha, beta, gamma

number_eval_points = [17,17,17,17,17,17]
    
#axis2eval = [1,1,0,0,0,0] 

############################################
############################################


device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32

# Input: Define the path direction of the map to use, the specific point cloud from a defined timestamp and the path direction to the csv file containing the GT poses (from NDT localization in Autoware) of the Localizer
path_map = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\Moriyama_Map.pcd"
path_GT_csv = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\14_Local_Pose.csv"
path_to_file = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\03_Moriyama_Evaluation"

name_txt = str(ID) + '_RotBaselineICPMoriyama.txt'
path_txt = os.path.join(path_to_file, name_txt)
name_csv = str(ID) + '_RotBaselineICPMoriyama.csv'
path_csv = os.path.join(path_to_file,name_csv)
name_csv_iter = str(ID) + '_IterStepsBaselineICPMoriyama.csv'
path_csv_iter = os.path.join(path_to_file, name_csv_iter)

#Prepare CSV file
with open(path_csv, 'w') as f:
    f.write('ID; Timestamp GT Pose; Axis; Initial Transl x; Initial Transl y; Initial Transl z; Initial fitness; Initial RMSE Inliers; Initial Inlier correspondences; Initial Transl. Error [m]; Initial Rot. Error 1 [°]; Initial Rot. Error 2 [°]; Fitness; RMSE Inliers; Inlier correspondences; Transl. Error [m]; Rot. Error 1 [°]; Rot. Error 2 [°]; Number Iterations; Execut. Time [s]\n')

#Prepare txt file
with open(path_txt, 'w') as f:
    f.write('Evaluation of ICP algorithm for map matching on Moriyama dataset \n' + str(datetime.now()) + "\n\n")

#Prepare Iter step CSV file
with open(path_csv_iter, 'w') as f:
    f.write('ID; Timestamp; Axis; Init Error (Trans or Rot); Iteration Step Index; Fitness; Inlier RMSE [m]; t11; t12; t13; t14; t21; t22; t23; t24; t31; t32; t33; t34; t41; t42; t43; t44\n')


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
pc_map = o3d.t.io.read_point_cloud(path_map)  #map point cloud
target_pc = copy.deepcopy(pc_map)

path_pc = list_path_pc[0]
init_pose = arr_GT_poses[0]
timestamp = timestamps[0]

list_inter_results = []

for x in range(0,len(arr_GT_poses)):

    path_pc = list_path_pc[x]
    init_pose = arr_GT_poses[x]
    timestamp = timestamps[x]
    
    #Load online scans
    pc_scan = o3d.t.io.read_point_cloud(path_pc)  #online scan point cloud from above declared timestamp
    source_pc = copy.deepcopy(pc_scan)
    
    #Set color of pc scan
    scan_colors = np.zeros((len(source_pc.point.positions), 3))

    for i in range(0, len(scan_colors)):
        scan_colors[i,0] = 0.4  # RED
        scan_colors[i,1] = 0.0  # GREEN
        scan_colors[i,2] = 0.5  # BLUE

    source_pc.point.colors = o3d.core.Tensor(scan_colors, dtype, device)
    
    #Define initial transformation matrix from GT pose in the map coordinate system

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

    bbox = o3d.t.geometry.AxisAlignedBoundingBox(o3d.core.Tensor([min_bound[0][0], min_bound[1][0], min_bound[2][0]]),
                                                           o3d.core.Tensor([max_bound[0][0], max_bound[1][0], max_bound[2][0]]))
    target_pc = o3d.t.geometry.PointCloud.crop(target_pc,bbox)
    
    
    #source_pc.transform(transform_GT)
    #draw_registration_result(source_pc, target_pc, transform_GT)

    #Add the current information to the txt.-file
    with open(path_txt, 'a') as f:
        f.write('\n\n' + '#'*30 + '\n\n')
        f.write('Source point cloud:\nTimestamp = %s\nNumber of points = %s\nCropped (Y/N)? = N\n\n' 
                %((str(timestamp).zfill(10)), str(len(source_pc.point.positions))))
        f.write('Target point cloud: Moriyama map,\nNumber of points after cropping (RoI): %s\nCropped (Y/N)? = Y\n\n' %str(len(target_pc.point.positions)))

    with open(path_txt, 'a') as f:
        f.write('GT Transformation matrix:\n%s ' %str(transform_GT) + "\n\n")  
    
    iter_log = []
    # Set parameter for localization algorithm

    callback_after_iteration = lambda updated_result_dict : iter_log.append([updated_result_dict["iteration_index"].item(), 
                                                                             updated_result_dict["fitness"].item(),
                                                                             updated_result_dict["inlier_rmse"].item(),
                                                                             updated_result_dict["transformation"].numpy()])
        
    
    trans_init = transform_GT.copy()
 
    criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=rel_fit,
                                       relative_rmse=rel_rmse,
                                       max_iteration=m_iters)

    list_eval_points = []
    m = 0
    for entry in number_eval_points:
        list_eval_points.append(np.linspace(lower_limits[m],upper_limits[m],number_eval_points[m]))
        m += 1
    
    with open(path_txt, 'a') as f:
        f.write('Parameter Set:\n\n')
        f.write('Threshold: %s\n' %str(max_correspondence_distance))
        f.write ('Voxel Size: %s\n' %str(voxel_size))
        f.write ('Max. Iterations: %s\n' %str(m_iters))
        f.write('Relative RMSE for convergence: %s\n' %str(rel_rmse))
        f.write('Relative Fitness for convergence: %s\n' %str(rel_fit))
        f.write('\nTranslation boundaries:\nLower limits: x = %s, y = %s, z = %s\nUpper limits: x = %s, y = %s, z = %s\nNumber of evaluation points: x = %s, y = %s, z = %s\n\n' 
                %(lower_limits[0], lower_limits[1], lower_limits[2], upper_limits[0], upper_limits[1], upper_limits[2],
                  number_eval_points[0], number_eval_points[1], number_eval_points[2]))
        f.write('Rotation boundaries [rad]:\nLower limits: alpha = %s, beta = %s, gamma = %s\nUpper limits: alpha = %s, beta = %s, gamma = %s\nNumber of evaluation points: alpha = %s, beta = %s, gamma = %s\n\n' 
                %(lower_limits[3], lower_limits[4], lower_limits[5], upper_limits[3], upper_limits[4], upper_limits[5],
                  number_eval_points[3], number_eval_points[4], number_eval_points[5]))
    
    if bool_1D == True:
        
        if bool_trans == True:
            k0 = 0
        elif bool_rot == True:
            k0 = 3
        else: 
            k0 = 0
            print ('WARNING! Setting of Translation or Rotation Evaluation not correct.')
        
        for k in range(k0,k0+3):
    
            eval_points = list_eval_points[k]

            list_axes = np.zeros((1,3))
            arr_euler = np.zeros((1,3))

    
            for n in eval_points:
            
                trans_init_updated = trans_init.copy()
                R_Euler_updated = R_Euler.copy()
            
                if bool_trans == True:
                    list_axes [0][k] = n
                    trans_init_updated [k,3] = trans_init_updated [k,3] + n
                    csv_output = list_axes[0]
                
                if bool_rot == True:
                    arr_euler [0][k-3] = n
                    R_Euler_updated = R_Euler_updated + arr_euler [0]
                
                    r = R.from_euler('xyz',R_Euler_updated, degrees = False)
                    R_matrix_updated = r.as_matrix()
                    trans_init_updated [0:3,0:3] = R_matrix_updated
                
                    csv_output = arr_euler[0]
        
                #print ("New initial transformation matrix is:\n", trans_init_updated, "\n ------------------- \n")
        
     

                print("ICP Registration for variation in direction %f; value of delta = %f" %(k,n))
                
                evaluation = o3d.t.pipelines.registration.evaluate_registration(
                            source_pc, target_pc, max_correspondence_distance, trans_init_updated)

                # Get relative translation error
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
            
                rso_deg_2 = abs(R_Euler[2]-R_Euler_updated[2]) * (180/np.pi)  # Error of the yaw angle in deg
                #rso_deg_2 = np.rad2deg(np.arccos((np.trace(R_matrix * np.transpose(final_rot_matrix)) - 1)/2)) #from Literature
                
                arr_correspondences = evaluation.correspondences_.numpy()
                arr_corr = arr_correspondences[arr_correspondences != -1]
         
                with open(path_csv, 'a') as f: 
                    f.write(str(ID) + '; ' + str(timestamp) + '; ' + str(int(k)) + '; ' + str(csv_output[0]) + '; ' + str(csv_output[1]) + '; ' + str(csv_output[2]) + 
                         '; ' + str(evaluation.fitness) + '; ' + str(evaluation.inlier_rmse) 
                           + '; ' + str(float(len(arr_corr))) + '; '
                           + str(rse_transl) + '; ' + str(rso_deg_1) + '; ' + str(rso_deg_2) + '; ')
    

                s = time.time()

                registration_icp = o3d.t.pipelines.registration.icp(source_pc, target_pc, max_correspondence_distance,
                            trans_init_updated, estimation, criteria,
                            voxel_size, callback_after_iteration)

                icp_time = time.time() - s
            
                final_transform = registration_icp.transformation.numpy()
        
        
                ### Get relative translation error of predicted pose compared to GT - Euclidean distance (L2 norm)
                l2 = np.sum(np.power((final_transform[:,3]-transform_GT[:,3]),2))
                rse_transl = np.sqrt(l2)      #Euclidian distance between GT pose and estimated pose
        
                ### Get relative rotation error of predicted rotation vector compared to GT rotation vector - Angle in deg between those vectors
                final_rot_matrix = copy.deepcopy(final_transform [0:3,0:3])
                r = R.from_matrix(final_rot_matrix)
                R_vec_final = r.as_rotvec()
            
                R_Euler_final = r.as_euler('xyz')


                unit_vector_1 = R_vec / np.linalg.norm(R_vec)
                unit_vector_2 = R_vec_final / np.linalg.norm(R_vec_final)
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                angle = abs(np.arccos(dot_product))
                rso_deg_1 = np.rad2deg(angle)
            
                rso_deg_2 = abs(R_Euler[2]-R_Euler_final[2]) * (180/np.pi)  # Error of the yaw angle in deg
                #rso_deg_2 = np.rad2deg(np.arccos((np.trace(R_matrix * np.transpose(final_rot_matrix)) - 1)/2)) #from Literature
                
                arr_correspondences = registration_icp.correspondences_.numpy()
                arr_corr = arr_correspondences[arr_correspondences != -1]
                
                number_iterations = len(iter_log)

                print("Time taken by ICP: ", icp_time)
                print("Inlier Fitness: ", registration_icp.fitness)
                print("Inlier RMSE: ", registration_icp.inlier_rmse)
            
                with open(path_csv, 'a') as f: 
                    f.write(str(registration_icp.fitness) + '; ' + str(registration_icp.inlier_rmse) 
                            + '; ' + str(float(len(arr_corr))) + '; '
                            + str(rse_transl) + '; ' + str(rso_deg_1) + '; ' + str(rso_deg_2)
                            + '; ' + str(number_iterations) + '; ' + str(icp_time) + '\n')
            
            # iter_log = copy.deepcopy(iter_log)
                for entry in iter_log:   
                    with open(path_csv_iter, 'a') as f:
                        f.write(str(ID) + '; ' + str(timestamp) + '; ' +
                                str(k) + '; ' + str(n) + '; ' + str(entry[0]) + '; '
                                + str(entry[1]) + '; ' + str(entry[2]) + '; '
                                + str(entry[3][0][0]) + '; ' + str(entry[3][0][1]) + '; ' + str(entry[3][0][2]) + '; ' + str(entry[3][0][3]) + '; '
                                + str(entry[3][1][0]) + '; ' + str(entry[3][1][1]) + '; ' + str(entry[3][1][2]) + '; ' + str(entry[3][1][3]) + '; '
                                + str(entry[3][2][0]) + '; ' + str(entry[3][2][1]) + '; ' + str(entry[3][2][2]) + '; ' + str(entry[3][2][3]) + '; '
                                + str(entry[3][3][0]) + '; ' + str(entry[3][3][1]) + '; ' + str(entry[3][3][2]) + '; ' + str(entry[3][3][3]) + '\n')
        
                #list_iter_steps.append([iter_log])

                iter_log = []
    
    #list_inter_results.append(list_iter_steps)
    #list_iter_steps = []
       

#draw_registration_result(source_pc, target_pc, registration_icp.transformation)

