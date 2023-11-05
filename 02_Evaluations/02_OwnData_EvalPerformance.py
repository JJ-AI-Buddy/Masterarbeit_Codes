# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:45:48 2023

@author: Johanna Hable

Notes:
    - This file executes the P2Point- and P2Plane-ICP algorithm for an own dataset; similar to the python script of the baseline studies ending in "withStepsCombined"
    - Only difference: The initial pose will be randomly perturbated in x,y and yaw at the same time
    - The limits for the delta values that will be added to the GT pose are set with the lists 'lower_limits' and 'upper_limits'
    - Set the 'ID' value and the 'estimation' and also the desired 'num_runs' which will tell how often the initial pose should be randomly set per selected point cloud
    - Also set the other input values, like the paths and especially the name of the output files before running this script
    - The results of this evaluation script will later be used as labes for training a machine learning model; especially the translation/rotation errors as well as the number of iterations
"""

import open3d as o3d
import copy
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os
from datetime import datetime
import time
import random
import glob


def quat2transform (pose):
    #pose N-tuple with 7 entries (x,y,z,orient x, orient y, orient z, orient w)
    
    r = R.from_quat(pose[3:7])
    R_matrix = r.as_matrix()
    R_vec = r.as_rotvec()
    R_Euler = r.as_euler('xyz')

    t_vec = np.zeros((3,1))

    for k in range(0,len(t_vec)):
        t_vec[k] = pose[k]

    tranform_raw = np.hstack((R_matrix, t_vec)) # 3x3 matrix for rotation and translation
    transform = np.vstack((tranform_raw,np.array([0,0,0,1]))) # 4x4 homography transformation matrix
    
    return transform, R_Euler, R_vec, t_vec

def evalMetrics (tf_GT, R_vec_GT, R_Euler_GT, tf):
    ### Get relative translation error of predicted pose compared to GT - Euclidean distance (L2 norm)
    l2 = np.sum(np.power((tf[:,3]-tf_GT[:,3]),2))
    rse_transl = np.sqrt(l2)      #Euclidian distance between GT pose and estimated pose

    ### Get relative rotation error of predicted rotation vector compared to GT rotation vector - Angle in deg between those vectors
    final_rot_matrix = copy.deepcopy(tf [0:3,0:3])
    r = R.from_matrix(final_rot_matrix)
    R_vec_final = r.as_rotvec()

    R_Euler_final = r.as_euler('xyz')


    unit_vector_1 = R_vec_GT / np.linalg.norm(R_vec_GT)
    unit_vector_2 = R_vec_final / np.linalg.norm(R_vec_final)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = abs(np.arccos(dot_product))
    rso_deg_1 = np.rad2deg(angle)

    rso_deg_2 = abs(R_Euler_GT[2]-R_Euler_final[2]) * (180/np.pi)  # Error of the yaw angle in deg
    #rso_deg_2 = np.rad2deg(np.arccos((np.trace(R_matrix * np.transpose(final_rot_matrix)) - 1)/2)) #from Literature
    
    return rse_transl, rso_deg_1, rso_deg_2



ID = 'V1'        # Set ID like specified in the documentation

# ICP parameters
max_correspondence_distance = 0.5   # max. distance between two points to be seen as correct correspondence (=: inlier)
    
#estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
   
#Convergence criteria
m_iters = 30
rel_rmse = 0.0001
rel_fit = 0.0001

voxel_size = -1   # -1 no downsampling
save_loss_log = True
validation_set = True
    
lower_limits = [-2,-2,-np.pi/8]    #x,y,yaw
upper_limits = [2,2, np.pi/8]       #x,y,yaw

num_runs = 5

############################################
############################################


device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32

# Input: Define the path direction of the map to use, the specific point cloud from a defined timestamp and the path direction to the csv file containing the GT poses (from NDT localization in Autoware) of the Localizer
path_map = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Evaluations\03_Route_2_Data\FinalMap_Route2.pcd"
path_GT_csv = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data\Dataset_Validation\Suburban\GT_Poses_Suburban.csv"
path_to_file = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data\Dataset_Validation\Suburban"

path_to_scans = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data\Dataset_Validation\Suburban\prepro\*.pcd"

name_csv = str(ID) + '_P2Point-ICP_Suburban.csv'
path_csv = os.path.join(path_to_file,name_csv)
name_csv_iter = str(ID) + '_Iter_P2Point-ICP_Suburban.csv'
path_csv_iter = os.path.join(path_to_file, name_csv_iter)

#Prepare CSV file
with open(path_csv, 'w') as f:
    f.write('ID;Timestamp GT Pose;Scan Nr.;Run;x delta [m];y delta [m];yaw delta [rad];Initial fitness;Initial RMSE Inliers;Initial Inlier correspondences;Initial Transl. Error [m];Initial Rot. Error 1 [°];Initial Rot. Error 2 [°];Fitness;RMSE Inliers;Inlier correspondences;Transl. Error [m];Rot. Error 1 [°];Rot. Error 2 [°];Number Iterations;Execut. Time [s]\n')


#Prepare Iter step CSV file
with open(path_csv_iter, 'w') as f:
    f.write('ID;Timestamp;Scan Nr.;Run;Init Perturbation (x,y,yaw);Iteration Step Index;Fitness;Inlier RMSE [m];t11;t12;t13;t14;t21;t22;t23;t24;t31;t32;t33;t34;t41;t42;t43;t44\n')



#Load GT poses from csv
df_GT = pd.read_csv(path_GT_csv, delimiter = ';', header = 0, index_col = False)

if validation_set == True:
    path_map_1 = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Evaluations\02_Route_1_Data\FinalMap_Route1.pcd"
    path_map_2 = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Evaluations\03_Route_2_Data\FinalMap_Route2.pcd"
    path_map_3 = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Evaluations\04_Route_3_Data\FinalMap_Route3.pcd"
    path_map_4 = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Evaluations\05_Route_4_Data\FinalMap_Route4.pcd"
    
    list_map = [path_map_1,path_map_2,path_map_3,path_map_4]

#Save GT position (x,y,z) and orientation as quaternions (x,y,z,w) in numpy array
arr_GT_poses = df_GT.iloc[:,2::].to_numpy(dtype = 'float')

arr_GT_corrected = np.zeros((len(arr_GT_poses), 7))

for i in range(0,len(arr_GT_corrected[0])):
    arr_GT_corrected[:,i]=arr_GT_poses[:,i] + arr_GT_poses[:,i + 7]


list_path_pc = df_GT["pc.timestamp.path"].values.tolist()

if validation_set == True:
    list_idx_map = df_GT["map"].values.tolist()
else:
    #Load map
    pc_map = o3d.t.io.read_point_cloud(path_map)  #map point cloud
    timestamps = df_GT["%time"].values.tolist()

x = 0

for i in glob.glob(str(path_to_scans)):
    
    if validation_set == True:
        path_map = list_map[list_idx_map[x]]
        
        #Load map
        pc_map = o3d.t.io.read_point_cloud(path_map)  #map point cloud
    

    target_pc = copy.deepcopy(pc_map)

    #pc_file = os.path.basename(str(list_path_pc[x]))
    path_pc = os.path.join(path_to_scans,i)
    init_pose = arr_GT_corrected[x]
    
    if validation_set == False:
        timestamp = timestamps[x]
    else:
        timestamp = x
    
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
    
    transform_GT, R_Euler, R_vec, t_vec = quat2transform(init_pose)
    
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
    
    target_pc.estimate_normals(max_nn = 30) #kNN by default = 30; radius for hybrid search optional
    
    for k in range(0,num_runs):

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


        x_delta = random.uniform(lower_limits[0], upper_limits[0])
        y_delta = random.uniform(lower_limits[1], upper_limits[1])
        yaw_delta = random.uniform(lower_limits[2], upper_limits[2])
    
        list_axes = np.zeros((1, 3))
        arr_euler = np.zeros((1, 3))
    
        trans_init_updated = trans_init.copy()
        R_Euler_updated = R_Euler.copy()
    
        list_axes [0][0] = x_delta
        list_axes [0][1] = y_delta
        arr_euler [0][2] = yaw_delta
    
        trans_init_updated [0:3,3] += list_axes[0]
        R_Euler_updated += arr_euler[0]
    
        r = R.from_euler('xyz', R_Euler_updated, degrees=False)
        R_matrix_updated = r.as_matrix()
        trans_init_updated[0:3, 0:3] = R_matrix_updated
    
        print("ICP Registration of Scan Nr. %i / Run %i with initial pose perturbation:\nx: %f, y: %f, yaw: %f\n" %(x,k, x_delta,y_delta,yaw_delta))
    
        evaluation = o3d.t.pipelines.registration.evaluate_registration(
            source_pc, target_pc, max_correspondence_distance, trans_init_updated)
    
        if evaluation != None:

            arr_correspondences = evaluation.correspondences_.numpy()
            arr_corr = arr_correspondences[arr_correspondences != -1]

            init_fitness = evaluation.fitness
            init_rmse = evaluation.inlier_rmse

        else:
            arr_corr = []

            init_fitness = np.nan
            init_rmse = np.nan
    
    
        # Calculate evaluation metrics
        rse_transl, rso_deg_1, rso_deg_2 = evalMetrics(
            transform_GT, R_vec, R_Euler, trans_init_updated)
    
    
        with open(path_csv, 'a') as f: 
            f.write(str(ID) + ';' + str(timestamp) + ';' + str(int(x)) + ';' + str(k) + ';' + str(x_delta) + ';' + str(y_delta) + ';' + str(yaw_delta) + 
             ';' + str(init_fitness) + ';' + str(init_rmse) 
               + ';' + str(float(len(arr_corr))) + ';'
               + str(rse_transl) + ';' + str(rso_deg_1) + ';' + str(rso_deg_2) + ';')

        time.sleep(0.1)
    
        ############ REGISTRATION ####################
        s = time.time()

        registration_icp = o3d.t.pipelines.registration.icp(source_pc, target_pc, max_correspondence_distance,
                trans_init_updated, estimation, criteria,
                voxel_size, callback_after_iteration)

        icp_time = time.time() - s
    
        # Check registration results
        if registration_icp != None: 

            final_transform = registration_icp.transformation.numpy()
        
            arr_correspondences = registration_icp.correspondences_.numpy()
            arr_corr = arr_correspondences[arr_correspondences != -1]
        
            number_iterations = len(iter_log)
        
            fitness = registration_icp.fitness
            rmse = registration_icp.inlier_rmse
        
        else: 
            final_transform = transform_GT.copy()
            arr_corr = []
            number_iterations = np.nan
            fitness = np.nan
            rmse = np.nan
        
            iter_log = []

        # Calculate evaluation metrics
        rse_transl, rso_deg_1, rso_deg_2 = evalMetrics(transform_GT, R_vec, R_Euler, final_transform)


        print("Time taken by ICP: ", icp_time)
        print("Inlier Fitness: ", fitness)
        print("Inlier RMSE: ", rmse)
        print("\n\n")
    
        with open(path_csv, 'a') as f: 
            f.write(str(fitness) + ';' + str(rmse) 
                + ';' + str(float(len(arr_corr))) + ';'
                + str(rse_transl) + ';' + str(rso_deg_1) + ';' + str(rso_deg_2)
                + ';' + str(number_iterations) + ';' + str(icp_time) 
                 + '\n')
        
        time.sleep(0.1)
        
        if k == 0:
    
            if iter_log != []:
                for entry in iter_log:   
                    with open(path_csv_iter, 'a') as f:
                        f.write(str(ID) + ';' + str(timestamp) + ';' +
                        str(x) + ';' + str(k) + ';' + str(x_delta) + ',' + str(y_delta) + ',' + str(yaw_delta) + ';' + str(entry[0]) + ';'
                        + str(entry[1]) + ';' + str(entry[2]) + ';'
                        + str(entry[3][0][0]) + ';' + str(entry[3][0][1]) + ';' + str(entry[3][0][2]) + ';' + str(entry[3][0][3]) + ';'
                        + str(entry[3][1][0]) + ';' + str(entry[3][1][1]) + ';' + str(entry[3][1][2]) + ';' + str(entry[3][1][3]) + ';'
                        + str(entry[3][2][0]) + ';' + str(entry[3][2][1]) + ';' + str(entry[3][2][2]) + ';' + str(entry[3][2][3]) + ';'
                        + str(entry[3][3][0]) + ';' + str(entry[3][3][1]) + ';' + str(entry[3][3][2]) + ';' + str(entry[3][3][3]) + '\n')
    
            else:
                with open(path_csv_iter, 'a') as f:
                    f.write(str(ID) + ';' + str(timestamp) + ';' +
                   str(x) + ';' + str(k) + ';' + str(x_delta) + ',' + str(y_delta) + ',' + str(yaw_delta) + ';' + str(np.nan) + ';'
                   + str(np.nan) + ';' + str(np.nan) + ';'
                   + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + ';'
                   + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + ';'
                   + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + ';'
                   + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + '\n')
           

        iter_log = []
    
    x += 1



