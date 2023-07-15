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
    global t_vec
    source_temp = source.clone()
    target_temp = target.clone()
    
    mesh_map = o3d.t.geometry.TriangleMesh.create_coordinate_frame(
        size=20, origin=[t_vec.T[0][0]-2, t_vec.T[0][1]-45, t_vec.T[0][2]]) #[t_vec.T[0][0]-20, t_vec.T[0][1]-20, t_vec.T[0][2]]

    mesh_lidar = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 15, origin = [0,0,0])
    mesh_lidar.transform(transformation)

    source_temp.transform(transformation)
    o3d.visualization.draw(
        [source_temp.to_legacy(),
        target_temp.to_legacy(),
        mesh_map,
        mesh_lidar])

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
   

#############################################
##### SET INPUTS    
bool_trans = False     # if True then bool_rot has to be False

if bool_trans == False:
    bool_rot = True      # if True then bool_trans has to be False
else: bool_rot = False

bool_1D = True      #if True only one of the above also has to be true (ATTENTION! both at the same time cannot be true)


bool_2D = False        # if True you also have to check the 'axis2Deval' variable; the first two entries denote the index of the two axes to evaluate at the same time 0 = x-axis, 1 = y-axis, 2 = z-axis
bool_2D_Yaw = False     # if True you also have to check the 'axis2Deval' variable; the first two entries denote the index of the two axes to evaluate at the same time 0 = x-axis, 1 = y-axis, 2 = z-axis
                        # The third entry encodes the rotation axis; if we want the yaw-angle (z-axis) we have to set the last entry to 5

ID = 'C110'        # Set ID like specified in the documentation

# ICP parameters
max_correspondence_distance = 0.5   # max. distance between two points to be seen as correct correspondence (=: inlier)
    
#estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
   
#Convergence criteria
m_iters = 30
rel_rmse = 0.0001
rel_fit = 0.0001

voxel_size = -1   # -1 no downsampling
save_loss_log = True
    
lower_limits = [-50,-50,0,-np.pi/2,-np.pi/2,-np.pi/2]    #x,y,z, alpha, beta, gamma
upper_limits = [50,50,0, np.pi/2,np.pi/2,np.pi/2]       #x,y,z, alpha, beta, gamma

number_eval_points = [21,21,1,10,10,10]    #[17,17,17,17,17,17] for 1D, [9,9,9,9,9,9] for 2D
    
axis2Deval = [0,1,5] 

############################################
############################################


device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32

# Input: Define the path direction of the map to use, the specific point cloud from a defined timestamp and the path direction to the csv file containing the GT poses (from NDT localization in Autoware) of the Localizer
path_map = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\Moriyama_Map.pcd"
path_GT_csv = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\14_Local_Pose.csv"
path_GNSS_csv = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\13_GNSS_pose.csv"
path_to_file = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\03_Moriyama_Evaluation"

name_txt = str(ID) + '_BigRotBaselineICPMoriyama.txt'
path_txt = os.path.join(path_to_file, name_txt)
name_csv = str(ID) + '_BigRotBaselineICPMoriyama.csv'
path_csv = os.path.join(path_to_file,name_csv)
name_csv_iter = str(ID) + '_BigRotStepsBaselineICPMoriyama.csv'
path_csv_iter = os.path.join(path_to_file, name_csv_iter)

#Prepare CSV file
with open(path_csv, 'w') as f:
    f.write('ID;Timestamp GT Pose;Axis;Initial Transl x;Initial Transl y;Initial Transl z;Initial fitness;Initial RMSE Inliers;Initial Inlier correspondences;Initial Transl. Error [m];Initial Rot. Error 1 [°];Initial Rot. Error 2 [°];Fitness;RMSE Inliers;Inlier correspondences;Transl. Error [m];Rot. Error 1 [°];Rot. Error 2 [°];Number Iterations;Execut. Time [s];GNSS Transl. Error[m];GNSS Rot. Error 1 [°];GNSS Rot. Error 2 [°]\n')

#Prepare txt file
with open(path_txt, 'w') as f:
    f.write('Evaluation of ICP algorithm for map matching on Moriyama dataset \n' + str(datetime.now()) + "\n\n")

if bool_1D == True:
    #Prepare Iter step CSV file
    with open(path_csv_iter, 'w') as f:
        f.write('ID;Timestamp;Axis;Init Error (Trans or Rot);Iteration Step Index;Fitness;Inlier RMSE [m];t11;t12;t13;t14;t21;t22;t23;t24;t31;t32;t33;t34;t41;t42;t43;t44\n')


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

#Load GNSS poses from csv
df_GNSS = pd.read_csv(path_GNSS_csv, delimiter = ',', header = 0)

arr_GNSS_poses = np.zeros((5,7))
i = 0
for i in range(0, len(timestamps)):
    df_new = df_GNSS ["%time"]
    df_new = abs(df_new - timestamps[i])
    #print(df_new.idxmin())
    arr_GNSS_poses[i,:] = np.asarray(df_GNSS.iloc[df_new.idxmin(), 4:11])

    
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

#list_inter_results = []

#x = 3
#x = 3

for x in range(0,len(arr_GT_poses)):

    path_pc = list_path_pc[x]
    init_pose = arr_GT_poses[x]
    GNSS_pose = arr_GNSS_poses[x]
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
    transform_GT, R_Euler, R_vec, t_vec = quat2transform(init_pose)
    
    #Define GNSS transformation matrix in map coordinate system
    transform_GNSS, R_Euler_GNSS, R_vec_GNSS, t_vec_GNSS = quat2transform(GNSS_pose)
    
    
    # Calculate errors of GNSS pose
    rse_transl_GNSS, rso_deg_1_GNSS, rso_deg_2_GNSS = evalMetrics(transform_GT, R_vec, R_Euler, transform_GNSS)


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
    
    target_pc.estimate_normals() #kNN by default = 30; radius for hybrid search optional
    
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
        f.write('Voxel Size: %s\n' %str(voxel_size))
        f.write('Max. Iterations: %s\n' %str(m_iters))
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
                rse_transl, rso_deg_1, rso_deg_2 = evalMetrics(transform_GT, R_vec, R_Euler, trans_init_updated)
                
                time.sleep(0.5)

                with open(path_csv, 'a') as f: 
                    f.write(str(ID) + ';' + str(timestamp) + ';' + str(int(k)) + ';' + str(csv_output[0]) + ';' + str(csv_output[1]) + ';' + str(csv_output[2]) + 
                         ';' + str(init_fitness) + ';' + str(init_rmse) 
                           + ';' + str(float(len(arr_corr))) + ';'
                           + str(rse_transl) + ';' + str(rso_deg_1) + ';' + str(rso_deg_2) + ';')
    
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
                
                time.sleep(0.5)
            
                with open(path_csv, 'a') as f: 
                    f.write(str(fitness) + ';' + str(rmse) 
                            + ';' + str(float(len(arr_corr))) + ';'
                            + str(rse_transl) + ';' + str(rso_deg_1) + ';' + str(rso_deg_2)
                            + ';' + str(number_iterations) + ';' + str(icp_time) 
                            + ';' + str(rse_transl_GNSS) + ';' + str(rso_deg_1_GNSS) + ';' + str(rso_deg_2_GNSS) + '\n')
            
                if iter_log != []:
                    for entry in iter_log:   
                        with open(path_csv_iter, 'a') as f:
                            f.write(str(ID) + ';' + str(timestamp) + ';' +
                                str(k) + ';' + str(n) + ';' + str(entry[0]) + ';'
                                + str(entry[1]) + ';' + str(entry[2]) + ';'
                                + str(entry[3][0][0]) + ';' + str(entry[3][0][1]) + ';' + str(entry[3][0][2]) + ';' + str(entry[3][0][3]) + ';'
                                + str(entry[3][1][0]) + ';' + str(entry[3][1][1]) + ';' + str(entry[3][1][2]) + ';' + str(entry[3][1][3]) + ';'
                                + str(entry[3][2][0]) + ';' + str(entry[3][2][1]) + ';' + str(entry[3][2][2]) + ';' + str(entry[3][2][3]) + ';'
                                + str(entry[3][3][0]) + ';' + str(entry[3][3][1]) + ';' + str(entry[3][3][2]) + ';' + str(entry[3][3][3]) + '\n')
        
                else:
                   with open(path_csv_iter, 'a') as f:
                       f.write(str(ID) + ';' + str(timestamp) + ';' +
                           str(k) + ';' + str(n) + ';' + str(np.nan) + ';'
                           + str(np.nan) + ';' + str(np.nan) + ';'
                           + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + ';'
                           + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + ';'
                           + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + ';'
                           + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + ';' + str(np.nan) + '\n')
                   

                iter_log = []
                
    
    if bool_2D == True:
        list_axes = np.zeros((1,3))
        R_Euler_updated = R_Euler.copy()
        
        for el1 in list_eval_points[axis2Deval[0]]:
            #trans_init_updated = trans_init.copy()
            #R_Euler_updated = R_Euler.copy()
            
            list_axes [0][axis2Deval[0]] = el1
            
            for el2 in list_eval_points[axis2Deval[1]]:
                trans_init_updated = trans_init.copy()
                list_axes [0][axis2Deval[1]] = el2
                
                trans_init_updated [0,3] = trans_init_updated [0,3] + list_axes [0][0]
                trans_init_updated [1,3] = trans_init_updated [1,3] + list_axes [0][1]
                trans_init_updated [2,3] = trans_init_updated [2,3] + list_axes [0][2]
                
                csv_output = list_axes[0]
                
                
                
                print("ICP Registration for variation in directions %f and %f; value of deltas = %f and %f" %(axis2Deval[0],axis2Deval[1],
                                                                                                              list_axes [0][axis2Deval[0]],
                                                                                                              list_axes [0][axis2Deval[1]]))
                ######### EVALUATION OF INITIAL ALIGNMENT ############
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
                rse_transl, rso_deg_1, rso_deg_2 = evalMetrics(transform_GT, R_vec, R_Euler, trans_init_updated)

                time.sleep(0.5)
                with open(path_csv, 'a') as f: 
                    f.write(str(ID) + ';' + str(timestamp) + ';' + str(axis2Deval[0]) + ',' + str(axis2Deval[1]) + ';' 
                            + str(csv_output[0]) + ';' + str(csv_output[1]) + ';' + str(csv_output[2]) + 
                         ';' + str(init_fitness) + ';' + str(init_rmse) 
                           + ';' + str(float(len(arr_corr))) + ';'
                           + str(rse_transl) + ';' + str(rso_deg_1) + ';' + str(rso_deg_2) + ';')
    
                ####### REGISTRATION ##############################
                s = time.time()

                registration_icp = o3d.t.pipelines.registration.icp(source_pc, target_pc, max_correspondence_distance,
                            trans_init_updated, estimation, criteria,
                            voxel_size, callback_after_iteration)

                icp_time = time.time() - s
                
                #Check if Registration was successful
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
        
        
                # Calculate evaluation metrics
                rse_transl, rso_deg_1, rso_deg_2 = evalMetrics(transform_GT, R_vec, R_Euler, final_transform)

                print("Time taken by ICP: ", icp_time)
                print("Inlier Fitness: ", fitness)
                print("Inlier RMSE: ", rmse)
            
                time.sleep(0.5)
                with open(path_csv, 'a') as f: 
                    f.write(str(fitness) + ';' + str(rmse) 
                            + ';' + str(float(len(arr_corr))) + ';'
                            + str(rse_transl) + ';' + str(rso_deg_1) + ';' + str(rso_deg_2)
                            + ';' + str(number_iterations) + ';' + str(icp_time) 
                            + ';' + str(rse_transl_GNSS) + ';' + str(rso_deg_1_GNSS) + ';' + str(rso_deg_2_GNSS) +'\n')
            
            # No Log-File in this case because amount of data simply too much!
            
            # iter_log = copy.deepcopy(iter_log)
            #    for entry in iter_log:   
             #   with open(path_csv_iter, 'a') as f:
             #           f.write('No log because too much data.')
             #           f.write(str(ID) + '; ' + str(timestamp) + '; ' +
             #                   str(axis2Deval[0]) + ',' + str(axis2Deval[1]) + '; ' + 
             #                   str(list_axes [0][axis2Deval[0]]) + ',' + str(list_axes[0][axis2Deval[1]]) + '; ' + str(entry[0]) + '; '
             #                   + str(entry[1]) + '; ' + str(entry[2]) + '; '
             #                   + str(entry[3][0][0]) + '; ' + str(entry[3][0][1]) + '; ' + str(entry[3][0][2]) + '; ' + str(entry[3][0][3]) + '; '
              #                  + str(entry[3][1][0]) + '; ' + str(entry[3][1][1]) + '; ' + str(entry[3][1][2]) + '; ' + str(entry[3][1][3]) + '; '
             #                   + str(entry[3][2][0]) + '; ' + str(entry[3][2][1]) + '; ' + str(entry[3][2][2]) + '; ' + str(entry[3][2][3]) + '; '
             #                   + str(entry[3][3][0]) + '; ' + str(entry[3][3][1]) + '; ' + str(entry[3][3][2]) + '; ' + str(entry[3][3][3]) + '\n')
             
                iter_log = []
            
    if bool_2D_Yaw == True:
        list_axes = np.zeros((1,3))
        
        for el1 in list_eval_points[axis2Deval[0]]:
           # trans_init_updated = trans_init.copy()
           # R_Euler_updated = R_Euler.copy()
            
            list_axes [0][axis2Deval[0]] = el1
            
            for el2 in list_eval_points[axis2Deval[1]]:
                trans_init_updated = trans_init.copy()
                list_axes [0][axis2Deval[1]] = el2
                
                trans_init_updated [0,3] = trans_init_updated [0,3] + list_axes [0][0]
                trans_init_updated [1,3] = trans_init_updated [1,3] + list_axes [0][1]
                trans_init_updated [2,3] = trans_init_updated [2,3] + list_axes [0][2]
                
                csv_output = list_axes[0]
                
                for el3 in list_eval_points[axis2Deval[2]]:
                    R_Euler_updated = R_Euler.copy()
                    R_Euler_updated [2] = R_Euler_updated [2] + el3
                    
                    r = R.from_euler('xyz',R_Euler_updated, degrees = False)
                    R_matrix_updated = r.as_matrix()
                    trans_init_updated [0:3,0:3] = R_matrix_updated
                
                
                
                    print("ICP Registration for variation in directions %f and %f; value of deltas = %f and %f; yaw angle offset = %f" %(axis2Deval[0],axis2Deval[1],
                                                                                                              list_axes [0][axis2Deval[0]],
                                                                                                              list_axes [0][axis2Deval[1]],el3))
                    
                    ############ EVALUATION OF INITIAL ALIGNMENT ###############################
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
                    rse_transl, rso_deg_1, rso_deg_2 = evalMetrics(transform_GT, R_vec, R_Euler, trans_init_updated)

                    time.sleep(0.5)
                    with open(path_csv, 'a') as f: 
                        f.write(str(ID) + ';' + str(timestamp) + ';' + str(axis2Deval[0]) + ',' + str(axis2Deval[1]) + 
                            ';' + str(csv_output[0]) + ';' + str(csv_output[1]) + ';' + str(el3) + 
                         ';' + str(init_fitness) + ';' + str(init_rmse) 
                           + ';' + str(float(len(arr_corr))) + ';'
                           + str(rse_transl) + ';' + str(rso_deg_1) + ';' + str(rso_deg_2) + ';')
    
                    ############# REGISTRATION ###########################
                    s = time.time()

                    registration_icp = o3d.t.pipelines.registration.icp(source_pc, target_pc, max_correspondence_distance,
                            trans_init_updated, estimation, criteria,
                            voxel_size, callback_after_iteration)

                    icp_time = time.time() - s
                    
                    #Check if Registration was successful
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
            
        
                    # Calculate evaluation metrics
                    rse_transl, rso_deg_1, rso_deg_2 = evalMetrics(transform_GT, R_vec, R_Euler, final_transform)
 
                    print("Time taken by ICP: ", icp_time)
                    print("Inlier Fitness: ", fitness)
                    print("Inlier RMSE: ", rmse)
            
                    time.sleep(0.5)
                    with open(path_csv, 'a') as f: 
                        f.write(str(fitness) + ';' + str(rmse) 
                            + ';' + str(float(len(arr_corr))) + ';'
                            + str(rse_transl) + '; ' + str(rso_deg_1) + ';' + str(rso_deg_2)
                            + ';' + str(number_iterations) + ';' + str(icp_time) 
                            + ';' + str(rse_transl_GNSS) + ';' + str(rso_deg_1_GNSS) + ';' + str(rso_deg_2_GNSS) +'\n')
                        
                # No Log-File in this case because amount of data simply too much!!!
            
                # iter_log = copy.deepcopy(iter_log)
                #   for entry in iter_log:   
                    #   with open(path_csv_iter, 'a') as f:
                        #            f.write('No log because too much data')
                        #            f.write(str(ID) + '; ' + str(timestamp) + '; ' +
                        #                    str(axis2Deval[0]) + ',' + str(axis2Deval[1]) + '; ' + 
                        #                   str(list_axes [0][axis2Deval[0]]) + ',' + str(list_axes[0][axis2Deval[1]]) + ',' str(el3) + '; ' + str(entry[0]) + '; '
                        #                   + str(entry[1]) + '; ' + str(entry[2]) + '; '
                        #                  + str(entry[3][0][0]) + '; ' + str(entry[3][0][1]) + '; ' + str(entry[3][0][2]) + '; ' + str(entry[3][0][3]) + '; '
                        #                  + str(entry[3][1][0]) + '; ' + str(entry[3][1][1]) + '; ' + str(entry[3][1][2]) + '; ' + str(entry[3][1][3]) + '; '
                        #                 + str(entry[3][2][0]) + '; ' + str(entry[3][2][1]) + '; ' + str(entry[3][2][2]) + '; ' + str(entry[3][2][3]) + '; '
                        #                 + str(entry[3][3][0]) + '; ' + str(entry[3][3][1]) + '; ' + str(entry[3][3][2]) + '; ' + str(entry[3][3][3]) + '\n')
               
                    iter_log = []
            
            
            
            
    
    #list_inter_results.append(list_iter_steps)
    #list_iter_steps = []
       

#draw_registration_result(source_pc, target_pc, transform_GT)
