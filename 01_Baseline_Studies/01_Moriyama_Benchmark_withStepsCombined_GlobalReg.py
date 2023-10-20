# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:50:17 2023

@author: Johanna Hable

Notes: 
    - This script provides an implementation of the RANSAC global registration algorithm in Open3D
    - "withSteps" means that results are yield after every iteration step
    
    
    - !!! Using Spyder as IDE is recommended because changes to the inputs need to be applied before every execution of this script !!!
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
    global transform_GT,t_vec
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    GT_temp = copy.deepcopy(source)

    #source_temp.paint_uniform_color([1, 0.706, 0])
    #target_temp.paint_uniform_color([0, 0.651, 0.929])
    GT_temp.paint_uniform_color([1,0,0])
    GT_temp.transform(transform_GT)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp,target_temp, GT_temp],
                                      zoom=0.4559,
                                      front=[0, 0, 1],
                                      lookat=[t_vec[0][0], t_vec[1][0], t_vec[2][0]],
                                      up=[0, 1, 0])

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
   


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh



def prepare_dataset(path_pc, path_map, voxel_size):
    
    global t_vec
    
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(path_pc)
    target = o3d.io.read_point_cloud(path_map)
    
    min_bound = np.zeros((3,1))
    max_bound = np.zeros((3,1))

    delta = np.array([[200],             # in x-direction +/- 200 m
                      [200],             # in y-direction +/- 200 m
                      [100]])            # in z-direction +/- 100 m

    for i in range(0,len(t_vec)):
        min_bound[i,0]= t_vec[i,0] - delta[i,0]
        max_bound[i,0] = t_vec[i,0] + delta[i,0]

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    target = o3d.geometry.PointCloud.crop(target,bbox)
    
    #trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                         [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    #source.transform(trans_init)
    #draw_registration_result(source_down, target_down, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    
    #draw_registration_result(source_down, target_down, np.identity(4))
    return source, target, source_down, target_down, source_fpfh, target_fpfh



def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


#############################################
##### SET INPUT ############################ 

ID = 'C300'        # Set ID like specified in the documentation

#count = 5
voxel_size_scale = [0.1,0.5,1.0,1.5,2.0]


# Input: Define the path direction of the map to use, the specific point cloud from a defined timestamp and the path direction to the csv file containing the GT poses (from NDT localization in Autoware) of the Localizer
path_map = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\Moriyama_Map.pcd"
path_GT_csv = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\14_Local_Pose.csv"
path_GNSS_csv = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\02_Moriyama_Data\13_GNSS_pose.csv"
path_to_file = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\03_Moriyama_Evaluation"

name_txt = str(ID) + '_GlobalBaselineICPMoriyama.txt'
path_txt = os.path.join(path_to_file, name_txt)
name_csv = str(ID) + '_GlobalBaselineICPMoriyama.csv'
path_csv = os.path.join(path_to_file,name_csv)


############################################
############################################

#Prepare CSV file
with open(path_csv, 'w') as f:
    f.write('ID;Timestamp GT Pose;Voxel size [m];Fitness;RMSE Inliers;Inlier correspondences;Transl. Error [m];Rot. Error 1 [°];Rot. Error 2 [°];Execut. Time [s];GNSS Transl. Error[m];GNSS Rot. Error 1 [°];GNSS Rot. Error 2 [°];t11;t12;t13;t14;t21;t22;t23;t24;t31;t32;t33;t34;t41;t42;t43;t44\n')

#Prepare txt file
with open(path_txt, 'w') as f:
    f.write('Evaluation of Fast Global RANSAC algorithm for map matching on Moriyama dataset \n' + str(datetime.now()) + "\n\n")


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


for x in range(0,len(arr_GT_poses)):
    
    path_pc = list_path_pc[x]
    init_pose = arr_GT_poses[x]
    GNSS_pose = arr_GNSS_poses[x]
    timestamp = timestamps[x]
    
    #Define initial transformation matrix from GT pose in the map coordinate system
    transform_GT, R_Euler, R_vec, t_vec = quat2transform(init_pose)

    #Define GNSS transformation matrix in map coordinate system
    transform_GNSS, R_Euler_GNSS, R_vec_GNSS, t_vec_GNSS = quat2transform(GNSS_pose)
    
    # Calculate errors of GNSS pose
    rse_transl_GNSS, rso_deg_1_GNSS, rso_deg_2_GNSS = evalMetrics(transform_GT, R_vec, R_Euler, transform_GNSS)
    
    for n in range(0,len(voxel_size_scale)):

        voxel_size = float(voxel_size_scale[n])
        #Prepare / Preprocess point clouds (Downsampling, FPFH Features)
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(path_pc, path_map,voxel_size)
    
        #Add the current information to the txt.-file
        with open(path_txt, 'a') as f:
            f.write('\n\n' + '#'*30 + '\n\n')
            f.write('Voxel size: \n%s ' %str(voxel_size) + "\n\n")
            f.write('Source point cloud:\nTimestamp = %s\nNumber of points = %s\nCropped (Y/N)? = N\n\n' 
                %((str(timestamp).zfill(10)), str(len(source_down.points))))
            f.write('Target point cloud: Moriyama map,\nNumber of points after cropping (RoI): %s\nCropped (Y/N)? = Y\n\n' %str(len(target_down.points)))

        with open(path_txt, 'a') as f:
            f.write('GT Transformation matrix:\n%s ' %str(transform_GT) + "\n\n")
    

        
        start = time.time()
        result_fast = execute_fast_global_registration(source_down, target_down,
                                                       source_fpfh, target_fpfh,
                                                       voxel_size)
        s = time.time()-start
        print("Fast global registration took %.3f sec.\n" % (time.time() - start))
        print(result_fast)
        
        final_transform = result_fast.transformation

        arr_correspondences = result_fast.correspondence_set
        arr_corr = arr_correspondences[arr_correspondences != -1]


        fitness = result_fast.fitness
        rmse = result_fast.inlier_rmse


        rse_transl, rso_deg_1, rso_deg_2 = evalMetrics(transform_GT, R_vec, R_Euler, final_transform)


        with open(path_csv, 'a') as f: 
            f.write(str(ID) + ';' + str(timestamp) + ';' + str(voxel_size) + ';' + str(fitness) + ';' + str(rmse) + ';' + str(float(len(arr_corr))) + 
                         ';' + str(rse_transl) + ';' + str(rso_deg_1) 
                           + ';' + str(rso_deg_2) + ';'
                           + str(s) + ';' + str(rse_transl_GNSS) + ';' + str(rso_deg_1_GNSS) + ';' + str(rso_deg_2_GNSS) + ';')
            f.write(str(final_transform[0][0]) + ';' + str(final_transform[0][1]) + ';' + str(final_transform[0][2]) + ';' + str(final_transform[0][3]) + ';' +
                    str(final_transform[1][0]) + ';' + str(final_transform[1][1]) + ';' + str(final_transform[1][2]) + ';' + str(final_transform[1][3]) + ';' +
                    str(final_transform[2][0]) + ';' + str(final_transform[2][1]) + ';' + str(final_transform[2][2]) + ';' + str(final_transform[2][3]) + ';' +
                    str(final_transform[3][0]) + ';' + str(final_transform[3][1]) + ';' + str(final_transform[3][2]) + ';' + str(final_transform[3][3]) + ';\n')
    
        #draw_registration_result(source_down, target_down, transform_GT)      