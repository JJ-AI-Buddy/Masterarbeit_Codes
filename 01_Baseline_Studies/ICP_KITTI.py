# -*- coding: utf-8 -*-
"""
Created on Sat May  6 19:09:59 2023

@author: Johanna
"""
import open3d as o3d
import copy
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from SupportFunctions import *

#### Functions
def crop_pc_bbox (pc, min_bd,max_bd):
    min_bound=min_bd
    max_bound=max_bd
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped_pc = o3d.geometry.PointCloud.crop(pc,bbox)
    
    return cropped_pc

def downsampling_pc (pc, voxel_size):
    pc_down = pc.voxel_down_sample(voxel_size)
    
    return pc_down

def detect_ground_thres(pc, thres_z, thres_norm_min, thres_norm_max):
    
# Estimate normals for each point (big radius because the input pc is already downsampled)

    pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=30))
    
    #GREEN = [0., 1., 0.]

    # Get the min value along the z-axis:
    z_min = min(pc.points, key=lambda x: x[2])[2]


    # Get the original points color to be updated:
    #pc.paint_uniform_color([0, 0.706, 1])
    #pc_colors = np.asarray(pc.colors)


    # Number of points:
    #n_points = pc_colors.shape[0]


    # update color:
    #for i in range(n_points):
        # if the current point is a ground point:
    #    if pc.points[i][2] <= z_min + threshold:
            #pc_colors[i] = GREEN  # color it green

    #pc.colors = o3d.utility.Vector3dVector(pc_colors)
    
    # Only select non-ground points for further operations
    ## z-value is greater than a specific threshold AND
    ## the normal is not parallel to the z-axis of the KS, (-0.9 , 0.9)
    
    pc = pc.select_by_index(np.where((np.asarray(pc.points)[:, 2] > z_min + thres_z) &
                                     (np.asarray(pc.normals)[:,2] < thres_norm_max) &
                                     (np.asarray(pc.normals)[:,2] > thres_norm_min))[0])

    return pc

def DBSCAN_clustering(pc,eps,min_points):
    
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(pc.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    #colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    #colors[labels < 0] = 0
    #pc_without_ground.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    np_clust_centers = np.zeros((max_label+1,3))
    clust_bboxes = []
    num_points = []
    
    np_pc = np.asarray(pc.points)
    np_pc_clust = np.insert(np_pc,0,labels,1)
    
    for lb in range(0,max_label+1):
        np_obj = np_pc_clust[np_pc_clust[:,0] == lb]
        num_points.append(len(np_obj))
        # center x
        np_clust_centers [lb][0] = np_obj[:,1].mean()
        # center y
        np_clust_centers [lb][1] = np_obj[:,2].mean()
        # center z
        np_clust_centers [lb][2] = np_obj[:,3].mean()
        
        
        obj = o3d.geometry.PointCloud()
        obj.points = o3d.utility.Vector3dVector(np_obj[:,1::])
        obj_bbox = obj.get_axis_aligned_bounding_box()
        obj_bbox.color=(0,0,0)
        clust_bboxes.append(obj_bbox)
    
        
    pc_clusters = o3d.geometry.PointCloud()
    pc_clusters.points = o3d.utility.Vector3dVector(np_clust_centers)
    pc_clusters.paint_uniform_color([0, 0, 1.0])
    
    # return point cloud containing all the centroids of the clusters in blue
    # return list of bboxes of all clusters in black
    return pc_clusters, clust_bboxes, num_points

def create_oriented_graph (pc_nodes, color):
    
    ##### Create graph with points from clustererd pc
    lineset = o3d.geometry.LineSet()

    # Sort point clouds by x-value from lowest to highest
    arr_points = np.asarray(pc_nodes.points)
    arr_points = arr_points[arr_points[:, 1].argsort()]

    #Set indices so that already ordered graph nodes are connected
    arr_indices = np.zeros((len(arr_points)-1,2))
    j = 0 
    k = 1 
    for row in arr_indices:
        row [0] = j 
        row [1] = k 
        j += 1 
        k += 1
    
    # Set color of graph in RGB
    arr_colors = np.zeros((len(arr_indices),3))
    for i in range(0,len(arr_colors)):
        arr_colors [i] = color

    lineset.points = o3d.utility.Vector3dVector(arr_points)
    lineset.lines = o3d.utility.Vector2iVector(arr_indices)
    lineset.colors = o3d.utility.Vector3dVector(arr_colors)  
    
    return lineset

def project2xy (pc_points):
    np_points = np.asarray(pc_points.points)
    np_points = np.around(np_points,2)
    np_points[:,2] = 0 
    np_points =  np.unique(np_points, axis=0)

    pc_points_xy = o3d.geometry.PointCloud()
    pc_points_xy.points = o3d.utility.Vector3dVector(np_points)
    
    return pc_points_xy
    
############### SETTINGS ###################################################################################
timestamp_src = 1
timestamp_trg = 1

name_txt = "TEST_05132023.txt"


path_to_data = r"C:\Users\Johanna\Downloads\2011_09_26\2011_09_26_drive_0001_sync\velodyne_points\data"
path_to_file = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline"

path_src = os.path.join(path_to_data, str(timestamp_src).zfill(10) + ".bin")
path_trg  = os.path.join(path_to_data, str(timestamp_trg).zfill(10) + ".bin")

path_txt = os.path.join(path_to_file, name_txt)
####################################################################################################################

# Load point clouds as PCD object
pc_1 = KITTI_Bin2PCD(path_src)
pc_2 = KITTI_Bin2PCD(path_trg)

source_pc = copy.deepcopy(pc_1)
target_pc = copy.deepcopy(pc_2)


# Voxel based downsampling of the PCs
source_pc = downsampling_pc(source_pc,0.5)
target_pc = downsampling_pc(target_pc,0.5)

# Statistical outlier removal (alternative: Radial outlier removal)
cl_src, ind_src = source_pc.remove_statistical_outlier(nb_neighbors=10,std_ratio=2.5)
cl_trg, ind_trg = target_pc.remove_statistical_outlier(10,2.5)

source_pc = cl_src.select_by_index(ind_src)
target_pc = cl_trg.select_by_index(ind_trg)

# Remove ground points from both PCs
source_pc = detect_ground_thres(source_pc, 2.0, -0.95,0.95)
target_pc = detect_ground_thres (target_pc, 2.0, -0.95, 0.95)

# Cropping of the target pc with a bounding box which represents the overlapping area of both PCs
#target_pc = crop_pc_bbox(target_pc, (-50,-50,-50),(-10,50,50))

# Initial pose guess: Initial transform matrix
transform_matrix = np.asarray([[1.0, 0.0, 0.0, 5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
source_pc = source_pc.transform(transform_matrix)

# Paint the PCs in distinctive colors
source_pc.paint_uniform_color([1, 0.0, 0.5])
target_pc.paint_uniform_color([0, 1, 0.5])

# Create KS for visualization
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

# Clustering
pc_clust_src, bboxes_src, num_points_src = DBSCAN_clustering(source_pc,5,10)
pc_clust_trg, bboxes_trg, num_points_trg = DBSCAN_clustering(target_pc,5,10)


# Cropping of the target pc with a bounding box which represents the overlapping area of both PCs
pc_clust_trg = crop_pc_bbox(pc_clust_trg, (-50,-50,-50),(-10,50,50))

# Project points to xy-plane
#pc_clust_xy_src = project2xy(pc_clust_src)
#pc_clust_xy_trg = project2xy(pc_clust_trg)

# Color the graph nodes
pc_clust_src.paint_uniform_color([1, 0, 0])  # RED
pc_clust_trg.paint_uniform_color([0,0,1])   # BLUE

##### Create graph with points from clustererd pc
lineset_src = create_oriented_graph(pc_clust_src, [1, 0, 0])
lineset_trg = create_oriented_graph(pc_clust_trg, [0, 0, 1])

lineset_src.lines = o3d.utility.Vector2iVector([[0,1],
                                                [0,2],
                                                [0,3],
                                                [0,4],
                                                [0,5],
                                                [0,6],
                                                [0,7],
                                                [0,8],
                                                [1,2],
                                                [1,3],
                                                [1,4],
                                                [1,5],
                                                [1,6],
                                                [1,7],
                                                [1,8],
                                                [2,3],
                                                [2,4],
                                                [2,5],
                                                [2,6],
                                                [2,7],
                                                [2,8],
                                                [3,4],
                                                [3,5],
                                                [3,6],
                                                [3,7],
                                                [3,8],
                                                [4,5],
                                                [4,6],
                                                [4,7],
                                                [4,8],
                                                [5,6],
                                                [5,7],
                                                [5,8],
                                                [6,7],
                                                [6,8],
                                                [7,8]])

visualizer_list = bboxes_3D
visualizer_list.append(origin)
visualizer_list.append(source_pc)
visualizer_list.append(pc_clust)
visualizer_list.append(lineset)

o3d.visualization.draw_geometries([lineset_src],
                                 zoom=0.7, front=[0, 0, 1],
                                 lookat=[0, 0, 0],
                                 up=[0, 1, 0],
                                 point_show_normal=False)

################### MAIN #######################################

with open(path_txt, 'w') as f:
    f.write('Evaluation of ICP algorithm for pc scan matching on KITTI dataset "2011_09_26_drive_0001_sync"\n ' + str(datetime.now()) + "\n\n")
    f.write('Source point cloud:\nTimestamp = %s\nNumber of points = %s\nCropped (Y/N)? = N\n\n' 
            %((str(timestamp_src).zfill(10)), str(len(source_pc.points))))
    f.write('Target point cloud:\nTimestamp = %s\nNumber of points = %s\nCropped (Y/N)? = Y\n\n'
            %((str(timestamp_trg).zfill(10)), str(len(target_pc.points))))


#### Scan matching with rectangualr bbox cropping
transform = icp(source_pc, target_pc, path_txt, transform_matrix)

print(transform)

with open(path_txt, 'a') as f:
    f.write('Transformation matrix:\n%s ' %str(transform) + "\n\n")    
    

##### Polygon volume

corners = np.array([[ 10, 50,  0.30217625],
 [ 30, 40,  0.29917539],
 [ 50, 30,  0.30329364],
 [ 70, 20,  0.3062945 ],
 [ 90, 0,  1.03551451],
 [ 70, -20,  1.03251366],
 [ 50, -30,  1.03663191],
 [ 30, -40,  1.03963277],
 [ 10, -50,  1.03963277]])

#corners = np.array(...)

# Convert the corners array to have type float64
bounding_polygon = corners.astype("float64")

# Create a SelectionPolygonVolume
vol = o3d.visualization.SelectionPolygonVolume()

# You need to specify what axis to orient the polygon to.
# I choose the "Y" axis. I made the max value the maximum Y of
# the polygon vertices and the min value the minimum Y of the
# polygon vertices.
vol.orthogonal_axis = "Z"
vol.axis_max = 10
vol.axis_min = -10

# Set all the Y values to 0 (they aren't needed since we specified what they
# should be using just vol.axis_max and vol.axis_min).
bounding_polygon[:, 2] = 0

# Convert the np.array to a Vector3dVector
vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)

# Crop the point cloud using the Vector3dVector
cropped_pcd = vol.crop_point_cloud(source_pc)

# Get a nice looking bounding box to display around the newly cropped point cloud
# (This part is optional and just for display purposes)
bounding_box = cropped_pcd.get_axis_aligned_bounding_box()
bounding_box.color = (1, 0, 0)

# Draw the newly cropped PCD and bounding box
o3d.visualization.draw_geometries([cropped_pcd, bounding_box],
                                  zoom=2,
                                  front=[5, -2, 0.5],
                                  lookat=[7.67473496, -3.24231903,  0.3062945],
                                  up=[1.0, 0.0, 0.0])