# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import open3d as o3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pcl
import os
%matplotlib qt

print(o3d.__version__)


path_pc = r"/home/nextcar-dev/Downloads/Moriyama_Scan.pcd"
filename = os.path.basename(path_pc)

name = 'Prepro_' + filename   
new_path_pc= os.path.join(os.path.dirname(path_pc),name)




#### Online scan processing pipeline ####################33

#Input =  Point cloud
pcd = o3d.io.read_point_cloud(path_pc)
print("Point Cloud has been loaded with %i points" %len(pcd.points))

### PREPROCESSING ###

# (1) Outlier removal --> Output = pcd_inliers
cl, ind = pcd.remove_radius_outlier(nb_points = 10,radius = 0.15)
pcd_inliers = pcd.select_by_index(ind, invert = True)
pcd_outliers = pcd.select_by_index(ind)
pcd_outliers.paint_uniform_color([1.0, 0, 1.0])

print("Point Cloud contains %i points after outlier removal" %len(pcd_inliers.points))

# (2) Ground filtering - model (plane) fitting --> Output = pcd_non_ground
plane_model, ground = pcd_inliers.segment_plane(distance_threshold=0.35,  #0.15
                                         ransac_n=10,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation of ground filtering: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

pcd_ground = pcd_inliers.select_by_index(ground)
pcd_ground.paint_uniform_color([1.0, 0, 0.0])
pcd_non_ground = pcd_inliers.select_by_index(ground, invert=True)
print("Point Cloud contains %i points after ground removal" %len(pcd_non_ground.points))

# (3) Downsampling --> Output = pcd_non_ground_downvox, pcd_non_ground_downfpd

# (3.1) Voxel Downsampling
voxel_size = 0.5

pcd_non_ground_downvox = pcd_non_ground.voxel_down_sample(voxel_size)
print("Point Cloud contains %i points after downsampling with voxel size of %f m" %(len(pcd_non_ground_downvox.points),voxel_size))

# (3.2) Farthest point downsampling - only available with version 0.17 of Open3D
if o3d.__version__ == '0.17.0':
    num_points = 500
    
    tensor_pcd_non_ground = o3d.t.geometry.PointCloud.from_legacy_pointcloud(pcd_non_ground)
    
    if len(pcd_non_ground.points) > num_points:
        pcd_non_ground_downfpd = tensor_pcd_non_ground.farthest_point_down_sample(num_points)
        pcd_non_ground_downfpd.paint_uniform_color([0.0,1.0,0.0])
        print("Point Cloud has been downsampled to %i points by Farthest Point downsampling" %num_points)
    else:
        print("Point Cloud does not contain enough points to perform Farthest Point downsampling")

else:
    print("Open3D version 0.17.0 required to perform Farthest Point Downsampling.\nYour installed version is %s" %o3d.__version__)
    
    
### Save preprocessed point cloud for further operations with PCL ###

o3d.io.write_point_cloud(new_path_pc,name),pcd_non_ground_downvox)


### Feature/Keypoint detection ###

# Intrinsic shape signatures (ISS) - Keypoints with most changes in environment (see Open3D documentation)
radius_feature = voxel_size * 5

keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd_non_ground_downvox,
                                                        salient_radius=0.001,
                                                        non_max_radius=0.0,
                                                        gamma_21=0.975,
                                                        gamma_32=0.975,
                                                        min_neighbors = 5)

keypoints.paint_uniform_color([1.0,0.0,0.0])
print("%i keypoints have been detected." %len(keypoints.points))

### Feature description ###

# Fast point feature histogram (FPFH) - hand-crafted, 33d feature vector, encoding of local geometric variations (by using surface normals)
radius_normal = voxel_size * 2

keypoints.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

keypoints_fpfh = o3d.pipelines.registration.compute_fpfh_feature(keypoints,
                                                                 o3d.geometry.KDTreeSearchParamHybrid(
                                                                         radius=radius_feature,
                                                                         max_nn=100))
fpfh = np.asarray(keypoints_fpfh.data)
df_fpfh = pd.DataFrame(fpfh)
df_fpfh = df_fpfh.transpose()

print ("FPFH has been computed successfully for the keypoints")


### VISUALIZATION ###
o3d.visualization.draw_geometries([keypoints],
                                  zoom=0.8,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])

plt.imshow(df_fpfh)
plt.show()


series_values = df_fpfh.sum()
series_occurence = df.fillna(0).astype(bool).sum(axis=0)
series_occurence_norm = series_occurence/len(df_fpfh)
#Plots of the histograms
series_values.plot(kind="bar",figsize=(15, 8)) #Sum of every bin
series_occurence.plot(kind="bar",figsize=(15, 8)) #How many points are represented in each bin with a feature value non-zero - absolute
series_occurence_norm.plot(kind="bar",figsize=(15, 8)) #How many of all keypoints have a feature value non-zero for each bin - relative
#plt.imshow(hist)
#plt.show()



#### POINT CLOUD LIBRARY ###################

#### 3D Harris - feature detector

cloud = pcl.load(new_path_pc)
cloud.is_dense

#outrem = cloud.make_RadiusOutlierRemoval()
#outrem.set_radius_search(0.8)
#outrem.set_MinNeighborsInRadius(2)
#cloud_inliers = outrem.filter()


detector = cloud.make_HarrisKeypoint3D()
#detector.set_NonMaxSupression(True)
#detector.set_Radius(1)
#detector.set_NonMaxSupression (False)
#detector.set_RadiusSearch (100)
keypoints_harris = detector.compute()

    # std::cout << "keypoints detected: " << keypoints->size() << std::endl;
print("keypoints detected: " + str(keypoints_harris.size))