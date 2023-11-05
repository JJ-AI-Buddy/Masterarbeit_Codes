# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:35:21 2023

@author: Johanna Hable

Script to speed up the training of the ranking model; the input feature vector is created for every point cloud of the dataset in advance and saved in a csv file
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import glob
import os
import numpy as np
import pandas as pd
import time
import math
from datetime import datetime
import matplotlib.pyplot as plt

import open3d as o3d

directory = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Data\Dataset_02_10162023\*.pcd"
path_general = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Data"
path_plots = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Data\Plots_Dataset_02_10162023"
file_csv = "InputVectors_Dataset_02.csv"

def preprocess_input_cloud(pcd,path_pcd,path_plots, save_plot):

    file = os.path.basename(path_pcd)
    filename = str(file).replace('.pcd','')
    str_path = str(path_pcd)
    
    #pcd = o3d.io.read_point_cloud(str_path)
    
    start = time.time()
    #### Feature 01 - Total number of points ####
    num_points = len(pcd.points)
    
    end = time.time()
    exe_time = end-start
    
    print("\n--------------------\n")
    print("Feature 01 - Total number of points:\n%i points.\nCalculation took %f sec.\n\n" %(num_points, exe_time))
    
    # Subsample the point cloud to gain fixed number of points
    
    num_subset = 500
    
    if num_points > num_subset:

        pcd_sub = pcd.farthest_point_down_sample(num_subset)
        
        
        start = time.time()
        #### Feature 02 - Point density ####
        
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_sub)
        
        radius = 10 # 1 m search radius
        area = 4/3 * np.pi * (radius**3)
        
        sum_density = 0
        
        for i in range(0,len(pcd_sub.points)):
            [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd_sub.points[i], radius)
            
            if k > 0:
                sum_density += k / area
            else: pass
        
        avg_density = sum_density/len(pcd_sub.points)
        
        end = time.time()
        exe_time = end-start
        
        print("Feature 02 - Average point density:\n %f points/m^3.\nCalculation took %f sec.\n\n" %(avg_density,exe_time))
        
    
    start = time.time()
    #### Feature 03 - Fast Point Feature Histogram (FPFH) #######
    
    radius_normal = 2
    
    #pcd_sub.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = radius_normal * 1.5
    #print("Compute FPFH feature with search radius %.3f. m" % radius_feature)
    
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    sum_fpfh = np.sum(pcd_fpfh.data.T,axis = 0)
    norm_sum_fpfh = sum_fpfh/len(pcd.points)
    
    end = time.time()
    exe_time = end-start
    
    print("Feature 03 - FPFH\nCalculation took %f sec.\n\n" %exe_time)
    
    if save_plot == True:
        
        #Heatmap for feature space of all points
        plt.figure(figsize = (10,5))    
        heatmap = plt.imshow(pcd_fpfh.data.T, cmap='hot',aspect='auto', origin = 'lower')
        title = "FPFH feature descriptor: " + filename
        plt.colorbar(heatmap)
        
        title = "FPFH feature descriptor: " + filename
        plt.title(title,fontweight ="bold") 
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
        plt.xlabel("Feature bins")
        plt.ylabel ("Point index")
        plt.tight_layout()
        save_path = path_plots + "\FPFH_All_" + filename + ".pdf"
        plt.savefig(save_path)
        #plt.show()
        plt.close()

        #Sum of all points standardised to total amount of points
        plt.figure(figsize = (10,5))
        title = "FPFH sum standardised: " + filename
        plt.title(title,fontweight ="bold") 
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
        x_ticks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
        plt.xlabel("Feature bins")
        plt.ylabel ("Sum over points / total amount of points")
        plt.tight_layout()
        #plt.plot(norm_sum_fpfh)
        plt.bar(x_ticks, norm_sum_fpfh, edgecolor = 'black')
        save_path = path_plots + "\FPFH_Sum_" + filename + ".pdf"
        plt.savefig(save_path)
        plt.close()
        
        print("Plots of the FPFH feature descriptor have been saved successfully")
    print("\n---------------\n")
        
    # Input feature vector 1x35 - 33 elements FPFH sum histogram standardised to total number of points; 1 element total number of points; 1 element average point density in points/m^3
        
    input_feature_vector = np.zeros((35)) #35
    input_feature_vector [:33]= norm_sum_fpfh
    input_feature_vector [33]=num_points
    input_feature_vector [34]=avg_density
        
    return input_feature_vector, filename
 


   
#prepare csv
with open(os.path.join(path_general,file_csv), 'w') as f:
    f.write("Name;i01;i02;i03;i04;i05;i06;i07;i08;i09;i10;i11;i12;i13;i14;i15;i16;i17;i18;i19;i20;i21;i22;i23;i24;i25;i26;i27;i28;i29;i30;i31;i32;i33;Tot.Points;Avg.Density\n")    
    
for i in glob.glob(directory):
    
    pcd = o3d.io.read_point_cloud(str(i))
    
    vector, name = preprocess_input_cloud(pcd,i,path_plots,False)

    with open(os.path.join(path_general,file_csv), 'a') as f:
        
        f.write(name)
        for el in vector:
            f.write(";" + str(el))
        f.write("\n")
    
    print("Feature vector saved in csv file successfully.")