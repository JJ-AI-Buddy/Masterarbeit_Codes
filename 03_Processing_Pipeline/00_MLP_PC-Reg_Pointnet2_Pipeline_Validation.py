# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 00:38:52 2023

@author: Johanna Hable

Script to validate the model with Pointnet++ backbone on different validation sets

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
#from torchsummary import summary
#from progress.bar import IncrementalBar
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

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
    
    pcd_fpfh = o3d.pipelines.registration.compöute_fpfh_feature(pcd,
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


def pc_normalize(pc):
    #l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    
    return pc

def subsample(pc, num):
    num_subset = num
    num_points = len(pc.points)
    if num_points > num_subset:

        pc = pc.farthest_point_down_sample(num_subset)

    return pc
    

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


validation_area = 'Urban'
print_plots = False

path_model_weights = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Results\Pointnet2_Own_Model\MLP_PC-Reg_Pointnet2_pretrained_V6_02_bestmodel.pth"
path_validation = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Urban\prepro"
path_plots = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Urban\Plots"
path_GT_output = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Urban\Labels_Urban_Validation.CSV"
path_results = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Urban\Results\Val-Pointnet_Results_02_Urban_1-0_1-0_1-0.csv"
path_input_vecs = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Urban\Results\Val-Pointnet_Inputs_02_Urban.csv"

path_errors = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Urban\Results\Val-Pointnet_Errors_02_Urban_1-0_1-0_1-0.csv"


weights = [1.0,1.0,1.0]

input_size = 33
num_el = 3
num_algorithms = 3
output_size = num_el * num_algorithms


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        #x = F.normalize(x,dim = 0)
        input_vec = x
        x = self.drop1(F.relu(self.fc1(x)))   #self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.fc2(x)))   #self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        #print(x)
        #x = F.log_softmax(x, -1)


        return x, input_vec


model_pretrained = get_model(10,normal_channel = False)

model_pretrained.fc1 = nn.Linear(1024,256)
model_pretrained.fc2 = nn.Linear(256,64)
model_pretrained.fc3 = nn.Linear(64,output_size)
#model_pretrained.fc3 = nn.Linear(256, output_size)

checkpoint = torch.load(path_model_weights)
model_pretrained.load_state_dict(checkpoint)
model = model_pretrained.eval()

#Prepare output

data_columns = ['Choice model', 'Choice GT', 'Matching', 'L1 (absolute) loss',
                'Pred. P2Point Transl. Error', 'Pred. P2Point Rot. Error', 'Pred. P2Point Number It.', 
                'Pred. P2Plane Transl. Error', 'Pred. P2Plane Rot. Error', 'Pred. P2Plane Number It.',
                'Pred. P2Dist Transl. Error', 'Pred. P2Dist Rot. Error', 'Pred. P2Dist Number It.']

len(data_columns)

list_results = [0,0,0]
list_results_GT = [0,0,0]
list_match = []

labels = pd.read_csv(path_GT_output, sep = ";", header = 0, index_col = False, encoding='unicode_escape' )

results = np.zeros((len(labels),13))
input_vecs = np.zeros((len(labels),1024))
errors = np.zeros((len(labels),12))

index = 0

# Validation
for index in range(0,len(labels)):
    
    #ID = str(labels.loc[index,'ID'])
    #ID_chars = [*ID]
    
    #filename_route = ''
    
    #if ID_chars[-1] == '0':
    #    filename_route = 'Route_1'
    #elif ID_chars[-1] == '1':
    #    filename_route = 'Route_2'
    #elif ID_chars[-1] == '2':
    #    filename_route = 'Route_3'
    #elif ID_chars[-1] == '3':
    #    filename_route = 'Route_4'
    #else: print("Something went wrong while reading the csv-file containing the labels!")

    nbr = str(labels.loc[index,'Scan Nr.'])
    nbr = nbr.zfill(2)
    
    filename_scan = validation_area + '_Scan_' + nbr
    filename = filename_scan + ".pcd"
    file = filename.replace('.pcd', '')
    
    path_pcd = os.path.join(path_validation,filename)
    
    pcd = o3d.io.read_point_cloud(str(path_pcd))
    
    start = time.time()
    point_cloud = subsample(pcd,10000)
    point_cloud = np.asarray(point_cloud.points)
    point_cloud = pc_normalize(point_cloud)
    point_cloud = np.expand_dims(point_cloud,0)
    #point_cloud = np.asarray(point_cloud.points)
    point_cloud = torch.from_numpy(point_cloud.astype(np.float32))
    point_cloud.to(device)
    point_cloud = point_cloud.permute(0,2,1)
    
    start = time.time()

    prediction, input_vec = model(point_cloud)
    
    end = time.time()
    
    print("Model execution took %f s" %(end-start))

    #input_vec, name = preprocess_input_cloud(pcd,path_pcd,path_plots,print_plots)
    #Normalize input vector 0 to 1
    #input_vec[0:33] = (input_vec[0:33]-np.min(input_vec[0:33]))/(np.max(input_vec[0:33])-np.min(input_vec[0:33]))
    #input_vec = input_vec[0:33]
    
    label = labels.iloc[index,4:14].to_numpy(dtype='float')
    #label = np.ones((1,35))
    #input_torch = torch.tensor(input_vec[0:33]).float()
    label_torch = torch.tensor(label).float()
    label_torch.to(device)

    
    loss = F.l1_loss(prediction[0], label_torch)
    
    prediction = prediction.detach().numpy()[0]
    
    list_results[0] = weights[0]*prediction[0] + weights[1]*prediction[1]+weights[2]*prediction[2]
    list_results[1] = weights[0]*prediction[3] + weights[1]*prediction[4]+weights[2]*prediction[5]
    list_results[2] = weights[0]*prediction[6] + weights[1]*prediction[7]+weights[2]*prediction[8]
    
    list_errors = abs(prediction - label)
    choice = list_results.index(min(list_results))
    
    # Ground Truth
    list_results_GT[0] = weights[0]*label[0] + weights[1]*label[1]+weights[2]*label[2]
    list_results_GT[1] = weights[0]*label[3] + weights[1]*label[4]+weights[2]*label[5]
    list_results_GT[2] = weights[0]*label[6] + weights[1]*label[7]+weights[2]*label[8]
    
    choice_GT = list_results_GT.index(min(list_results_GT))
    
    if choice == 0:
        algorithm = "Point-to-Point ICP"
        print("\nIn this environment the P2Point-ICP is expected to yield the best registration results\n")
        print("With prediction errors: Translation error %f [m] - Rotation error %f [°] - Number iterations %f [-]\n\n" %(list_errors[0],list_errors[1], list_errors[2]))
        
        if choice_GT == 0:
            print("\nThe Ground Truth choice is P2Point-ICP\n")
            list_match.append(1)
        elif choice_GT == 1:
            print("\nThe Ground Truth choice is P2Plane-ICP\n")
            list_match.append(0)
        elif choice_GT == 2:
            print("\nThe Ground Truth choice is P2Distribution-NDT\n")
            list_match.append(0)
            
    elif choice == 1:
        algorithm = "Point-to_Plane ICP"
        print("\nIn this environment the P2Plane-ICP is expected to yield the best registration results\n")
        print("With prediction errors: Translation error %f [m] - Rotation error %f [°] - Number iterations %f [-]\n\n" %(list_errors[3],list_errors[4], list_errors[5]))
        
        if choice_GT == 0:
            print("\nThe Ground Truth choice is P2Point-ICP\n")
            list_match.append(0)
        elif choice_GT == 1:
            print("\nThe Ground Truth choice is P2Plane-ICP\n")
            list_match.append(1)
        elif choice_GT == 2:
            print("\nThe Ground Truth choice is P2Distribution-NDT\n")
            list_match.append(0)
            
    elif choice == 2:
        algorithm = "Point-to_Distribution NDT"
        print("\nIn this environment the P2Distribution-NDT is expected to yield the best registration results\n")
        print("With prediction errors: Translation error %f [m] - Rotation error %f [°] - Number iterations %f [-]\n\n" %(list_errors[6],list_errors[7], list_errors[8]))
        
        if choice_GT == 0:
            print("\nThe Ground Truth choice is P2Point-ICP\n")
            list_match.append(0)
        elif choice_GT == 1:
            print("\nThe Ground Truth choice is P2Plane-ICP\n")
            list_match.append(0)
        elif choice_GT == 2:
            print("\nThe Ground Truth choice is P2Distribution-NDT\n")
            list_match.append(1)
            
    results[index,0] = choice
    results[index,1] = choice_GT
    errors[index,0] = choice 
    errors[index,1] = choice_GT
    errors[index,2] = loss
    errors[index,3::] = list_errors
    
    
    
    if choice == choice_GT:
        results[index,2] = 1 
    
    results[index,3] = loss
    results[index,4:13] = prediction
    
    input_vecs[index,:] = input_vec.detach().numpy()

df_out = pd.DataFrame(results, columns = data_columns)
df_inputs = pd.DataFrame(input_vecs)

# Write results to csv file
df_out.to_csv(path_results,sep = ';',index = False)
df_inputs.to_csv(path_input_vecs,sep = ';',index = False
                 )

error_columns = ['model ranking', 'GT ranking', 'L1 loss']
error_columns.extend(labels.columns[4::])

df_errors = pd.DataFrame(errors, columns = error_columns)

df_errors.to_csv(path_errors, sep = ';', index = False)

val = 1
true_matches = list_match.count(val) 
performance = true_matches/len(list_match)*100

print("The predicted ranking of registration algorithms is correct in %i out of %i validation scans." %(true_matches, len(list_match)))
print("Therefore, the model predicts in the area %s the correct registration algorithm to deploy in %f percent of all cases." %(validation_area,performance)) 
#print(results)         