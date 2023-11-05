# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:18:53 2023

@author: Johanna Hable

Script to train the ranking model with a Pointnet++ backbone which should lead to a more descriptive representation of the input point clouds in feature space
"""

import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
import torch
import torch.optim as optim
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
        #print(x)
        x = self.drop1(F.relu(self.fc1(x)))   #self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.fc2(x)))  #self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        #print(x)
        #x = F.log_softmax(x, -1)


        return x,l3_points


model = get_model(output_size, normal_channel = False)

model_pretrained = get_model(10,normal_channel = False)
# weight_paths = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Pythonscripts\Pointnet_Pointnet2_pytorch\log\classification\pointnet2_cls_msg_log\checkpoints\best_model.pth"
# checkpoint = torch.load(weight_paths)
# model_pretrained.load_state_dict(checkpoint['model_state_dict'])
model_pretrained.fc1 = nn.Linear(1024,256)
model_pretrained.fc2 = nn.Linear(256,64)
model_pretrained.fc3 = nn.Linear(64,output_size)
#model_pretrained.fc3 = nn.Linear(256, output_size)



for param in model_pretrained.parameters():
    #print(param)
    param.requires_grad = True

model_in_training = model_pretrained.to(device)
visualize_training_curve = True
model_path = 'MLP_PC-Reg_Pointnet2_8_bestmodel.pth'

path_dataset = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data\Dataset_02_10162023"
path_plots = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data"
path_results = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data\Results"
annotations_file = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data\Labels_SMALL_Dataset_02_10162023.CSV"
csv_file = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data\InputVectors_Dataset_02.csv"

def preprocess_input_cloud(pcd,filename,path_plots, save_plot):

    #file = os.path.basename(path)
    #filename = str(file).replace('.pcd','')
    #str_path = str(path)
    
    #pcd = o3d.io.read_point_cloud(str_path)
    
    start = time.time()
    #### Feature 01 - Total number of points ####
    num_points = len(pcd.points)
    
    end = time.time()
    exe_time = end-start
    
    #print("\n--------------------\n")
    #print("Feature 01 - Total number of points:\n%i points.\nCalculation took %f sec.\n\n" %(num_points, exe_time))
    
    # Subsample the point cloud to gain fixed number of points
    
    num_subset = 500
    
    if num_points > num_subset:

        pcd_sub = pcd.farthest_point_down_sample(num_subset)
        
        
        start = time.time()
        #### Feature 02 - Point density ####
        
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_sub)
        
        radius = 1 # 1 m search radius
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
        
    #    print("Feature 02 - Average point density:\n %f points/m^3.\nCalculation took %f sec.\n\n" %(avg_density,exe_time))
        
    
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
    
    #print("Feature 03 - FPFH\nCalculation took %f sec.\n\n" %exe_time)
    
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
        
   #     print("Plots of the FPFH feature descriptor have been saved successfully")
   #print("\n---------------\n")
        
    # Input feature vector 1x35 - 33 elements FPFH sum histogram standardised to total number of points; 1 element total number of points; 1 element average point density in points/m^3
        
    input_feature_vector = np.zeros((35)) #35
    input_feature_vector [:33]= norm_sum_fpfh
    input_feature_vector [33]=num_points
    input_feature_vector [34]=avg_density
        
    return input_feature_vector



class PCD_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, annotations_file, path_dataset, csv_file):
        self.labels = pd.read_csv(annotations_file, sep = ";", header = 0, index_col = False, encoding='unicode_escape' )
        #self.pcd_paths = glob.glob(os.path.join(self.directory, '*.pcd'))
        self.pcd_directory = path_dataset
        self.input_vectors = pd.read_csv(csv_file, sep = ";", header = 0, index_col = False, encoding='unicode_escape' )

    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        #print(index)
        ID = str(self.labels.loc[index,'ID'])
        ID_chars = [*ID]
        
        filename_route = ''
        
        if ID_chars[-1] == '0':
            filename_route = 'Route_1'
        elif ID_chars[-1] == '1':
            filename_route = 'Route_2'
        elif ID_chars[-1] == '2':
            filename_route = 'Route_3'
        elif ID_chars[-1] == '3':
            filename_route = 'Route_4'
        else: print("Something went wrong while reading the csv-file containing the labels!")

        nbr = str(self.labels.loc[index,'Scan Nr.'])
        nbr = nbr.zfill(2)
        
        filename_scan = '_Scan_' + nbr
        filename = filename_route + filename_scan + ".pcd"
        file = filename.replace('.pcd', '')
        #pcd_path = os.path.join(path_dataset,filename)
        
        #if os.path.lexists(pcd_path) == True:
            #print("Corresponding pcd file to the GT label was found and point cloud will now be loaded")
        #else: print("Corresponding pcd file to the GT label could not be found. Please check!")
        
        #point_cloud = o3d.io.read_point_cloud(str(pcd_path))
        
        idx_csv = self.input_vectors[self.input_vectors['Name']==file].index
        
        input_vector= self.input_vectors.iloc[idx_csv,1:36].to_numpy(dtype = 'float')
        
        #Normalize input vector 0 to 1
        input_vector[0:33] = (input_vector[0:33]-np.min(input_vector[0:33]))/(np.max(input_vector[0:33])-np.min(input_vector[0:33]))
        #input_vector = input_vector [0:33]

        label = self.labels.iloc[index,4:14].to_numpy(dtype='float')
        
        dir_pcd = os.path.join(self.pcd_directory,filename)

        #return image, torch.tensor(label).float()
        return str(dir_pcd), torch.tensor(input_vector[0,0:33]).float(), torch.tensor(label).float()





dataset = PCD_Dataset(annotations_file, path_dataset, csv_file)


#dataloader=torch.utils.data.DataLoader(dataset,shuffle=True)

test_percent = 0.2
num_test = int(test_percent * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

len(train_dataset), len(test_dataset)


b_size = 1
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=b_size,
    shuffle=True,
    #num_workers=1
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=b_size,
    shuffle=True,
    #num_workers=1
)



optimizer = optim.Adam(model_in_training.parameters(), lr=0.001) 

pytorch_total_params = sum(p.numel() for p in model_in_training.parameters() if p.requires_grad)

print(pytorch_total_params)


# Training
since = time.time()
num_epochs = 10
best_loss = 15
####################### Creating a txt-File als log-file ############

train_data_matrix = np.zeros((num_epochs,2))

log_file_name = model_path.replace('bestmodel.pth', 'train-log_' + datetime.today().strftime('%Y-%m-%d') +'.csv')

with open(os.path.join(path_results,log_file_name), 'w') as f:
    f.write("Train loss;Test loss")

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)
    
    # TRAINING #

    ################################
    model_in_training.train()
    train_loss = 0.0
    
    print("\n --- Training ---\n")
    #bar = IncrementalBar('ChargingBar', max = len(train_dataset))
    
    for pcd,inputs,label in iter(train_loader):
        optimizer.zero_grad()
        point_cloud = o3d.io.read_point_cloud(pcd[0])
        point_cloud = subsample(point_cloud,10000)
        point_cloud = np.asarray(point_cloud.points)
        point_cloud = pc_normalize(point_cloud)
        point_cloud = np.expand_dims(point_cloud,0)
        #point_cloud = np.asarray(point_cloud.points)
        point_cloud = torch.from_numpy(point_cloud.astype(np.float32))
        point_cloud.to(device)
        point_cloud = point_cloud.permute(0,2,1)
        
        output = model_in_training(point_cloud)
        #label = label.unsqueeze(0)
        loss = F.l1_loss(output[0], label) #F.l1_loss(output, label)                   
        train_loss += float(loss)
        loss.backward()
        optimizer.step()
        
        #bar.next()

        
    train_loss /= len(train_loader)
    #bar.finish()

    # TESTING #
    ################################
    model_in_training.eval()
    test_loss = 0.0
    
    print("\n\n --- Testing ---\n")
    #bar = IncrementalBar('ChargingBar', max = len(test_dataset))
    for pcd,inputs, label in iter(test_loader):
        point_cloud = o3d.io.read_point_cloud(pcd[0])
        point_cloud = subsample(point_cloud,10000)
        point_cloud = np.asarray(point_cloud.points)
        point_cloud = pc_normalize(point_cloud)
        point_cloud = np.expand_dims(point_cloud,0)
        #point_cloud = np.asarray(point_cloud.points)
        point_cloud = torch.from_numpy(point_cloud.astype(np.float32))
        point_cloud.to(device)
        point_cloud = point_cloud.permute(0,2,1)
        
        output = model_in_training(point_cloud)
        #label = label.unsqueeze(0)
        loss = F.l1_loss(output[0], label) #F.l1_loss(output, label)             
        test_loss += float(loss)
        

    #bar.next()
        
    test_loss /= len(test_loader)
    #bar.finish()

    print('\n\nTraining loss: %f, Testing loss: %f\n\n' % (train_loss, test_loss))
    
    if test_loss < best_loss:
        torch.save(model_in_training.state_dict(), os.path.join(path_results,model_path))
        best_loss = test_loss


    #with open(log_file_name, 'a') as f:
    #    f.write(f'\n\nEpoch {epoch}/{num_epochs - 1}\n' + '-'*50 + string_train + string_test +
    #       '\nTraining loss: %f, Testing loss: %f\n\n' % (train_loss, test_loss))
     
    
    ######################### Collect the training data in a matrix ############################
    
    #train_data_matrix[epoch][:4]=dev_mean[:4]
    train_data_matrix[epoch][0]=train_loss
    train_data_matrix[epoch][1]=test_loss   
    
    with open(os.path.join(path_results,log_file_name), 'a') as f:
        f.write("\n" + str(train_loss) + ";" + str(test_loss))


time_elapsed = time.time() - since
timeline = f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n'
result = f'Best loss: {best_loss:4f}'
print(timeline, result, train_data_matrix)


if visualize_training_curve == True:
    train_loss_vec = train_data_matrix[:,0] 
    test_loss_vec = train_data_matrix[:,1]
    
    best_train_loss = min(train_loss_vec)
    idx_best_train = np.argmin(train_loss_vec)
    best_test_loss = min(test_loss_vec)
    idx_best_test = np.argmin(test_loss_vec)
    
    
    plt.figure(figsize = (10,6))
    plt.grid(axis = 'x')
    title = "Training curve - Dataset_02 (1510 data points)"
    plt.title(title,fontweight ="bold", fontsize = 15) 
    plt.xticks(np.arange(0, len(train_data_matrix), 1).tolist())
    plt.xlabel("epoch", fontsize = 13)
    plt.ylabel ("L1 (absolute) loss", fontsize = 13)
    plt.tight_layout()
    #plt.plot(norm_sum_fpfh)
    plt.plot(train_loss_vec,color = 'blue', marker = 'o', label = 'training loss')
    plt.plot(test_loss_vec, color = 'red', marker = 'o', label = 'testing loss')
    plt.axhline(y = best_train_loss, color = 'blue', linestyle = 'dashed')
    plt.axhline(y = best_test_loss, color = 'red', linestyle = 'dashed')
    plt.annotate('best training loss',
            xy=(idx_best_train, best_train_loss), xycoords='data',
            size = 13,
            xytext=(-80, -80), textcoords='offset points',
            bbox=dict(boxstyle="round", fc="0.8"),
            arrowprops=dict(facecolor='black', shrink=30),
            horizontalalignment='center', verticalalignment='bottom')
    plt.annotate('best testing loss',
            xy=(idx_best_test, best_test_loss), xycoords='data',
            size = 13,
            xytext=(-80, 80), textcoords='offset points',
            bbox=dict(boxstyle="round", fc="0.8"),
            arrowprops=dict(facecolor='black', shrink=30),
            horizontalalignment='center', verticalalignment='bottom')
    plt.ylim(0)
    plt.legend(fontsize = 12, draggable = True, facecolor = 'white')
    save_path = os.path.join(path_plots,"TrainingCurve_Pointnet2_8_Dataset02.pdf")
    plt.savefig(save_path)
    plt.close()