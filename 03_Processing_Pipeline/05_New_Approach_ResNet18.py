# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:09:21 2023

@author: Johanna
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
import PIL.Image
import os
import numpy as np
import time
import math
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
#from torchsummary import summary
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cpu") #if torch.cuda.is_available() else "cpu")
output_size = 3
visualize_training_curve = True

path_dataset = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_02_10162023_Images"
path_plots = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training"
path_results = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Results"
annotations_file = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Labels_SMALL_Dataset_02_10162023.CSV"
csv_file = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\InputVectors_Dataset_02.csv"

ranking_weights = [1.0, 1.0, 1.0]

class ResNet18_mod (nn.Module):
    def __init__(self):
        super(ResNet18_mod,self).__init__()
        

        # load the pretrained ResNet50
        self.model = models.resnet18(ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        
        for param in self.model.parameters():
            param.requires_grad = True
        # remove the last layer fc
        #self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # change last layer output_size = input_size/2
        self.model.fc = nn.Linear(num_ftrs,num_ftrs, bias = True)
        self.model.fc1 = nn.Linear(num_ftrs, int(num_ftrs/2), bias = True)
        self.model.fc2 = nn.Linear(int(num_ftrs/2), int(num_ftrs/4), bias = True)
        self.model.fc3 = nn.Linear(int(num_ftrs/4), output_size, bias = False)
        self.model.relu = nn.ReLU()        
        
        self.softmax = nn.Softmax(dim=1)
        #self.model.threshold = nn.Threshold(0.6,1.0)
      
        
    def forward(self, x):
        batch_size, _, _, _ = x.shape

        
        x = self.model(x)
        
        x = self.model.fc(x)
        x = self.model.fc1(x)
        x = self.model.fc2(self.model.relu(x))
        x = self.model.fc3(self.model.relu(x))
        #x = self.model.threshold(x)

        #x = self.softmax(x)
        
        return x

class PCD_Image_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, annotations_file, path_dataset, csv_file, ranking_weights):
        self.labels = pd.read_csv(annotations_file, sep = ";", header = 0, index_col = False, encoding='unicode_escape' )
        #self.pcd_paths = glob.glob(os.path.join(self.directory, '*.pcd'))
        self.im_directory = path_dataset
        self.input_vectors = pd.read_csv(csv_file, sep = ";", header = 0, index_col = False, encoding='unicode_escape' )
        self.weights = ranking_weights
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        #print(index)
        ID = str(self.labels.loc[index,'ID'])
        ID_chars = [*ID]
        
        list_results = [0,0,0]
        ranking = [0,0,0]
        
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
        filename = filename_route + filename_scan + ".jpg"
        file = filename.replace('.jpg', '')
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
        
        list_results[0] = self.weights[0]*label[0] + self.weights[1]*label[1]+self.weights[2]*label[2]
        list_results[1] = self.weights[0]*label[3] + self.weights[1]*label[4]+self.weights[2]*label[5]
        list_results[2] = self.weights[0]*label[6] + self.weights[1]*label[7]+self.weights[2]*label[8]
        
        choice_GT = list_results.index(min(list_results))
        
        ranking[choice_GT] = 1
        
        dir_img = os.path.join(self.im_directory,filename)
        
        image = PIL.Image.open(dir_img)
        
        image = transforms.functional.resize(image, (224, 224))# Resize if necessary, no resize and cropping afterwards
        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()     # RGB to BGR (like camera later)
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        

        return image, torch.tensor(ranking).float()


# Loading the pretrained model als backbone architecture
# weights = ResNet18_Weights.DEFAULT
# model = models.resnet18(weights = weights)        # else: ResNet50_Weights.DEFAULT

# for param in model.parameters():
#     param.requires_grad = False

# # Defining the model and the training parameters (learning rate, optimizer)

# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, output_size)

model = ResNet18_mod()

model_in_training = model.to(device)
model_path = 'ResNet18_allweights_TH_bestmodel.pth'

dataset = PCD_Image_Dataset(annotations_file, path_dataset, csv_file, ranking_weights)

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

optimizer = optim.SGD(model_in_training.parameters(), lr=0.001) 

criterion = nn.BCEWithLogitsLoss()

pytorch_total_params = sum(p.numel() for p in model_in_training.parameters() if p.requires_grad)

print(pytorch_total_params)


# Training
since = time.time()
num_epochs = 20
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
    
    for images, labels in iter(train_loader):
        #m = nn.Sigmoid()
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model_in_training(images)
        #label = label.unsqueeze(0)
        loss = criterion(output,labels) #F.l1_loss(output, label)                   
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
    for images, labels in iter(test_loader):
        #m = nn.Sigmoid()
        images = images.to(device)
        labels = labels.to(device)
        output = model_in_training(images)
        #label = label.unsqueeze(0)
        loss = criterion(output, labels) #F.l1_loss(output, label)             
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
    plt.ylabel ("BCE loss", fontsize = 13)
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
    save_path = os.path.join(path_plots,"TrainingCurve_ResNet18_allweights_TH_mod_wHead.pdf")
    plt.savefig(save_path)
    plt.close()