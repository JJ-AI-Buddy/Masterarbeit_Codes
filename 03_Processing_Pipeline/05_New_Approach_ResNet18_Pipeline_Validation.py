# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 00:38:52 2023

@author: Johanna Hable

Script for validation of the ranking model with different small validation sets
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
import PIL
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

import open3d as o3d
#from torchsummary import summary
from torchvision.models import resnet18, ResNet18_Weights

def preprocess_image(dir_img):
    image = PIL.Image.open(dir_img)
    
    image = transforms.functional.resize(image, (224, 224))# Resize if necessary, no resize and cropping afterwards
    image = transforms.functional.to_tensor(image)
    image = image.numpy()[::-1].copy()     # RGB to BGR (like camera later)
    #image = np.expand_dims(image,0)  # for 4D input
    image = torch.from_numpy(image)
    image = transforms.functional.normalize(image, mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    
    return image

def preprocess_labels(df_labels, weights):
    list_results = [0,0,0]
    ranking = [0,0,0]
    
    label = df_labels.iloc[index,4:14].to_numpy(dtype='float')
    
    list_results[0] = weights[0]*label[0] + weights[1]*label[1]+weights[2]*label[2]
    list_results[1] = weights[0]*label[3] + weights[1]*label[4]+weights[2]*label[5]
    list_results[2] = weights[0]*label[6] + weights[1]*label[7]+weights[2]*label[8]
    
    choice_GT = list_results.index(min(list_results))
    
    ranking[choice_GT] = 1
    
    return torch.tensor(ranking).float()

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
        batch_size, _, _,_ = x.shape

        
        x = self.model(x)
        
        x = self.model.fc(x)
        x = self.model.fc1(x)
        x = self.model.fc2(self.model.relu(x))
        x = self.model.fc3(self.model.relu(x))
        #x = self.model.threshold(x)

        #x = self.softmax(x)
        
        return x


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


validation_area = 'Downtown'
print_plots = False

path_model_weights = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Results\ResNet18_allweights_bestmodel.pth"
path_validation = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Downtown\Images"
path_plots = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Downtown\Plots"
path_GT_output = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Downtown\Labels_Downtown_Validation.CSV"

path_results = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Downtown\Results\Val_Results_ResNet18_02_Downtown_1-0_1-0_1-0.csv"
path_errors = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Downtown\Results\Val_Errors_ResNet18_02_Downtown_1-0_1-0_1-0.csv"


weights = [1.0,1.0,1.0]
threshold = 0.0

input_size = 33
num_el = 1
num_algorithms = 3
output_size = num_el * num_algorithms

criterion = nn.BCEWithLogitsLoss()

model = ResNet18_mod()
model.load_state_dict(torch.load(path_model_weights))
model.eval()

#Prepare output

data_columns = ['Choice model', 'Choice GT', 'Matching', 'BCE loss',
           'P2Point-ICP','P2Plane-ICP', 'P2Dist-NDT']

len(data_columns)


list_match = []

labels = pd.read_csv(path_GT_output, sep = ";", header = 0, index_col = False, encoding='unicode_escape' )

results = np.zeros((len(labels),7))
errors = np.zeros((len(labels),6))

index = 0
# Validation
for index in range(0,len(labels)):


    nbr = str(labels.loc[index,'Scan Nr.'])
    nbr = nbr.zfill(2)
    
    filename_scan = validation_area + '_Scan_' + nbr
    filename = filename_scan + ".jpg"
    file = filename.replace('.jpg', '')
    
    path_img = os.path.join(path_validation,filename)
    img = preprocess_image(path_img)
    
    label = preprocess_labels(labels, weights)

    
    prediction = model(img)
    
    loss = criterion(prediction[0], label)
    
    prediction = prediction[0].detach().numpy()
    label = label.detach().numpy()
    
    list_errors = abs(prediction-label)
    
    choice_model = list(prediction).index(max(list(prediction)))
    
    # Ground Truth
    
    choice_GT = list(label).index(max(list(label)))
    
    if choice_model == 0:
        algorithm = "Point-to-Point ICP"
        print("\nIn this environment the P2Point-ICP is expected to yield the best registration results\n")
        print("With prediction errors: P2Point-ICP %f - P2Plane-ICP %f  - P2Dist-NDT %f \n\n" %(list_errors[0],list_errors[1], list_errors[2]))
        
        if choice_GT == 0:
            print("\nThe Ground Truth choice is P2Point-ICP\n")
            list_match.append(1)
        elif choice_GT == 1:
            print("\nThe Ground Truth choice is P2Plane-ICP\n")
            list_match.append(0)
        elif choice_GT == 2:
            print("\nThe Ground Truth choice is P2Distribution-NDT\n")
            list_match.append(0)
            
    elif choice_model == 1:
        algorithm = "Point-to_Plane ICP"
        print("\nIn this environment the P2Plane-ICP is expected to yield the best registration results\n")
        print("With prediction errors: P2Point-ICP %f - P2Plane-ICP %f  - P2Dist-NDT %f \n\n" %(list_errors[0],list_errors[1], list_errors[2]))
        
        if choice_GT == 0:
            print("\nThe Ground Truth choice is P2Point-ICP\n")
            list_match.append(0)
        elif choice_GT == 1:
            print("\nThe Ground Truth choice is P2Plane-ICP\n")
            list_match.append(1)
        elif choice_GT == 2:
            print("\nThe Ground Truth choice is P2Distribution-NDT\n")
            list_match.append(0)
            
    elif choice_model == 2:
        algorithm = "Point-to_Distribution NDT"
        print("\nIn this environment the P2Distribution-NDT is expected to yield the best registration results\n")
        print("With prediction errors: P2Point-ICP %f - P2Plane-ICP %f  - P2Dist-NDT %f \n\n" %(list_errors[0],list_errors[1], list_errors[2]))
        
        if choice_GT == 0:
            print("\nThe Ground Truth choice is P2Point-ICP\n")
            list_match.append(0)
        elif choice_GT == 1:
            print("\nThe Ground Truth choice is P2Plane-ICP\n")
            list_match.append(0)
        elif choice_GT == 2:
            print("\nThe Ground Truth choice is P2Distribution-NDT\n")
            list_match.append(1)
            
    results[index,0] = choice_model
    results[index,1] = choice_GT
    errors[index,0] = choice_model
    errors[index,1] = choice_GT
    errors[index,2] = loss
    errors[index,3::]= list_errors
    
    if choice_model == choice_GT:
        results[index,2] = 1 
    
    results[index,3] = loss
    results[index,4:7] = prediction

df_out = pd.DataFrame(results, columns = data_columns)

# Write results to csv file
df_out.to_csv(path_results,sep = ';',index = False)

error_columns = ['model ranking', 'GT ranking', 'BCE loss']
error_columns.extend(data_columns[4::])

df_errors = pd.DataFrame(errors, columns = error_columns)

df_errors.to_csv(path_errors, sep = ';', index = False)

val = 1
true_matches = list_match.count(val) 
performance = true_matches/len(list_match)*100

print("The predicted ranking of registration algorithms is correct in %i out of %i validation scans." %(true_matches, len(list_match)))
print("Therefore, the model predicts in the area %s the correct registration algorithm to deploy in %f percent of all cases." %(validation_area,performance)) 
#print(results)         


# path_image = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_02_10162023_Images\Route_1_Scan_33.jpg"

# image = preprocess_image(path_image)

# image.save('img.jpg')
# Image.show(image)
# #image = image.detach().numpy()

# image = image.numpy()[::-1].copy()     # RGB to BGR (like camera later)
# image = image.numpy()

# image.shape
# image = np.rollaxis(image,0,3)

# PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')