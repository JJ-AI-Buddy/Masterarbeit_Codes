# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:36:58 2023

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
import os
import numpy as np
import pandas as pd
import time
import math
from datetime import datetime
import matplotlib.pyplot as plt
import sklearn.metrics
import open3d as o3d


path_1 = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Labels_SMALL_Dataset_01_10122023.CSV"
path_2 = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Labels_SMALL_Dataset_02_10162023.CSV"

df_labels1 = pd.read_csv(path_1, sep = ";", header = 0, index_col = False, encoding='unicode_escape' )
df_labels1.columns = df_labels1.columns.str.replace('Â','')
df_labels2 = pd.read_csv(path_2, sep = ";", header = 0, index_col = False, encoding='unicode_escape' )
df_labels2.columns = df_labels2.columns.str.replace('Â','')


y1 = df_labels1.iloc[:,4::]
y2 = df_labels2.iloc[:,4::]
title = "Comparison of average label values of dataset 01 and 02"

# ax = df_GT.iloc[:,4:13].mean().plot(
#     kind='bar',
#     rot=90,
#     yerr = df_GT.iloc[:,4:13].std(),
#     capsize = 6,
#     #ylabel='Absolute error model prediction - GT label',
#     #title='Model validation - accuracy',
#     figsize=(14, 9))

x = np.arange(9)
width = 0.4
plt.figure(figsize=(20,12)) 
bar1 = plt.bar(x-(width/2), list(y1.mean()), width, color = 'magenta', edgecolor = 'black', zorder = 1, label = "Dataset 01") 
bar2 = plt.bar(x+(width/2), list(y2.mean()), width, color = 'yellow', edgecolor = 'black',zorder = 1, label = "Dataset 02")
plt.errorbar(x-(width/2), list(y1.mean()), yerr=list(y1.std()), fmt="o", color="black", capsize = 7, zorder = 2)
plt.errorbar(x+(width/2), list(y2.mean()), yerr=list(y2.std()), fmt="o", color="black", capsize = 7,  zorder = 2)
plt.axhline(y=0.0, color='black', linestyle='-', linewidth = 0.5, zorder = 2) 
plt.xticks(x, ['P2Point-ICP Translation Error [m]', 'P2Point-ICP Rotation Error [°]', 'P2Point-ICP Number of Iterations',
               'P2Plane-ICP Translation Error [m]', 'P2Plane-ICP Rotation Error [°]', 'P2Plane-ICP Number of Iterations',
               'P2Dist-NDT Translation Error [m]', 'P2Dist-NDT Rotation Error [°]', 'P2Dist-NDT Number of Iterations'],
           rotation=90, fontsize = 20) 
plt.yticks(fontsize=15) 
plt.ylabel("Translation Error /m, Rotation Error /°,\nNumber Iterations / -", fontsize = 20) 
plt.ylim(min(y1.mean()-y1.std())-50,max(y1.mean()+y1.std())+50)
plt.legend(handles = [bar1, bar2],loc='upper right', fontsize = 20) 
plt.grid(axis = 'y', color='gray', linestyle='dashed', zorder=-1)
plt.title(title, fontsize = 25)

# add error values
for k, i in enumerate(x):
    y_mean = list(y1.mean())
    y = y_mean[k] + list(y1.std())[k]
    r = list(y1.std())[k] / y_mean[k] * 100
    plt.annotate(f'{y_mean[k]:.2f}\n +/-\n {r:.1f}%', (i-(width/2), y), textcoords='offset points',
                xytext=(0, 5), ha='center', va='bottom', fontsize='xx-large')

k= 0 
i = 0    
for k, i in enumerate(x):
    y_mean = list(y2.mean())
    y = y_mean[k] + list(y2.std())[k]
    r = list(y2.std())[k] / y_mean[k] * 100
    plt.annotate(f'{y_mean[k]:.2f}\n +/-\n {r:.1f}%', (i+(width/2), 0), textcoords='offset points',
                xytext=(0, -210), ha='center', va='bottom', fontsize='xx-large')

plt.show()