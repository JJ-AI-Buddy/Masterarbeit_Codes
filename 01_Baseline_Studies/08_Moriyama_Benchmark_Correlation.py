# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:39:17 2023

@author: Johanna Hable


Creates correlation matrix plots for evaluation of the performance of different PC registration algorithms
"""

import open3d as o3d
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import os

Idx_timestamp = 3
Idx_axis = 0


# Your choice
one_axis = False
all_axes = True
one_timestamp = True
all_timestamps = False

NDT =  True #For NDT not all of the evaluation metric is available

if all_axes == True:
    Idx_axis = -1

if all_timestamps == True:
    Idx_timestamp = -1


# Uncomment and add double indent to the following code to generate correlation matrices for every axis of every timestamp automatically

#for i in range(0,5):
#    Idx_timestamp = i
    
#    for j in range(0,3):
#        Idx_axis = j

title = "Correlation of evaluation metrics on Moriyama Dataset\nTimestamp ID: " + str(Idx_timestamp) + ", Axis ID: " + str(Idx_axis)
name = 'C210-2_B_' + str(Idx_timestamp) + '_' + str(Idx_axis) + '.pdf'

path = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\04_Baseline\03_Moriyama_Evaluation\C210_RotlBaselineNDTMoriyamaCSV.csv"
path_plots = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\04_Baseline\03_Moriyama_Evaluation\Plots"

#Load GT poses from csv
df = pd.read_csv(path, delimiter = ';', header = 0, engine = 'python', encoding = 'unicode_escape')

i = 0
for i in range(0,len(df)):
    if df['Initial Rot. Error 2 [°]'].iloc[i] > 300:
        df['Initial Rot. Error 2 [°]'].iloc[i] = 360 - df['Initial Rot. Error 2 [°]'].iloc[i]
     
    ### only for NDT:
    elif df['Initial Rot. Error 2 [°]'].iloc[i] == 180:
        df['Initial Rot. Error 2 [°]'].iloc[i] = 180 - df['Initial Rot. Error 2 [°]'].iloc[i]

    if df['Rot. Error 2 [°]'].iloc[i] > 300:
        df['Rot. Error 2 [°]'].iloc[i] = 360 - df['Rot. Error 2 [°]'].iloc[i]

if NDT == False:
    categories = ["ID", "Timestamp GT Pose", "Axis", "Initial Transl. Error [m]", "Initial Rot. Error 2 [°]", "Fitness", "RMSE Inliers", "Transl. Error [m]", "Rot. Error 2 [°]", "Number Iterations"]
else:
    categories = ["ID", "Timestamp GT Pose", "Axis", "Initial Transl. Error [m]", "Initial Rot. Error 2 [°]", "Fitness", "Transl. Error [m]", "Rot. Error 2 [°]", "Number Iterations"]


df = df[categories]

timestamps = df.groupby('Timestamp GT Pose').groups.keys()
axes = df.groupby('Axis').groups.keys()

tstamp = list(timestamps)[Idx_timestamp]
axis = list(axes)[Idx_axis]

if one_timestamp == True:

    df = df.loc[df['Timestamp GT Pose'] == tstamp]
else: pass

if one_axis == True:
    
    df = df.loc[df ['Axis'] == axis]  
else: pass

if NDT == False:
    corr_categories = ["Fitness", "RMSE Inliers", "Number Iterations", "Initial Transl. Error [m]", "Transl. Error [m]", "Initial Rot. Error 2 [°]", "Rot. Error 2 [°]"]
else:
    corr_categories = ["Fitness", "Number Iterations", "Initial Transl. Error [m]", "Transl. Error [m]", "Initial Rot. Error 2 [°]", "Rot. Error 2 [°]"]



df_corr = df[corr_categories]
df_corr.fillna(0)
df_corr.astype('float64').dtypes
#print(df_corr[["Initial Rot. Error 1 [°]"]])

corr_matrix = df_corr.corr()           #.fillna(0.0)
matrix = np.triu(np.ones_like(corr_matrix))

plt.figure(figsize = (10,8))
sn.set(font_scale=1.3)
ax = sn.heatmap(corr_matrix, annot=True, mask = matrix, linewidth=.5, cmap=sn.cubehelix_palette(as_cmap=True), square = True, vmin=-1, vmax=1)
plt.title(title, fontsize = 15)
plt.tight_layout()
plt.savefig(os.path.join(path_plots, name))
plt.show()