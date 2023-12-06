# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:16:29 2023

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


idx = [0,1,2]
env = 'Urban'
own_model = False

path = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Urban\Results\Val-Pointnet_Errors_02_Urban_1-0_1-0_1-0.csv"
path_results = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Urban\Results\Val-Pointnet_Results_01_Urban_0-4_0-4_0-2.csv"
path_labels = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Urban\Labels_Urban_Validation.CSV"


# Load the csv. file containting the prediction errors
df_errors = pd.read_csv(path, sep = ";", header = 0, index_col = False, encoding='unicode_escape' )
df_results = pd.read_csv(path_results, sep = ";", header = 0, index_col = False, encoding='unicode_escape' )
df_errors.columns = df_errors.columns.str.replace('Â','')
df_results.columns = df_results.columns.str.replace('Â','')
df_GT = pd.read_csv(path_labels, sep = ";", header = 0, index_col = False, encoding='unicode_escape' )

### Plot all prediction errors with error bars

ax = df_errors.iloc[:,3:12].mean().plot(
    kind='bar',
    rot=90,
    yerr = df_errors.iloc[:,3:12].std(),
    capsize = 6,
    #ylabel='Absolute error model prediction - GT label',
    #title='Model validation - accuracy',
    figsize=(14, 9))

#ax.bar_label(ax.containers[1], label_type = 'edge', fmt = "{:.1f}", padding = 5)
ax.plot(df_errors.iloc[:,3:12].mean(), marker = 'o', color = 'black', linewidth = 0)

y1 = df_errors.iloc[:,3:12].mean()
y1err = df_errors.iloc[:,3:12].std()

ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.set_ylim (min(y1-y1err)-5,max(y1+y1err)+15)
ax.set_ylabel('error model prediction - GT label',fontsize='x-large')
ax.set_title('Model validation ' + env + ' - accuracy', fontsize = 'xx-large') 
plt.axhline(y=0.0, color='black', linestyle='-', linewidth = 0.5)

# add error values
for k, x in enumerate(np.arange(9)):
    y = y1[k] + y1err[k]
    r = y1err[k] / y1[k] * 100
    ax.annotate(f'{y1[k]:.2f}\n +/- {r:.2f}%', (x, y), textcoords='offset points',
                xytext=(0, 3), ha='center', va='bottom', fontsize='large')

plt.show()
#############################################################################################

### Plot Translation, rotation and number iteration errors separated - edit idx variable accordingly ####
# idx = 0; translation error deltas
# idx = 1; rotation error deltas
# idx = 2; number iterations deltas
df_errors_transl = df_errors[['P2Point-ICP Transl. Error [m]', 'P2Plane-ICP Transl. Error [m]', 'NDT Transl. Error [m]']]
df_errors_rotl = df_errors[['P2Point-ICP Rot. Error 2 [°]', 'P2Plane-ICP Rot. Error 2 [°]', 'NDT Rot. Error 2 [°]']]
df_errors_it = df_errors[['P2Point-ICP Number Iterations', 'P2Plane-ICP Number Iterations', 'NDT Number Iterations']]

list_dfs = [df_errors_transl, df_errors_rotl, df_errors_it]


for ids in idx:
    if ids == 0:
        color = 'blue'
    elif ids == 1:
        color = 'orange'
    elif ids == 2:
        color = 'green'

    
    df = list_dfs[ids]
    
    ax = df.mean().plot(
        kind='bar',
        rot=0,
        yerr = df.std(),
        capsize = 6,
        #ylabel='Absolute error model prediction - GT label',
        #title='Model validation - accuracy',
        figsize=(14, 9),
        color = color)
    
    #ax.bar_label(ax.containers[1], label_type = 'edge', fmt = "{:.1f}", padding = 5)
    ax.plot(df.mean(), marker = 'o', color = 'black', linewidth = 0)
    
    y1 = df.mean()
    y1err = df.std()
    
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.set_ylim (min(y1-y1err)-5,max(y1+y1err)+15)
    ax.set_ylabel('Absolute error model prediction - GT label',fontsize='x-large')
    ax.set_title('Model validation ' + env + ' - accuracy', fontsize = 'xx-large') 
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth = 0.5)
    
    # add error values
    for k, x in enumerate(np.arange(3)):
        y = y1[k] + y1err[k]
        r = y1err[k] / y1[k] * 100
        ax.annotate(f'{y1[k]:.2f}\n +/- {r:.2f}%', (x, y), textcoords='offset points',
                    xytext=(0, 3), ha='center', va='bottom', fontsize='large')
    
    plt.show()
    
    # df.plot.box(
    #     figsize=(14, 9))


#####################################################################################
######################## Comparison matching - l1 loss - prediction errors ##########
# 4 Plots ###########################################################################

### Plot 1 ###
# L1 loss - model ranking - GT ranking



# ax = df_errors[['L1 loss']].plot(
#     kind='line',
#     marker = 'o',
#     linewidth = 2,
#     rot=0,
#     legend = True,
#     #capsize = 6,
#     #ylabel='Absolute error model prediction - GT label',
#     #title='Model validation - accuracy',
#     figsize=(14, 10),
#     #secondary_y = True,
#     color = 'magenta',
#     #ylabel = 'L1 loss')
#     )

    
# ax1 = ax.twinx()
# ax.set_ylabel('L1 loss', fontsize = 'x-large')
# ax1.plot(df_errors[['GT ranking']], color = 'green', linewidth = 2, label = 'GT ranking')
# ax1.plot(df_errors[['model ranking']], color = 'black', linewidth = 2, label = 'model ranking')
# ax1.set_yticks([0,1,2])
# ax1.set_ylabel('ranking - index of algorithm', fontsize='x-large')
# ax1.set_xlabel('index of data point', fontsize = 'x-large')
# ax1.set_title('Model validation - accuracy', fontsize = 'xx-large') 
# ax1.set_axisbelow(True)
# ax1.yaxis.grid(color='gray', linestyle='dashed')
# plt.legend()
# plt.show()


fig, axs = plt.subplots(4, 1)

arr_matching = np.zeros((len(df_errors),1))

for i in range(0, len(df_errors)): 
    if df_errors[['model ranking']].iloc[i,0] == df_errors[['GT ranking']].iloc[i,0]:
        arr_matching[i,0] = 2.0 
        
df_matching = pd.DataFrame(arr_matching, columns = ['Matching model-GT'])   

axs[0] = df_matching.plot(
    kind='bar',
    width = 1,
    #capsize = 6,
    #ylabel='Absolute error model prediction - GT label',
    #title='Model validation - accuracy',
    figsize=(14, 10),
    #secondary_y = True,
    color = 'lightgray',
    #ylabel = 'L1 loss',
    label = 'Positive matching model-GT'
    )     

#axs[0].set_axisbelow(True)
axs[0].yaxis.grid(color='gray', linestyle='dashed')

axs[0].set_yticks([0,1,2])
axs[0].set_ylabel('ranking - index of algorithm', fontsize='x-large')
axs[0].set_title('Model Validation ' + env + '\nComparison loss curve and algorithm ranking results', fontsize = 'xx-large')

line1, = axs[0].plot(df_errors[['GT ranking']], color = 'green', linewidth = 2, label = 'GT ranking')
line2, = axs[0].plot(df_errors[['model ranking']], color = 'blue', linewidth = 2, label = 'model ranking')

ax1 = axs[0].twinx()
line3, = ax1.plot(df_errors[['L1 loss']], marker = 'o',color = 'magenta', linewidth = 2, label = 'L1 loss')
ax1.set_ylabel('L1 loss', fontsize='x-large')
axs[0].set_xlabel('dataset points index', fontsize = 'x-large')

plt.legend(handles = [line1, line2, line3],loc='upper right', bbox_to_anchor=(1.35, 1.0))
plt.show()

#######################################################################################################

### Plot 2 ###
# Transl. error delta curves of all algorithms over the validation dataset

df_matching.iloc[:,0] = df_matching.iloc[:,0] / 2

df = list_dfs[0]

axs[1] = df_matching.plot(
    kind='bar',
    width = 1,
    #capsize = 6,
    #ylabel='Absolute error model prediction - GT label',
    #title='Model validation - accuracy',
    figsize=(14, 10),
    #secondary_y = True,
    color = 'lightgray',
    #ylabel = 'L1 loss',
    label = 'Positive matching model-GT'
    ) 


axs[1].set_yticks([0,1])
axs[1].set_title('Model Validation ' + env + '\nComparison curves of translation error deltas', fontsize = 'xx-large')


ax1 = axs[1].twinx()
line1, = ax1.plot(df.iloc[:,0], color = 'green', linewidth = 2, label = 'P2Point-ICP Transl. Error [m] delta')
line2, = ax1.plot(df.iloc[:,1], color = 'blue', linewidth = 2, label = 'P2Plane-ICP Transl. Error [m] delta')
line3, = ax1.plot(df.iloc[:,2], color = 'red', linewidth = 2, label = 'P2Dist-NDT Transl. Error [m] delta')

ax1.set_ylabel('prediction error model-GT', fontsize='x-large')
axs[1].set_xlabel('dataset point index', fontsize = 'x-large')

plt.legend(handles = [line1, line2, line3],loc='upper right', bbox_to_anchor=(1.35, 1.0))
plt.show()

##########################################################################################################

### Plot 3 ###
# Rot. error delta curves of all algorithms over the validation dataset

df = list_dfs[1]

axs[2] = df_matching.plot(
    kind='bar',
    width = 1,
    #capsize = 6,
    #ylabel='Absolute error model prediction - GT label',
    #title='Model validation - accuracy',
    figsize=(14, 10),
    #secondary_y = True,
    color = 'lightgray',
    #ylabel = 'L1 loss',
    label = 'Positive matching model-GT'
    ) 


axs[2].set_yticks([0,1])
axs[2].set_title('Model Validation ' + env + '\nComparison curves of rotation error deltas', fontsize = 'xx-large')

ax1 = axs[2].twinx()
line1, = ax1.plot(df.iloc[:,0], color = 'green', linewidth = 2, label = 'P2Point-ICP Rot. Error 2 [°] delta')
line2, = ax1.plot(df.iloc[:,1], color = 'blue', linewidth = 2, label = 'P2Plane-ICP Rot. Error 2 [°] delta')
line3, = ax1.plot(df.iloc[:,2], color = 'red', linewidth = 2, label = 'P2Dist-NDT Rot. Error 2 [°] delta')

ax1.set_ylabel('prediction error model-GT', fontsize='x-large')
axs[2].set_xlabel('dataset point index', fontsize = 'x-large')

plt.legend(handles = [line1, line2, line3],loc='upper right', bbox_to_anchor=(1.35, 1.0))
plt.show()

###################################################################################################################

### Plot 4 ###
# Number iterations delta curves of all algorithms over the validation dataset

df = list_dfs[2]

axs[3] = df_matching.plot(
    kind='bar',
    width = 1,
    #capsize = 6,
    #ylabel='Absolute error model prediction - GT label',
    #title='Model validation - accuracy',
    figsize=(14, 10),
    #secondary_y = True,
    color = 'lightgray',
    #ylabel = 'L1 loss',
    label = 'Positive matching model-GT'
    ) 


axs[3].set_yticks([0,1])
axs[3].set_title('Model Validation ' + env + '\nComparison curves of number of iterations deltas', fontsize = 'xx-large')


ax1 = axs[3].twinx()
line1, = ax1.plot(df.iloc[:,0], color = 'green', linewidth = 2, label = 'P2Point-ICP Number iterations delta')
line2, = ax1.plot(df.iloc[:,1], color = 'blue', linewidth = 2, label = 'P2Plane-ICP Number iterations delta')
line3, = ax1.plot(df.iloc[:,2], color = 'red', linewidth = 2, label = 'P2Dist-NDT Number iterations delta')

ax1.set_ylabel('prediction error model-GT', fontsize='x-large')
axs[3].set_xlabel('dataset point index', fontsize = 'x-large')

plt.legend(handles = [line1, line2, line3],loc='upper right', bbox_to_anchor=(1.35, 1.0))
plt.show()

#################################################################################################

### Plot the prediction errors only of the positive matching results
columns = ['Transl. error [m] delta', 'Rot. Error 2 [°] delta', 'Number iterations delta']


list_matching_err = []
#count = df_matching['Matching model-GT'].value_counts(1.0)
mask = df_matching['Matching model-GT'] == 1.0
count = len(df_matching[mask])

for i in range(0, len(df_matching)):
    if df_matching[['Matching model-GT']].iloc[i,0] == 1.0:
        index = df_errors[['GT ranking']].iloc[i,0]
        
        if index == 0.0:
            ls_app = list(df_errors.iloc[i,3:6])
        elif index == 1.0:
            ls_app = list(df_errors.iloc[i,6:9])
        elif index == 2.0:
            ls_app = list(df_errors.iloc[i,9:12])
        
        list_matching_err.append(ls_app)
        
df_matching_err = pd.DataFrame(list_matching_err, columns = columns)

ax = df_matching_err.mean().plot(
    kind='bar',
    rot=0,
    yerr = df_matching_err.std(),
    capsize = 6,
    #ylabel='Absolute error model prediction - GT label',
    #title='Model validation - accuracy',
    figsize=(14, 9))

#ax.bar_label(ax.containers[1], label_type = 'edge', fmt = "{:.1f}", padding = 5)
ax.plot(df_matching_err.mean(), marker = 'o', color = 'black', linewidth = 0)

y1 = df_matching_err.mean()
y1err = df_matching_err.std()

ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.set_ylim (min(y1-y1err)-5,max(y1+y1err)+15)
ax.set_ylabel('error model prediction - GT label',fontsize='x-large')
ax.set_title('Model validation ' + env + ' - positive matching accuracy', fontsize = 'xx-large') 
plt.axhline(y=0.0, color='black', linestyle='-', linewidth = 0.5)

# add error values
for k, x in enumerate(np.arange(3)):
    y = y1[k] + y1err[k]
    r = y1err[k] / y1[k] * 100
    ax.annotate(f'{y1[k]:.2f}\n +/- {r:.2f}%', (x, y), textcoords='offset points',
                xytext=(0, 3), ha='center', va='bottom', fontsize='large')

plt.show()

#########################################################################

### Top1 accuracy ###

### Plot the prediction errors only of the positive matching results
#columns = ['Transl. error [m] delta', 'Rot. Error 2 [°] delta', 'Number iterations delta']
columns = df_errors.columns[3:12]

list_matching_err = []
#count = df_matching['Matching model-GT'].value_counts(1.0)
mask = df_matching['Matching model-GT'] == 1.0
count = len(df_matching[mask])

for i in range(0, len(df_matching)):
    if df_matching[['Matching model-GT']].iloc[i,0] == 1.0:
        index = df_errors[['model ranking']].iloc[i,0]
        
        ls_app = list(df_errors.iloc[i,3:12])
        
        #print(index)
        
        # if index == 0.0:
        #     ls_app = list(df_errors.iloc[i,3:6])
        # elif index == 1.0:
        #     ls_app = list(df_errors.iloc[i,6:9])
        # elif index == 2.0:
        #     ls_app = list(df_errors.iloc[i,9:12])
        
        list_matching_err.append(ls_app)
        
df_matching_err = pd.DataFrame(list_matching_err, columns = columns)

list_matching_value = []
#count = df_matching['Matching model-GT'].value_counts(1.0)
mask = df_matching['Matching model-GT'] == 1.0
count = len(df_matching[mask])

for i in range(0, len(df_matching)):
    if df_matching[['Matching model-GT']].iloc[i,0] == 1.0:
        index = df_errors[['model ranking']].iloc[i,0]
        
        #print(index)
        if own_model == True:
            ls_app = list(df_results.iloc[i,37:46])
        else:
            ls_app = list(df_results.iloc[i,4:13])
        # if index == 0.0:
        #     ls_app = list(df_results.iloc[i,4:7])
        # elif index == 1.0:
        #     ls_app = list(df_results.iloc[i,7:10])
        # elif index == 2.0:
        #     ls_app = list(df_results.iloc[i,10:13])
        
        list_matching_value.append(ls_app)
        
df_matching_value = pd.DataFrame(list_matching_value, columns = columns)

df_percent = df_matching_err.div(df_matching_value)*100

ax = df_percent.mean().plot(
    kind='bar',
    #width = 1,
    rot=90,
    yerr = df_percent.std(),
    capsize = 6,
    #ylabel='Absolute error model prediction - GT label',
    #title='Model validation - accuracy',
    figsize=(14, 9),
    color = 'magenta')

#ax.bar_label(ax.containers[1], label_type = 'edge', fmt = "{:.1f}", padding = 5)
ax.plot(df_percent.mean(), marker = 'o', color = 'black', linewidth = 0)

y1 = df_percent.mean()
y1err = df_percent.std()

ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.set_ylim (0,max(y1+y1err)+150)
ax.set_ylabel('correct model ranking - average prediction error in %',fontsize=15)
ax.set_title('Model validation ' + env + ' - correct ranking accuracy - Pointnet++ backbone', fontsize = 20) 
plt.axhline(y=0.0, color='black', linestyle='-', linewidth = 0.5)
plt.xticks(fontsize=15)
plt.yticks(fontsize = 15)

r_sum = 0
# add error values
for k, x in enumerate(np.arange(9)):
    y = y1[k] + y1err[k]
    r = y1err[k] / y1[k] * 100
    ax.annotate(f'{y1[k]:.2f} %\n +/- {r:.2f} %P', (x, y), textcoords='offset points',
                xytext=(0, 3), ha='center', va='bottom', fontsize='large')
    
    r_sum = r_sum + r

plt.show()

model_acc = 100 - r_sum/9
print("The average model accuracy is %f percent" %model_acc)




################## Confusion matrices ######################################

# True-positive, False-negative
# False-positive, True-negative


score_p2distndt = []
score_p2pointicp = []
score_p2planeicp = []

label_p2distndt = []
label_p2pointicp = []
label_p2planeicp = []

for i in range(0,len(df_results)):
    #### P2Point ICP
    if df_results["Choice model"].iloc[i] == idx[0]:
        score_p2pointicp.append(1)
    else: 
        score_p2pointicp.append(0)
        
    if df_results["Choice GT"].iloc[i] == idx[0]:
        label_p2pointicp.append(1)
    else: 
        label_p2pointicp.append(0)
    
    #### P2Plane ICP
    if df_results["Choice model"].iloc[i] == idx[1]:
        score_p2planeicp.append(1)
    else:
        score_p2planeicp.append(0)
        
    if df_results["Choice GT"].iloc[i] == idx[1]:
        label_p2planeicp.append(1)
    else:
        label_p2planeicp.append(0)
        
    #### P2Dist NDT
    if df_results["Choice model"].iloc[i] == idx[2]:
        score_p2distndt.append(1)
    else:
        score_p2distndt.append(0)
        
    if df_results["Choice GT"].iloc[i] == idx[2]:
        label_p2distndt.append(1)
    else:
        label_p2distndt.append(0)
        
# Confusion matrix for P2Point ICP        
r0 = sklearn.metrics.confusion_matrix(label_p2pointicp, score_p2pointicp)  
#r0 = np.flip(r0) 
conf0 = sklearn.metrics.ConfusionMatrixDisplay(r0, display_labels = ['negative', 'positive'])
conf0.plot(cmap = plt.cm.Blues)
fig = conf0.ax_.get_figure() 
fig.set_figwidth(8)
fig.set_figheight(8)
conf0.ax_.set_xlabel('Model choice', fontsize = 15);
conf0.ax_.set_ylabel('GT choice', fontsize = 15)
conf0.ax_.tick_params(labelsize = 12)
plt.title('Confusion matrix for the P2Point-ICP', fontsize = 20)
plt.show()
#Confusion matrix for P2Plane ICP     
r1 = sklearn.metrics.confusion_matrix(label_p2planeicp, score_p2planeicp) 
#r1 = np.flip(r1)
conf1 = sklearn.metrics.ConfusionMatrixDisplay(r1, display_labels = ['negative', 'positive'])
conf1.plot(cmap = plt.cm.Blues)
fig = conf1.ax_.get_figure() 
fig.set_figwidth(8)
fig.set_figheight(8)
conf1.ax_.set_xlabel('Model choice', fontsize = 15);
conf1.ax_.set_ylabel('GT choice', fontsize = 15)
conf1.ax_.tick_params(labelsize = 12)
plt.title('Confusion matrix for the P2Plane-ICP', fontsize = 20)
plt.show()
#Confusion matrix for P2Dist ICP
r2 = sklearn.metrics.confusion_matrix(label_p2distndt, score_p2distndt) 
#r2 = np.flip(r2)
conf2 = sklearn.metrics.ConfusionMatrixDisplay(r2, display_labels = ['negative', 'positive'])
conf2.plot(cmap = plt.cm.Blues)
fig = conf2.ax_.get_figure() 
fig.set_figwidth(8)
fig.set_figheight(8)
conf2.ax_.set_xlabel('Model choice', fontsize = 15);
conf2.ax_.set_ylabel('GT choice', fontsize = 15)
conf2.ax_.tick_params(labelsize = 12)
plt.title('Confusion matrix for the P2Dist-NDT', fontsize = 20)
plt.show()

acc0 = (r0[0][0] + r0[-1][-1]) / np.sum(r0)
prec0 = sklearn.metrics.precision_score(label_p2pointicp, score_p2pointicp)
rec0 = sklearn.metrics.recall_score(label_p2pointicp, score_p2pointicp)
print("Accuracy of the model for the P2Point-ICP = %f\nPrecision: %f\nRecall: %f\n" %(acc0, prec0, rec0))

acc1 = (r1[0][0] + r1[-1][-1]) / np.sum(r1)
prec1 = sklearn.metrics.precision_score(label_p2planeicp, score_p2planeicp)
rec1 = sklearn.metrics.recall_score(label_p2planeicp, score_p2planeicp)
print("Accuracy of the model for the P2Plane-ICP = %f\nPrecision: %f\nRecall: %f\n" %(acc1, prec1, rec1))

acc2 = (r2[0][0] + r2[-1][-1]) / np.sum(r2)
prec2 = sklearn.metrics.precision_score(label_p2distndt, score_p2distndt)
rec2 = sklearn.metrics.recall_score(label_p2distndt, score_p2distndt)
print("Accuracy of the model for the P2Dist-NDT = %f\nPrecision: %f\nRecall: %f\n" %(acc2,prec2, rec2))


###### Bar chart - Comparison of average model output and GT label with error bars ########

y1 = df_GT.iloc[:,4:13]
if own_model == True: 
    y2 = df_results.iloc[:,37::]
    title_start = "MLP - Model validation "
else:
    y2 = df_results.iloc[:,4:13]
    title_start = 'PointNet++ backbone & MLP - Model validation '

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
bar1 = plt.bar(x-(width/2), list(y1.mean()), width, color = 'lightgreen', edgecolor = 'black', zorder = 1, label = "Average GT label") 
bar2 = plt.bar(x+(width/2), list(y2.mean()), width, color = 'blue', edgecolor = 'black',zorder = 1, label = "Average model prediction")
plt.errorbar(x-(width/2), list(y1.mean()), yerr=list(y1.std()), fmt="o", color="black", capsize = 7, zorder = 2)
plt.errorbar(x+(width/2), list(y2.mean()), yerr=list(y2.std()), fmt="o", color="black", capsize = 7,  zorder = 2)
plt.axhline(y=0.0, color='black', linestyle='-', linewidth = 0.5, zorder = 2) 
plt.xticks(x, ['P2Point-ICP Translation Error [m]', 'P2Point-ICP Rotation Error [°]', 'P2Point-ICP Number of Iterations',
               'P2Plane-ICP Translation Error [m]', 'P2Plane-ICP Rotation Error [°]', 'P2Plane-ICP Number of Iterations',
               'P2Dist-NDT Translation Error [m]', 'P2Dist-NDT Rotation Error [°]', 'P2Dist-NDT Number of Iterations'],
           rotation=90, fontsize = 20) 
plt.yticks(fontsize=15) 
plt.ylabel("Translation Error /m, Rotation Error /°,\nNumber Iterations / -", fontsize = 20) 
plt.ylim(min(y1.mean()-y1.std())-5,max(y1.mean()+y1.std())+50)
plt.legend(handles = [bar1, bar2],loc='upper right', fontsize = 20) 
plt.grid(axis = 'y', color='gray', linestyle='dashed', zorder=-1)
plt.title(title_start + env, fontsize = 25)

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
                xytext=(0, -70), ha='center', va='bottom', fontsize='xx-large')

plt.show()