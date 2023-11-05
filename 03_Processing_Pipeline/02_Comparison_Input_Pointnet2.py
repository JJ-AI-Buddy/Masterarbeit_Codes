# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 17:21:01 2023

@author: Johanna Hable

Script to compare the input vectors of the validation set and also the resulting output vectors of the model to evluate the model performance

"""

import glob
import os
import numpy as np
import pandas as pd
import time
import math
from datetime import datetime
import matplotlib.pyplot as plt


path_downtown = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Downtown\Results\Val-Pointnet_Results_01_Downtown_0-4_0-4_0-2.csv"
path_highway = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Highway\Results\Val-Pointnet_Results_01_Highway_0-4_0-4_0-2.csv"
path_rural = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Rural\Results\Val-Pointnet_Results_01_Rural_0-4_0-4_0-2.csv"
path_suburban = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Suburban\Results\Val-Pointnet_Results_01_Suburban_0-4_0-4_0-2.csv"
path_urban = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Urban\Results\Val-Pointnet_Results_01_Urban_0-4_0-4_0-2.csv"

path_in_downtown = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Downtown\Results\Val-Pointnet_Inputs_01_Downtown.csv"
path_in_highway = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Highway\Results\Val-Pointnet_Inputs_01_Highway.csv"
path_in_rural = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Rural\Results\Val-Pointnet_Inputs_01_Rural.csv"
path_in_suburban = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Suburban\Results\Val-Pointnet_Inputs_01_Suburban.csv"
path_in_urban = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Urban\Results\Val-Pointnet_Inputs_01_Urban.csv"


df_downtown = pd.read_csv(path_downtown, delimiter = ';', index_col = False, header = 0)
df_highway = pd.read_csv(path_highway, delimiter = ';', index_col = False, header = 0)
df_rural = pd.read_csv(path_rural, delimiter = ';', index_col = False, header = 0)
df_suburban = pd.read_csv(path_suburban, delimiter = ';', index_col = False, header = 0)
df_urban = pd.read_csv(path_urban, delimiter = ';', index_col = False, header = 0)

df_in_downtown = pd.read_csv(path_in_downtown, delimiter = ';', index_col = False, header = 0)
df_in_highway = pd.read_csv(path_in_highway, delimiter = ';', index_col = False, header = 0)
df_in_rural = pd.read_csv(path_in_rural, delimiter = ';', index_col = False, header = 0)
df_in_suburban = pd.read_csv(path_in_suburban, delimiter = ';', index_col = False, header = 0)
df_in_urban = pd.read_csv(path_in_urban, delimiter = ';', index_col = False, header = 0)

list_dfs = [df_downtown,df_highway, df_rural,df_suburban, df_urban]
list_inputs = [df_in_downtown,df_in_highway, df_in_rural,df_in_suburban, df_in_urban]

list_dfs_input = []
list_dfs_input_avg = []
list_dfs_output = []
list_dfs_output_avg = []

for el in range(0, len(list_dfs)):
    
    input_data_cache = np.zeros((5,1024))
    input_data = np.zeros((10,1024))
    output_data_cache = np.zeros((5,9))
    output_data = np.zeros((10,9))

    k = 0
    counter = 0
    for i in range(0,len(list_dfs[el])):
        input_data_cache[k,:] = list_inputs[el].iloc[k,0:1024]
        output_data_cache[k,:] = list_dfs[el].iloc[i,4:13]
        k += 1
        
        if k == 5:
            input_data[counter,:] = input_data_cache.mean(axis = 0)
            input_data_cache = np.zeros((5,1024))
            
            output_data[counter,:] = output_data_cache.mean(axis = 0)
            output_data_cache = np.zeros((5,9))
            counter += 1 
            k = 0
    
    df_i_new = pd.DataFrame(input_data,columns = list(list_inputs[el].columns)[0:1024])
    df_o_new = pd.DataFrame(output_data,columns = list(list_dfs[el].columns)[4:13])
    
    list_dfs_input.append(df_i_new)
    list_dfs_input_avg.append(df_i_new.mean())
    list_dfs_output.append(df_o_new)
    list_dfs_output_avg.append(df_o_new.mean())

list_i_labels = ['Average input Downtown',
               'Average input Highway',
               'Average input Rural',
               'Average input Suburban',
               'Average input Urban']
list_o_labels = ['Average output Downtown',
               'Average output Highway',
               'Average output Rural',
               'Average output Suburban',
               'Average output Urban']

df_i_plot = pd.DataFrame(list(zip(list_dfs_input_avg[0],list_dfs_input_avg[1],
                                list_dfs_input_avg[2], list_dfs_input_avg[3],
                                list_dfs_input_avg[4])),
                columns =list_i_labels, index = list(df_i_new.columns))

df_i_plot.iloc[0:33,:].plot(kind='bar', rot=90, ylabel='Normalized input vector value', title='Comparison of input vectors - Dataset Validation', figsize=(12, 8))

df_o_plot = pd.DataFrame(list(zip(list_dfs_output_avg[0],list_dfs_output_avg[1],
                                list_dfs_output_avg[2], list_dfs_output_avg[3],
                                list_dfs_output_avg[4])),
               columns =list_o_labels, index = list(df_o_new.columns))

df_o_plot.plot(kind='bar', rot=90, ylabel='output vector values', title='Comparison of output vectors - Dataset Validation', figsize=(12, 8))

for i in range(0,len(list(df_i_plot.columns))):
    y = df_i_plot.iloc[:,i]
    plt.figure(figsize = (20,2)) 
    heatmap = plt.imshow(y[np.newaxis,:], cmap='inferno',aspect='auto', origin = 'lower', vmin = 0, vmax = 1.5)
    title = y.name
    plt.colorbar(heatmap)
    plt.title(title,fontweight ="bold") 
    #plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
    plt.yticks([])
    plt.xlabel("vector indices")
    #plt.ylabel ("Point index")
    plt.tight_layout()


# plt.figure(figsize = (15,11))
# #for el in list_dfs_input:
# #    el.T.plot(kind = 'line')
# #plt.grid(axis = 'x')
# x_ticks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
# title = "Comparison of input vectors - Dataset Validation"
# plt.title(title,fontweight ="bold", fontsize = 25) 
# plt.xticks(np.arange(0, 33, 1).tolist())
# plt.xlabel("FPFH bins", fontsize = 20)
# plt.ylabel ("Sum over bin\n standardised by total point number\n and normalized", fontsize = 20)

# # j = 0
# # for el in list_dfs_input_avg:
# #     plt.plot(el, marker = 'o', label = list_labels[j])
# #     j+= 1

# plt.legend(fontsize = 20, draggable = True, facecolor = 'white')
# plt.tight_layout()
# plt.bar(x_ticks, list_dfs_input_avg, edgecolor = 'black')
#plt.plot(norm_sum_fpfh)
#plt.plot(list_dfs_input[0],color = 'blue', marker = 'o', label = 'training loss')
#plt.plot(test_loss_vec, color = 'red', marker = 'o', label = 'testing loss')