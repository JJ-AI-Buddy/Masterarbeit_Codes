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


path_downtown = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data\Dataset_Validation\Downtown\Results\Val_Results_V10_02_Downtown_0-4_0-4_0-2.csv"
path_highway = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data\Dataset_Validation\Highway\Results\Val_Results_V10_02_Highway_0-4_0-4_0-2.csv"
path_rural = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data\Dataset_Validation\Rural\Results\Val_Results_V10_02_Rural_0-4_0-4_0-2.csv"
path_suburban = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data\Dataset_Validation\Suburban\Results\Val_Results_V10_02_Suburban_0-4_0-4_0-2.csv"
path_urban = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data\Dataset_Validation\Urban\Results\Val_Results_V10_02_Urban_0-4_0-4_0-2.csv"


df_downtown = pd.read_csv(path_downtown, delimiter = ';', index_col = False, header = 0)
df_highway = pd.read_csv(path_highway, delimiter = ';', index_col = False, header = 0)
df_rural = pd.read_csv(path_rural, delimiter = ';', index_col = False, header = 0)
df_suburban = pd.read_csv(path_suburban, delimiter = ';', index_col = False, header = 0)
df_urban = pd.read_csv(path_urban, delimiter = ';', index_col = False, header = 0)

list_dfs = [df_downtown,df_highway, df_rural,df_suburban, df_urban]

list_dfs_input = []
list_dfs_input_avg = []
list_dfs_output = []
list_dfs_output_avg = []
for el in list_dfs:
    
    input_data = np.zeros((10,33))
    output_data = np.zeros((10,9))

    k = 0
    for i in range(0,len(input_data)):
        input_data[i,:] = el.iloc[k,4:37]
        output_data[i,:] = el.iloc[k,37:46]
        k += 5
    
    df_i_new = pd.DataFrame(input_data,columns = list(el.columns)[4:37])
    df_o_new = pd.DataFrame(output_data,columns = list(el.columns)[37:46])
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

df_i_plot.plot(kind='bar', rot=90, ylabel='Normalized input vector value', title='Comparison of input vectors - Dataset Validation', figsize=(12, 8))

df_o_plot = pd.DataFrame(list(zip(list_dfs_output_avg[0],list_dfs_output_avg[1],
                                list_dfs_output_avg[2], list_dfs_output_avg[3],
                                list_dfs_output_avg[4])),
               columns =list_o_labels, index = list(df_o_new.columns))

df_o_plot.plot(kind='bar', rot=90, ylabel='output vector values', title='Comparison of output vectors - Dataset Validation', figsize=(12, 8))

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