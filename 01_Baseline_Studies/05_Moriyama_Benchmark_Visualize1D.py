# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:15:44 2023

@author: Johanna Hable

Notes:
    - This script provides plots only for the 1D initial pose perturbations (x,y or z) or (roll, pitch or yaw)
    - Plots should be saved as pdf for better quality using Latex text editor later
    
    - !!! It is recommended to use Spyder as IDE because changes have to be made to the input before execution !!!
"""

import open3d as o3d
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.io as io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
io.renderers.default='svg'


######################### INPUT/SETTINGS #####################################################
Idx_timestamp = 3
Idx_axis = 1 #0 = x, 1 = y, 2 = z

# Your choice
one_axis = True
#all_axes = False
all_timestamps = False

max_y_value_transl = 2.5
max_y_axis_transl = 3

max_y_value_rotl = 30
max_y_axis_rotl = 35

offset = 1          # Offset for green arrow on the left - rotl = 1; transl = 0.1

#if all_axes == True:
#    Idx_axis = -1

if all_timestamps == True:
    Idx_timestamp = -1
    one_timestamp = False
else: one_timestamp = True


file_name = "\C203_TranslBaselineNDTMoriyama.csv"       #File name must be given with a backslash in front: "\xxxx"
title = "Evaluation of 1D initial pose perturbation on Moriyama Dataset\n - Timestamp ID: " + str(Idx_timestamp) + ", Axis ID: " + str(Idx_axis)
name = '1D_C103_' + str(Idx_timestamp) + '_' + str(Idx_axis) + '.pdf'

#######################################################################################################################

path_origin = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\03_Moriyama_Evaluation"
path = path_origin + file_name
path_plots = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\03_Moriyama_Evaluation\Plots1D_Python"

#Load GT poses from csv
df = pd.read_csv(path, delimiter = ';', header = 0, engine = 'python', encoding = 'unicode_escape')

#df = df[["ID", "Timestamp GT Pose", "Axis", "Initial Transl. Error [m]", "Initial Rot. Error 1 [°]", "Fitness", "RMSE Inliers", "Transl. Error [m]", "Rot. Error 1 [°]", "Number Iterations", "Execut. Time [s]", "GNSS Transl. Error[m]", "GNSS Rot. Error 1 [°]"]]
df = df.fillna(0)
df.dtypes

#Rotatory or translatory perturbation of initial pose?
if file_name[3] == '0':
    transl = True
    rotl = False
elif file_name[3]== '1': 
    transl = False
    rotl = True
else: print("File cannot be evaluated with this Pythonscript!!!")


timestamps = df.groupby('Timestamp GT Pose').groups.keys()
#timestamps = list(timestamps)
#del timestamps[3]                        #without timestamp 04; Intersection
axes = df.groupby('Axis').groups.keys()

tstamp = list(timestamps)[Idx_timestamp]
axis = list(axes)[Idx_axis]

if one_timestamp == True:

    df = df.loc[df['Timestamp GT Pose'] == tstamp]
    df_data = df
    
    mean_execut = np.mean(df_data['Execut. Time [s]'] / df_data['Number Iterations'])*1000 
    std_execut = np.std(df_data['Execut. Time [s]'] / df_data['Number Iterations'])*1000 
    max_execut = np.max(df_data['Execut. Time [s]'] / df_data['Number Iterations'])*1000  
    min_execut =  np.min(df_data['Execut. Time [s]'] / df_data['Number Iterations'])*1000

    print('Mean execution time per iteration: %f ms/Iteration' %mean_execut) 
    print('With standard deviation: %f ms/Iteration' %std_execut) 
    print('With max value: %f ms/Iteration' %max_execut)  
    print('With min value: %f ms/Iteration' %min_execut) 
else: pass

if one_axis == True:
    
    df = df.loc[df ['Axis'] == axis]  
    df_data = df
else: pass


if all_timestamps == True:
    #Calculate average values over all timestamps
    feature_list = ['Timestamp GT Pose','Number Iterations', 'Execut. Time [s]', 'Rot. Error 1 [°]', 'Initial Rot. Error 1 [°]',
                 'GNSS Transl. Error[m]', 'GNSS Rot. Error 1 [°]', 'Initial Transl. Error [m]', 'Transl. Error [m]' ]
    df_new = df[feature_list]
    
    mean_execut = np.mean(df_new['Execut. Time [s]'] / df_new['Number Iterations'])*1000 
    std_execut = np.std(df_new['Execut. Time [s]'] / df_new['Number Iterations'])*1000 
    max_execut = np.max(df_new['Execut. Time [s]'] / df_new['Number Iterations'])*1000  
    min_execut =  np.min(df_new['Execut. Time [s]'] / df_new['Number Iterations'])*1000

    print('Mean execution time per iteration: %f ms/Iteration' %mean_execut) 
    print('With standard deviation: %f ms/Iteration' %std_execut) 
    print('With max value: %f ms/Iteration' %max_execut)  
    print('With min value: %f ms/Iteration' %min_execut) 
    
    #Check if angle is >90° and subtract 180°
    df_new = df_new.reset_index()
    k= 0
    for k in range(0,len(df_new['Initial Rot. Error 1 [°]'])):
        if df_new.loc[:,'Initial Rot. Error 1 [°]'][k]>= 90:
            df_new.loc[:,'Initial Rot. Error 1 [°]'][k] = abs(df_new.loc[:,'Initial Rot. Error 1 [°]'][k]- 180)
    
    k= 0
    for k in range(0,len(df_new['Rot. Error 1 [°]'])):
        if df_new.loc[:,'Rot. Error 1 [°]'][k]>= 90:
            df_new.loc[:,'Rot. Error 1 [°]'][k] = abs(df_new.loc[:,'Rot. Error 1 [°]'][k]- 180)
            
    df_mean = pd.DataFrame(0, index=np.arange(17), columns=feature_list)
    df_dev = pd.DataFrame(0, index=np.arange(17), columns=feature_list)
    df_min = df_new.loc[df_new['Timestamp GT Pose']==list(timestamps)[0]]
    df_max = df_new.loc[df_new['Timestamp GT Pose']==list(timestamps)[0]]
    del df_mean[df_mean.columns[0]]
    del df_dev[df_dev.columns[0]]
    del df_min[df_min.columns[0]]
    df_min = df_min.reset_index()
    del df_min[df_min.columns[0]]
    del df_max[df_max.columns[0]]
    df_max = df_max.reset_index()
    del df_max[df_max.columns[0]]
    
    for ts in timestamps:
        df_cache = df_new.loc[df_new['Timestamp GT Pose']==ts]
        del df_cache[df_cache.columns[0]]
        df_cache = df_cache.reset_index()
        del df_cache[df_cache.columns[0]]
        df_mean += df_cache
        
        #Save minimum and maximum values in a dataframe
        for j in range(0, df_cache.shape[0]):
            for i in range(0, df_cache.shape[1]):
                if df_cache.iloc[j,i] < df_min.iloc[j,i]:
                    df_min.iloc[j,i] = df_cache.iloc[j,i]
                    
                elif df_cache.iloc[j,i] > df_max.iloc[j,i]:
                    df_max.iloc[j,i] = df_cache.iloc[j,i]
         
        
    #Dataframe with average values    
    df_data = df_mean / len(timestamps)
    
    #Calculate standard deviation
    for ts in timestamps:
        df_cache = df_new.loc[df_new['Timestamp GT Pose']==ts]
        del df_cache[df_cache.columns[0]]
        df_cache = df_cache.reset_index()
        del df_cache[df_cache.columns[0]]
        df_dev += pow(df_cache - df_data,2)
    
    #Dataframe with standard deviation
    df_dev = np.sqrt(df_dev/len(timestamps))
        

#fig = px.line(x=df.iloc[:,3], y=df['Number Iterations'], color=px.Constant("Number of Iterations"),
#             #labels=dict(x="Fruit", y="Amount", color="Time Period")
#             )

#fig = px.bar (df, x=df.iloc[:,3], y=((df['Transl. Error [m]'] - df['Initial Transl. Error [m]'])/df['Initial Transl. Error [m]'])*100,
             #secondary_y = True,
            #hover_data=['Rot. Error 1 [°]', 'Transl. Error [m]'],
             #color='Rot. Error 1 [°]',
             #range_color = [0,df['Rot. Error 1 [°]'].max()]
             #labels={'pop':'population of Canada'},
             #height=400
             #)
#del df_data[df_data.columns[0]]
#df_data.dtypes
#df_data = df_data.astype(float) 
### PLOTTING ###            
#Create figure with secondary axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

#Add curve for number of iterations
fig.add_trace(go.Scatter(x= df.iloc[:,Idx_axis + 3],
                         y=df_data['Number Iterations'],
                         mode = 'lines+markers',
                             name = 'Iterations',
                         line = dict(width = 2, color = 'magenta'),
                         marker = dict(color = 'magenta', symbol = 'square', size = 5)
                         ),
              secondary_y=True)

#Add curve for execution time
fig.add_trace(go.Scatter(x= df.iloc[:,Idx_axis+3],
                         y=df_data['Execut. Time [s]'],
                         mode = 'lines+markers',
                         name = 'Execution time in s',
                         line = dict(width = 2, color = 'orange'),
                         marker = dict(color = 'orange', symbol = 'circle', size = 5)
                         ),
              secondary_y=True)

#Title of x-axis
if transl == True:
    x_title = '1D perturbation of initial pose / m'
    #max_val = df_data[' Initial Transl.Error[m]'].max() + 1
if rotl == True:
    x_title = '1D perturbation of initial pose / rad'
    #max_val = df_data[' Initial Rot.Error 1[°]'].max() + 1
    

if transl == True:
    
    #Handling out-of-range data of the rotation error curve
    x_out = []
    ls_data = list(df_data['Rot. Error 1 [°]'])

    for i in range(len(ls_data)):
        if ls_data[i] > max_y_value_transl:
            x_out.append(i)

            ls_data[i]= max_y_value_transl
    
    #Add annotations for data points that can not be shown due to out-of-range of y-axis
    for outlier in x_out: 
        fig.add_annotation(x= df.iloc[outlier,Idx_axis+3],
                             y=max_y_value_transl,
                             name = 'Handling: Outlier of Rotation Error',
                             text = ">2.5",
                             showarrow = True,
                             font=dict(
                                 family="Arial",
                                 size=9,
                                 color='green'
                                 ),
                             align="center",
                             arrowhead=0,
                             arrowsize=1,
                             arrowwidth=1,
                             arrowcolor='green',
                             ax=0,
                             ay=-25,
                             #bordercolor="#c7c7c7",
                             #borderwidth=2,
                             #borderpad=4,
                             #bgcolor="#ff7f0e",
                             #opacity=0.8
                             #showlegend = False
                             
                  )

    #df_data.loc[df_data[' Rot.Error 1[°]'] > max_y_value, ' Rot.Error 1[°]'] = max_y_value
    
    #Add curve for rotation error - Clipped to max. displayable y-value
    fig.add_trace(go.Scatter (x=df.iloc[:,Idx_axis + 3],
                              y=ls_data,
                              mode = 'lines+markers',
                              name = 'Rotation Error in °',
                              line = dict(width = 2, color = 'green'),
                              marker = dict(color = 'green', symbol = 'diamond', size = 5),
                              ),
                  )
    
    #Add curve for GNSS Translation error
    fig.add_trace(go.Scatter (x=df.iloc[:,Idx_axis+3],
                              y=df_data['GNSS Transl. Error[m]'],
                              mode = 'lines',
                              name = ' GNSS Translation Error in m',
                              line = dict(width = 3, color = 'black', dash = 'dot')
                              ),
                  )
    
    #Add markers for initial perturbation
    fig.add_trace(go.Scatter (x=df.iloc[:,Idx_axis+3],
                          y=df_data['Initial Transl. Error [m]'],
                          mode = 'markers',
                          name = 'Initial Translation Error in m' ,
                          marker = dict(color = 'black', symbol = 'hourglass', size = 9)
                         #line = dict(width = , color = 'black')
                         ),
              )

    #Add bars for induced error of the intial pose
    fig.add_trace(go.Bar (x=df.iloc[:,Idx_axis+3],
                      y=df_data['Initial Transl. Error [m]'],
                      name = 'Test',
                      marker=dict(color = 'lightgrey',
                     #colorscale='viridis' ,#name = 'Percentual change of translation offset'
                     #showscale=True,
                      line = dict(width = 1, color = 'grey' )
                     ),
                     showlegend = False)
              )
    
    #Handling out-of-range data of the translation error delta curve
    x_out = []
    ls_delta = list(df_data['Transl. Error [m]']- df_data['Initial Transl. Error [m]'])

    for i in range(len(ls_delta)):
        if ls_delta[i] > max_y_value_transl-list(df_data.loc[:,'Initial Transl. Error [m]'])[i]:
            x_out.append(i)
            
            ls_delta[i] = max_y_value_transl-list(df_data.loc[:,'Initial Transl. Error [m]'])[i]

    for outlier in x_out: 
        fig.add_annotation(x= df.iloc[outlier,Idx_axis+3],
                             y=max_y_value_transl,
                             #mode = 'text',
                             name = 'Handling: Outlier of delta of translation error',
                             #line = dict(width = 2, color = 'orange'),
                             #marker = dict(color = 'blue', symbol = 'arrow-up', size = 15),
                             text = ">2.5",
                             showarrow = True,
                             font=dict(
                                 family="Arial",
                                 size=9,
                                 color='black'
                                 ),
                             align="center",
                             arrowhead=0,
                             arrowsize=1,
                             arrowwidth=1,
                             arrowcolor='black',
                             ax=0,
                             ay=-15,
                             #bordercolor="#c7c7c7",
                             #borderwidth=2,
                             #borderpad=4,
                             #bgcolor="#ff7f0e",
                             #opacity=0.8
                             #showlegend = False
                  )


    #Add change of pose error on top of bars of initial error + display its amount in color
    fig.add_trace(go.Bar (x=df.iloc[:,Idx_axis+3],
                      y=ls_delta ,#df_data[' Transl.Error[m]']- df_data[' Initial Transl.Error[m]'],
                      name = 'Delta Translation Error in m',
                      base = df_data['Initial Transl. Error [m]'],
                      marker=dict(color =df_data['Transl. Error [m]']- df_data['Initial Transl. Error [m]'], #df['Rot. Error 1 [°]'],
                                  colorscale=[[0, 'red'], [1.0, 'blue']] ,
                     #name = 'Percentual change of translation offset'
                     #range_color = [0,df['Rot. Error 1 [°]'].max()],
                     #cmin = -2,
                     #cmax = 2,
                     #showscale=True,
                                 line = dict(width = 1, color = 'black' )
             #secondary_y = True,
            #hover_data=['Rot. Error 1 [°]', 'Transl. Error [m]'],
             #color='Rot. Error 1 [°]',
             #range_color = [0,df['Rot. Error 1 [°]'].max()])
             #labels={'pop':'population of Canada'})
             #height=400
                     )
                        )
              )
    
    if all_timestamps == True:
        #Add error bars
        fig.add_trace(go.Scatter(x=df.iloc[:,Idx_axis+3],
                             y=df_data['Transl. Error [m]'],
                             name = 'Standard deviation Translation Error in m',
                             error_y=dict(type='data', 
                                          symmetric=True, 
                                          array=df_dev['Transl. Error [m]'], 
                                          #arrayminus=df_min['Transl. Error [m]']
                                          ),
                             marker = dict(color = 'darkslategray', symbol = 'line-ew-open', size = 10,
                                           line = dict(width = 2, color = 'darkslategray' ))
                             ))
    
if rotl == True:
    
    x_out = []
    ls_data = list(df_data['Transl. Error [m]'])

    for i in range(len(ls_data)):
        if ls_data[i] > max_y_value_rotl:
            x_out.append(i)

            ls_data[i]= max_y_value_rotl

    for outlier in x_out: 
        fig.add_annotation(x= df.iloc[outlier,Idx_axis+3],
                             y=max_y_value_rotl,
                             #mode = 'text',
                             name = 'Handling: Outlier of Translation Error',
                             #line = dict(width = 2, color = 'orange'),
                             #marker = dict(color = 'blue', symbol = 'arrow-up', size = 15),
                             text = ">30",
                             #textposition="top right",
                             showarrow = True,
                             font=dict(
                                 family="Arial",
                                 size=9,
                                 color='green'
                                 ),
                             align="center",
                             arrowhead=0,
                             arrowsize=1,
                             arrowwidth=1,
                             arrowcolor='green',
                             ax=0,
                             ay=-25,
                             #bordercolor="#c7c7c7",
                             #borderwidth=2,
                             #borderpad=4,
                             #bgcolor="#ff7f0e",
                             #opacity=0.8
                             #showlegend = False
                             
                  )
    
    #Add curve for translation error
    fig.add_trace(go.Scatter (x=df.iloc[:,Idx_axis + 3],
                              y=ls_data,
                              mode = 'lines+markers',
                              name = 'Translation Error in m',
                              line = dict(width = 2, color = 'green'),
                              marker = dict(color = 'green', symbol = 'diamond', size = 5)
                              ),
                  )
    
    #Add curve for GNSS Rotation error
    fig.add_trace(go.Scatter (x=df.iloc[:,3],
                              y=df_data['GNSS Rot. Error 1 [°]'], 
                              mode = 'lines',
                              name = 'GNSS Rotation Error in °',
                              line = dict(width = 3, color = 'black', dash = 'dot')
                             ),
                  )
    
    
    #Check if angle is >90° and subtract 180°
    df_data = df_data.reset_index()
    k= 0
    for k in range(0,len(df_data['Initial Rot. Error 1 [°]'])):
        if df_data.loc[:,'Initial Rot. Error 1 [°]'][k]>= 90:
            df_data.loc[:,'Initial Rot. Error 1 [°]'][k] = abs(df_data.loc[:,'Initial Rot. Error 1 [°]'][k]- 180)
    
    k= 0
    for k in range(0,len(df_data['Rot. Error 1 [°]'])):
        if df_data.loc[:,'Rot. Error 1 [°]'][k]>= 90:
            df_data.loc[:,'Rot. Error 1 [°]'][k] = abs(df_data.loc[:,'Rot. Error 1 [°]'][k]- 180)
    
    
    #Add markers for initial perturbation
    fig.add_trace(go.Scatter (x=df.iloc[:,Idx_axis+3],
                          y=df_data['Initial Rot. Error 1 [°]'],
                          mode = 'markers',
                          name = 'Initial Rotation Error in °' ,
                          marker = dict(color = 'black', symbol = 'hourglass', size = 9)
                         #line = dict(width = , color = 'black')
                         ),
              )

    #Add bars for induced error of the intial pose
    fig.add_trace(go.Bar (x=df.iloc[:,Idx_axis+3],
                      y=df_data['Initial Rot. Error 1 [°]'],
                      name = 'Test',
                      marker=dict(color = 'lightgrey',
                     #colorscale='viridis' ,#name = 'Percentual change of translation offset'
                     #showscale=True,
                      line = dict(width = 1, color = 'grey' )
                     ),
                     showlegend = False)
            )

    x_out = []
    ls_delta = list(df_data['Rot. Error 1 [°]']- df_data['Initial Rot. Error 1 [°]'])

    for i in range(len(ls_delta)):
        if ls_delta[i] > max_y_value_rotl-list(df_data.loc[:,'Initial Rot. Error 1 [°]'])[i]:
            x_out.append(i)
            
            ls_delta[i] = max_y_value_rotl-list(df_data.loc[:,'Initial Rot. Error 1 [°]'])[i]

    for outlier in x_out: 
        fig.add_annotation(x= df.iloc[outlier,Idx_axis+3],
                             y=max_y_value_rotl,
                             name = 'Handling: Outlier of delta of rotation error',
                             text = ">30",
                             showarrow = True,
                             font=dict(
                                 family="Arial",
                                 size=9,
                                 color='black'
                                 ),
                             align="center",
                             arrowhead=0,
                             arrowsize=1,
                             arrowwidth=1,
                             arrowcolor='black',
                             ax=0,
                             ay=-15,
                             #bordercolor="#c7c7c7",
                             #borderwidth=2,
                             #borderpad=4,
                             #bgcolor="#ff7f0e",
                             #opacity=0.8
                             #showlegend = False
                  )


    #Add change of pose error on top of bars of initial error + display its amount in color
    fig.add_trace(go.Bar (x=df.iloc[:,Idx_axis+3],
                      y=ls_delta,
                      name = 'Delta Rotation Error in °',
                      base = df_data['Initial Rot. Error 1 [°]'],
                      marker=dict(color =df_data['Rot. Error 1 [°]']- df_data['Initial Rot. Error 1 [°]'], #df['Rot. Error 1 [°]'],
                                  colorscale=[[0, 'red'], [1.0, 'blue']] ,
                     #name = 'Percentual change of translation offset'
                     #range_color = [0,df['Rot. Error 1 [°]'].max()],
                     #cmin = -2,
                     #cmax = 2,
                     #showscale=True,
                                 line = dict(width = 1, color = 'black' )
             #secondary_y = True,
            #hover_data=['Rot. Error 1 [°]', 'Transl. Error [m]'],
             #color='Rot. Error 1 [°]',
             #range_color = [0,df['Rot. Error 1 [°]'].max()])
             #labels={'pop':'population of Canada'})
             #height=400
                     )
                        )
              )
    
    if all_timestamps == True:
        #Add error bars
        fig.add_trace(go.Scatter(x=df.iloc[:,Idx_axis+3],
                             y=df_data['Rot. Error 1 [°]'],
                             name = 'Standard deviation Rotation Error in °',
                             error_y=dict(type='data', 
                                          symmetric=True, 
                                          array=df_dev['Rot. Error 1 [°]'], 
                                          #arrayminus=df_min['Transl. Error [m]']
                                          ),
                             marker = dict(color = 'darkslategray', symbol = 'line-ew-open', size = 10,
                                           line = dict(width = 2, color = 'darkslategray' ))
                             ))
    

#fig.add_traces(list(px.bar (df, x=df.iloc[:,3], y=df['Transl. Error [m]'],
             #secondary_y = True,
            #hover_data=['Rot. Error 1 [°]', 'Transl. Error [m]'],
#             color='Rot. Error 1 [°]',
#             range_color = [0,df['Rot. Error 1 [°]'].max()],
#             labels={'pop':'population of Canada'},
             #height=400
#             ).select_traces()))

#Add markers to show which curves refer to the secondary y-axis
fig.add_trace(go.Scatter(x=df.iloc[15:16, Idx_axis + 3],
                         y=(34,34),
                         mode="markers+lines",
                         marker=dict(
                             symbol="arrow-right",
                             color="magenta",
                             size=10,
                             ),
                         showlegend = False),
              secondary_y=True)


fig.add_trace(go.Scatter(x=df.iloc[14:15, Idx_axis + 3],
                         y=(34,34),
                         mode="markers+lines",
                         marker=dict(
                             symbol="arrow-right",
                             color="orange",
                             size=10,
                             ),
                         showlegend = False),
              secondary_y=True)
if transl == True:
    max_y = max_y_axis_transl
elif rotl == True:
    max_y = max_y_axis_rotl
else: pass

fig.add_trace(go.Scatter(x=df.iloc[0:1, Idx_axis + 3],
                         y=(max_y-offset,max_y-offset),                           
                         mode="markers+lines",
                         marker=dict(
                             symbol="arrow-left",
                             color="green",
                             size=10,
                             ),
                         showlegend = False),
                              )

#fig.update_traces(patch={"line": {"color": "black", "width": 2, "dash": 'dot'}}, selector={"name": "Delta GNSS Translation Error in m"}) 
#fig.update_traces(patch={"line": {"color": "grey", "width": 2, "dash": 'dot'}}, selector={"name": "Delta GNSS Rotation Error in °"}) 
#fig.update_traces(patch={"marker": {"color": "black", "size":10, "symbol":"arrow-bar-down"}}, selector={"name": "Initial Translation Error in m"})

#fig.add_line(x=df.iloc[:,3], y=df['Number Iterations'], color=px.Constant("Number of Iterations"),
             #labels=dict(x="Fruit", y="Amount", color="Time Period")
#             )

fig.update_xaxes(ticktext=round(df.iloc[:, Idx_axis + 3],2),
                 tickvals=df.iloc[:,Idx_axis + 3])

#fig.update_yaxes(range=[None, max_y_value])


fig.update_layout(font=dict(size=11 ),
                  font_family="Arial",
                  yaxis_title='Translation Error / m, Rotation Error / °',
                  xaxis_title=x_title,
                  template='simple_white',
                  bargap = 0,
                  barmode='overlay',
                  yaxis1 = dict(range = [0, max_y], autorange = False),
                  yaxis2 = dict(range = [0, 35], autorange = False, title = 'Iterations / -, Execution time / s'),
                  title = dict(text = title, font=dict(size=15), yref='container'),
                  legend=dict(yanchor="top",xanchor="left", x = 1.1,font=dict(size=10))
 
                  )
#fig.update_yaxes(range=[None, max_y_value], rangebreaks = [enabled = False, name = 'Test'])
#fig.update_traces(selector={"name": "Rotation Error in °"},cliponaxis = True)
#fig.update_yaxes(rangebreaks=[dict(enabled = False)])
#fig.write_image(os.path.join(path_plots,name))
fig.show()
