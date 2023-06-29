# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:05:01 2023

@author: Johanna
"""

import plotly.graph_objects as go

import pandas as pd
import numpy as np

import plotly.io as pio
pio.renderers.default='browser'   #'browser'

#########################################
### Choose file with x2x name
path_to_eval = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\03_Moriyama_Evaluation\C123_TranslXYBaselineICPMoriyama.csv"
### Choose which timestamp to evaluate
Idx_timestamp = 3
### Choose if transl. error or rotation error should be displayed
bool_rot = True

##########################################

if bool_rot == True:
    bool_trans = False
    
    title = 'Delta Rot. Error in °'
    title_z = 'Delta Rot. Error / °'
else: 
    bool_trans = True
    title = 'Delta Transl. Error in m'
    title_z = 'Delta Transl. Error / m'



# Read data from a csv
data = pd.read_csv(path_to_eval, delimiter = ';', header = 0, engine = 'python', encoding= 'unicode_escape', )

data = data.fillna(0.0)

timestamps = data.groupby('Timestamp GT Pose').groups.keys()
first_axis = data.groupby('Initial Transl x').groups.keys()

timestamp = list(timestamps)[Idx_timestamp]

df_eval = data.loc[data['Timestamp GT Pose'] == timestamp]


z_data = np.zeros((3,len(list(first_axis)), len(list(first_axis))))

for i in range(0, len(list(first_axis))):
    df = df_eval.loc[df_eval['Initial Transl x'] == list(first_axis)[i]]
    df.reset_index()
    for j in range(0,len(df)):
        z_data[0,i,j] = df.iloc[j]["Transl. Error [m]"] - df.iloc[j]["Initial Transl. Error [m]"]
        z_data[1,i,j] = df.iloc[j]["Rot. Error 1 [°]"] - df.iloc[j]["Initial Rot. Error 1 [°]"]
        z_data[2,i,j] = df.iloc[j]["RMSE Inliers"]
 

z1 = z_data[0]
z2 = z_data[1]
z3 = z_data[2]

cols = ['-2.0', '-1.5', '-1.0', '-0.5',  '0.0',
       '0.5',  '1.0',  '1.5',  '2.0']

idx = [-2,-1.5, -1.0, -0.5,0.0,0.5,1.0,1.5,2.0]

#z1 = pd.DataFrame(z1, columns = cols, index = idx)
#z2 = pd.DataFrame(z1, columns = cols, index = idx)


x, y = np.linspace(-2, 2, 8), np.linspace(-2, 2, 8)

if bool_rot == True:
    z = z2
else: z = z1

fig = go.Figure( go.Surface(x = x, y = y, z=z, colorscale='Viridis', lighting=dict(specular=0.2)))
    #go.Surface(x = x, y = y, z=z2, showscale=True, opacity=1.0),
    #go.Surface(x = x, y = y, z=z3, showscale=True, opacity=1.0)])

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0, z=1)
)



fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))

fig.update_layout(title=title, autosize=True,
                   scene_camera = camera,
                  margin=dict(l=65, r=50, b=65, t=90),
                  scene = dict(
                    xaxis = dict(
                        title = 'Offset in x / m',
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor="white",
                         showbackground=False,
                         zerolinecolor="black",),
                    yaxis = dict(
                        title = 'Offset in y / m',
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=False,
                        zerolinecolor="black"),
                    zaxis = dict(
                        title = title_z,
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="black",)))


fig.show()

