# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 23:36:00 2023

@author: Johanna
"""

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np


import plotly.io as pio
pio.renderers.default='browser'   #'browser'

################################
path_to_eval = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\03_Moriyama_Evaluation\C033_TranslXYYawBaselineICPMoriyama.csv"
Idx_timestamp = 1 # 0 or 1 (only two timestamps available)
bool_rot = True

##########################################

if bool_rot == True:
    bool_trans = False
    

else: 
    bool_trans = True

# Read data from a csv
data = pd.read_csv(path_to_eval, delimiter = ';', header = 0, engine = 'python', encoding= 'unicode_escape')


timestamps = data.groupby('Timestamp GT Pose').groups.keys()
first_axis = data.groupby('Initial Transl x').groups.keys()
second_axis = data.groupby('Initial Transl y').groups.keys()

timestamp = list(timestamps)[Idx_timestamp]

df_eval = data.loc[data['Timestamp GT Pose'] == timestamp]

df_eval[['Initial Transl x', 'Initial Transl y', 'Initial Rot z', 'Initial Transl. Error [m]', 'Transl. Error [m]']]


#df = px.data.iris()
#fig = px.scatter_3d(df_eval, x='Initial Rot x', y='Initial Rot y', z='Initial Rot z',
#              color='Transl. Error [m]')

#fig.update_traces(marker_size =20, marker_symbol = 'square', angle = 45 )
    
#fig.update_traces()
#fig.show()

if bool_rot == True:
    value = df_eval['Rot. Error 1 [°]'] - df_eval['Initial Rot. Error 1 [°]']
else: value = df_eval['Transl. Error [m]'] - df_eval['Initial Transl. Error [m]']


fig = go.Figure(data=go.Volume(
    x= df_eval[['Initial Transl x']], #X.flatten(),
    y= df_eval[['Initial Transl y']], #Y.flatten(),
    z= df_eval[['Initial Rot z']] , #Z.flatten(),
    value= value, #values.flatten(),
    isomin=0,
    isomax=0.5,
    opacity=1, # needs to be small to see through all surfaces
    surface_count=10, # needs to be a large number for good volume rendering
    #slices_z=dict(show=True, locations=[0.4]),
    slices_x = dict(show = True, locations = [-2.0, -1.0, 0.0, 1.0, 2.0]),
    #slices_y = dict(show = True, locations = [-2.0, -1.0, 0.0, 1.0, 2.0]),
    surface=dict(fill=0.0, pattern = 'all'),
    caps= dict(x_show=False, y_show=False, z_show=False), # no caps
    ))
fig.show()


