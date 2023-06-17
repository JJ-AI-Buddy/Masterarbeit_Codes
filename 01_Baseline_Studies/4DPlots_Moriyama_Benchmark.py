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

path_to_eval = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\Baseline\03_Moriyama_Evaluation\C030_TranslXYYawBaselineICPMoriyama.csv"


# Read data from a csv
data = pd.read_csv(path_to_eval, delimiter = ';', header = 0, engine = 'python', encoding= 'unicode_escape')


Idx_timestamp = 0

timestamps = data.groupby('Timestamp GT Pose').groups.keys()
first_axis = data.groupby('Initial Rot x').groups.keys()
second_axis = data.groupby('Initial Rot y').groups.keys()

timestamp = list(timestamps)[Idx_timestamp]

df_eval = data.loc[data['Timestamp GT Pose'] == timestamp]

df_eval[['Initial Rot x', 'Initial Rot y', 'Initial Rot z', 'Initial Transl. Error [m]', 'Transl. Error [m]']]


#df = px.data.iris()
#fig = px.scatter_3d(df_eval, x='Initial Rot x', y='Initial Rot y', z='Initial Rot z',
              color='Transl. Error [m]')

#fig.update_traces(marker_size =20, marker_symbol = 'square', angle = 45 )
    
#fig.update_traces()
#fig.show()



fig = go.Figure(data=go.Volume(
    x= df_eval[['Initial Rot x']], #X.flatten(),
    y= df_eval[['Initial Rot y']], #Y.flatten(),
    z= df_eval[['Initial Rot z']] , #Z.flatten(),
    value= df_eval['Transl. Error [m]'] - df_eval['Initial Transl. Error [m]'], #values.flatten(),
    isomin=-2,
    isomax=2,
    opacity=1, # needs to be small to see through all surfaces
    surface_count=10, # needs to be a large number for good volume rendering
    #slices_z=dict(show=True, locations=[0.4]),
    #slices_x = dict(show = True, locations = [-2.0,-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),
    #surface=dict(fill=0.0, pattern = 'all'),
    caps= dict(x_show=False, y_show=False, z_show=False), # no caps
    ))
fig.show()


