# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:03:27 2023

@author: Johanna
"""

import open3d as o3d 
import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
import time


path_downtown = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Downtown\prepro\Downtown_Scan_00.pcd"
path_highway = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Highway\prepro\Highway_Scan_00.pcd"
path_rural = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Rural\prepro\Rural_Scan_00.pcd"
path_suburban = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Suburban\prepro\Suburban_Scan_00.pcd"
path_urban = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Urban\prepro\Urban_Scan_09.pcd"

pc_downtown = o3d.io.read_point_cloud(str(path_downtown))
pc_highway = o3d.io.read_point_cloud(str(path_highway))
pc_rural = o3d.io.read_point_cloud(str(path_rural))
pc_suburban = o3d.io.read_point_cloud(str(path_suburban))
pc_urban = o3d.io.read_point_cloud(str(path_urban))

# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pc_downtown)
# ctr = vis.get_view_control()
# vis.run()
# ctr.set_front((0.5,0.5,0.5))
# ctr.set_lookat((0,0,0))
# ctr.set_up((-0.5,-0.5,0.5))
# ctr.set_zoom(0.5)
# ctr.rotate(90,0.0)

o3d.visualization.draw_geometries([pc_urban],
                                  zoom=0.5,
                                  front=[0.5, 0.5, 0.5],
                                  lookat=[0,0, 0],
                                  up=[-0.5, -0.5, 0.5])

### Screen capture Shift+p

# time.sleep(1)
# vis.capture_screen_image('cameraparams.jpg', do_render = True)
# # image = vis.capture_screen_float_buffer(False)
# # plt.imsave("test.png",np.asarray(image), dpi = 300)
# # vis.run()
# time.sleep(2)
# vis.destroy_window()
