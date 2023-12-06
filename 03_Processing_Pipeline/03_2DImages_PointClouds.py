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
import pyautogui

print(o3d.__version__)

path = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Urban\prepro\*.pcd"
path_img = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Urban\Images"

# path_downtown = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Downtown\prepro\Downtown_Scan_00.pcd"
# path_highway = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Highway\prepro\Highway_Scan_00.pcd"
# path_rural = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Rural\prepro\Rural_Scan_00.pcd"
# path_suburban = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Suburban\prepro\Suburban_Scan_00.pcd"
# path_urban = r"C:\Users\Johanna\OneDrive - bwedu\Masterarbeit_OSU\05_Data_Training\Dataset_Validation\Urban\prepro\Urban_Scan_09.pcd"

# pc_downtown = o3d.io.read_point_cloud(str(path_downtown))
# pc_highway = o3d.io.read_point_cloud(str(path_highway))
# pc_rural = o3d.io.read_point_cloud(str(path_rural))
# pc_suburban = o3d.io.read_point_cloud(str(path_suburban))
# pc_urban = o3d.io.read_point_cloud(str(path_urban))

# list_pc = [pc_downtown, pc_highway, pc_rural, pc_suburban, pc_urban]
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

# o3d.visualization.draw_geometries([pc_urban],
#                                   zoom=0.5,
#                                   front=[0.5, 0.5, 0.5],
#                                   lookat=[0,0, 0],
#                                   up=[-0.5, -0.5, 0.5])



### Screen capture Shift+p

# time.sleep(1)
# vis.capture_screen_image('cameraparams.jpg', do_render = True)
# # image = vis.capture_screen_float_buffer(False)
# # plt.imsave("test.png",np.asarray(image), dpi = 300)
# # vis.run()
# time.sleep(2)
# vis.destroy_window()


# width = 500
# height = 500
# fx = 1
# fy = 1
# cx = 0.5
# cy = 0.5

# x = 250
# y = 250
# z = 1

# # Example intrinsic parameters
# intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# K = [[fx * width, 0, width / 2 - 0.5],
#      [0, fy * width, height / 2 - 0.5],
#      [0, 0, 1]]

# intrinsic.intrinsic_matrix = K



# # Example extrinsic parameters (viewpoint)
# params = o3d.camera.PinholeCameraParameters()
# params.extrinsic = np.array([[1, 0, 0, x],
#                                 [0, 1, 0, y],
#                                 [0, 0, 1, z],
#                                 [0, 0, 0, 1]])

# params.intrinsic = intrinsic

for file in glob.glob(path):
    print (file)
    
    filename = os.path.basename(file)
    image = filename.replace('.pcd','.jpg')
    pcd = o3d.io.read_point_cloud(str(file))
    
    save_path = os.path.join(path_img,image)
   
    
    #pcd = pc_urban
    vis = o3d.visualization.Visualizer()
    ctr = vis.get_view_control()
    vis.create_window(width=224, height=224, visible = True)
    vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
    vis.get_render_option().point_size = 2.0
    vis.add_geometry(pcd)
    #vis.get_view_control().convert_from_pinhole_camera_parameters(params,allow_arbitrary=True)
    ctr = vis.get_view_control()
    ctr.set_front((0.5,0.5,0.5))
    ctr.set_lookat((0,0,0))
    ctr.set_up((-0.5,-0.5,0.5))
    ctr.set_zoom(0.5)
    #ctr.rotate(90,0.0)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.run()
    vis.capture_screen_image(str(save_path), do_render=False)
    vis.clear_geometries()
    vis.destroy_window()
    
    #time.sleep(1)
    #pyautogui.keyDown('esc')
