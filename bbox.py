#!/usr/bin/env python3
# coding=utf-8

from __future__ import absolute_import, print_function

import pandas as pd
import cv2
import numpy as np
import PyQt5 # for Mayavi
from mayavi import mlab
import math

def draw_gt_box(b, fig, color=(1,1,1), line_width=1, text_scale=(0.2,0.2,0.2), color_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    ''' 
    for i, pt in enumerate(b):
        mlab.text3d(pt[0], pt[1], pt[2], '%d' % (i), scale=text_scale, color=color, figure=fig)
    for k in range(0,4):
        # Axe X
        i,j=k,(k+1)%4
        mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

        # Axe Z
        i,j=k+4,(k+1)%4 + 4
        mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

        # Axe Y
        i,j=k,(k+1)%4 + 4
        mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    return fig

if __name__ == '__main__':
    img = cv2.imread("/media/ganoufa/GAnoufaSSD/datasets/generated_datasets/hola/test/anomaly/0.png")
    pc = pd.read_csv("/media/ganoufa/GAnoufaSSD/datasets/generated_datasets/hola/test/anomaly/0.csv", usecols=["X", "Y", "Z", "Reflectivity", "image_x", "image_y"])

    pc = np.array(pc)
    y_idx =  (pc[:, 5]*img.shape[0]).astype(int)
    x_idx =  (pc[:, 4]*img.shape[1]).astype(int)
    valid_idx = (x_idx >= 0) & (x_idx < img.shape[1]) & (y_idx >= 0) & (y_idx < img.shape[0])
    pc_rgb = np.column_stack( [pc[valid_idx, :4], img[y_idx[valid_idx], x_idx[valid_idx]]] )
    
    # from utils import visualize_colored_pointcloud
    # visualize_colored_pointcloud(pc_rgb)
    fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
    lut_idx = np.arange(len(pc_rgb))
    lut = np.column_stack([pc_rgb[:, 4:][:, ::-1], np.ones_like(pc_rgb[:, 0]) * 255])
    # plot
    p3d = mlab.points3d(pc_rgb[:, 0], pc_rgb[:, 1], pc_rgb[:, 2], lut_idx, mode='point')
    p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut_idx)
    p3d.module_manager.scalar_lut_manager.lut.table = lut


#     box = np.array([[68.3641, -0.813775, -3.16352],
# [69.0934, -2.68036, -1.33332],
# [67.7372, -2.68036, -0.792853],
# [67.0078, -0.813775, -2.62306],
# [67.968, 1.92377, -0.213691],
# [69.3242, 1.92377, -0.754156],
# [70.0536, 0.05719, 1.07605],
# [68.6973, 0.05719, 1.61651]])

    box = np.array([
[62.235, -3.06343, 1.30565],
[61.3997, -1.8945, 0.902169],
[60.4926, -1.8945, 2.77999],
[61.3279, -3.06343, 3.18347],
[60.2273, -4.03333, 2.65184],
[61.1344, -4.03333, 0.774022],
[60.2991, -2.8644, 0.370542],
[59.392, -2.8644, 2.24836],
])

    # Centers (either works)
    # print((box[1] + box[4]) /2)
    # print((box[2] + box[5]) /2)
    # print((box[3] + box[6]) /2)

    # Pour l'angle, et les dimensions
    # On suppose que l'arrête [0, 1] soit dirigée sur l'axe X (avant rotation)
    angle = math.atan2(box[1, 1] - box[0, 1], box[1, 0] - box[0, 0])
    
    length = math.sqrt((box[1, 0] - box[0, 0])**2 
        + (box[1, 1] - box[0, 1])**2 + (box[1, 2] - box[0, 2])**2)
    width = math.sqrt((box[5, 0] - box[0, 0])**2 
        + (box[5, 1] - box[0, 1])**2 + (box[5, 2] - box[0, 2])**2) 
    height = math.sqrt((box[3, 0] - box[0, 0])**2 
        + (box[3, 1] - box[0, 1])**2 + (box[3, 2] - box[0, 2])**2)

    dimensions = [length, width, height]

    print('center: ', (box[0] + box[7]) /2)
    print('dimensions: ', dimensions)
    print('angle: ', angle)

    fig = draw_gt_box(box, fig)

    
    mlab.show()
