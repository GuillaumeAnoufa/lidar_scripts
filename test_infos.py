#!/usr/bin/env python3
# coding=utf-8

from __future__ import absolute_import, print_function

import pandas as pd
import numpy as np
import PyQt5 # for Mayavi
from mayavi import mlab
import utils
import pickle
import argparse
PC_RANGE = [0, -39.68, -1, 69.12, 39.68, 7]

def boxToCorners(box):#center, dims, angle):
    """
    Args:
        pred_box: List (cx, cy, cz, l, w, h, rz)
    Returns:
        corners_3d: (8, 3) corners of box3d
    """
    center = box[0:3]
    l, w, h = box[3:6]
    angle = box[6]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]

    # R = np.array([[np.cos(angle), 0, np.sin(angle)],
    #                 [0, 1, 0],
    #                 [-np.sin(angle), 0, np.cos(angle)]])
    R = np.stack([[np.cos(angle), np.sin(angle), 0],
                        [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    corners3d = np.dot(R, corners3d).T
    corners3d = corners3d + center
    return corners3d

if __name__ == '__main__':
    dataset_path = "/media/ganoufa/GAnoufaSSD/datasets/generated_datasets/unigine4/"
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int, help="index")
    args = parser.parse_args()
    sample_idx = args.index
    
    pc_path = dataset_path + "lidar/" + str(sample_idx) + ".npy"
    info_path = dataset_path +  "unigine_infos_train.pkl"
    with open(info_path, 'rb') as pickle_file:
        info = pickle.load(pickle_file)
    pc = np.load(pc_path)
    valid_idx = (pc[:, 0] >= PC_RANGE[0]) & (pc[:, 0] < PC_RANGE[3]) & (pc[:, 1] >= PC_RANGE[1]) & (pc[:, 1] < PC_RANGE[4]) & (pc[:, 2] >= PC_RANGE[2]) & (pc[:, 2] < PC_RANGE[5])
    pc = pc[valid_idx, :]

    print(info[sample_idx])
    fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))

    # draw gt
    for box in info[sample_idx]['gt_boxes']:
        fig = utils.draw_gt_box(boxToCorners(box), fig, color=(1, 0,0))

    vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
    mlab.view(distance=25)
    mlab.show()
    
    
