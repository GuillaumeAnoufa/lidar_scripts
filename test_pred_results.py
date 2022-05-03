#!/usr/bin/env python3
# coding=utf-8

from __future__ import absolute_import, print_function

import pandas as pd
import numpy as np
import PyQt5 # for Mayavi
from mayavi import mlab
import math
import utils
import pickle

class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.cls_type = label[0]
        self.corners = np.array(label[1:], dtype=np.float32).reshape([8, 3])
        self.cornersToBox()
    
    def cornersToBox(self):
        box = self.corners
        angle = math.atan2(box[1, 1] - box[2, 1], box[1, 0] - box[2, 0])
        dim = np.max(box, axis=0) - np.min(box, axis=0) # max sur les colonnes
        center = (box[0] + box[6]) /2
        self.gt_box = np.concatenate([center, dim, angle], axis=None)
        

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
    pc_path = "/media/ganoufa/GAnoufaSSD/datasets/generated_datasets/unigine3/lidar/15.npy"
    pred_path = "/home/ganoufa/workSpace/deep_learning/lidar_detection/OpenPCDet/output/unigine_models/pv_rcnn/default/eval/epoch_30/val/default/result.pkl"
    pred = []
    with open(pred_path, 'rb') as pickle_file:
        pred = pickle.load(pickle_file)
    pc = np.load(pc_path)
    pc[:, 3] = pc[:, 3] / 255

    idx = int("".join(pc_path.split('/')[-1]).replace('.npy', ''))
    
    label_file = pc_path.replace('.npy', '.txt').replace('lidar', 'label')
    with open(label_file) as f:
        lines = f.readlines()
        objects = [Object3d(line) for line in lines]

    fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))

    # draw gt
    for obj in objects:
        fig = utils.draw_gt_box(obj.corners, fig, color=(1, 0,0))

    # draw pred
    for pred_box in pred[idx]['boxes_lidar']:
        pred_corners = boxToCorners(pred_box)
        fig = utils.draw_gt_box(pred_corners, fig, color=(1, 1,1))

    vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
    mlab.view(distance=25)
    mlab.show()
    
    

