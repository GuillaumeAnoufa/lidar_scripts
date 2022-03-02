#!/usr/bin/env python3
# coding=utf-8

from __future__ import absolute_import, print_function

import argparse
import os
import sys

import pandas as pd
import cv2
import numpy as np

import utils
import algos.projections as proj

if __name__ == '__main__':

    # img = cv2.imread(args.img)
    # pc = utils.load_pc(args.pcd)

    img = cv2.imread("/media/ganoufa/GAnoufaSSD/datasets/ung_pictures/test1/2.png")
    pc = pd.read_csv("/media/ganoufa/GAnoufaSSD/datasets/ung_pictures/test1/2.csv", usecols=["X", "Y", "Z", "Reflectivity", "image_x", "image_y"])

    # print( (pc["image_x"]*img.shape[1]).astype(int) )

    # y_idx = max( (pc["image_y"]*img.shape[0]).astype(int), img.shape[0] - 1)
    # x_idx = max( (pc["image_x"]*img.shape[1]).astype(int), img.shape[1] - 1)

    pc = np.array(pc)
    y_idx =  (pc[:, 5]*img.shape[0]).astype(int)
    x_idx =  (pc[:, 4]*img.shape[1]).astype(int)

    # x_idx = x_idx[ (x_idx >= 0) & (x_idx < img.shape[1]) ]
    valid_idx = (x_idx >= 0) & (x_idx < img.shape[1]) & (y_idx >= 0) & (y_idx < img.shape[0])
    pc_rgb = np.column_stack( [pc[valid_idx, :4], img[y_idx[valid_idx], x_idx[valid_idx]]] )
    
    print(pc_rgb.shape)

    utils.visualize_colored_pointcloud(pc_rgb)


