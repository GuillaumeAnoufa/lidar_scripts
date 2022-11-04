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
PC_RANGE = [-69.12, -69.12, -1, 69.12, 69.12, 7]

if __name__ == '__main__':
    dataset_path = "/media/ganoufa/GAnoufaSSD/datasets/generated_datasets/new_misc/"
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
    print("gt_boxes[{}] = {}".format(sample_idx, info[sample_idx]['gt_boxes']))
    for box in info[sample_idx]['gt_boxes']:
        utils.draw_gt_box(utils.boxToCorners(box), fig, color=(1, 0,0))
    
    # for corners in info[sample_idx]['gt_corners']:
    #     utils.draw_gt_box(corners, fig, color=(1, 1,1))
        
    db_info_path = dataset_path +  "unigine_dbinfos_train.pkl"
    with open(db_info_path, 'rb') as pickle_file:
        db_info = pickle.load(pickle_file)
    
    # print("db_info ", db_info)

    vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
    mlab.view(distance=25)
    mlab.show()
    
    

