#!/usr/bin/env python3
# coding=utf-8

from __future__ import absolute_import, print_function

import numpy as np
import PyQt5 # for Mayavi
from mayavi import mlab
import utils
import pickle
import argparse
import signal
# PC_RANGE = [0, -39.68, -1, 69.12, 39.68, 7]
PC_RANGE = [-69.12, -69.12, -1, 69.12, 69.12, 7]
def handler(signum, frame):
    exit(1)
 
# signal.signal(signal.SIGINT, handler)
if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help="dataset_path")
    args = parser.parse_args()
    dataset_path = args.dataset_path
    info_path = dataset_path +  "unigine_infos_train.pkl"
    with open(info_path, 'rb') as pickle_file:
        info = pickle.load(pickle_file)

    for idx in range(len(info)):
        fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
        vis = mlab.points3d(0, 0, 0, 0, mode='point', figure=fig)
        mlab.view(distance=25)
        print(idx)
        pc_path = dataset_path + "lidar/" + str(idx) + ".npy"
        pc = np.load(pc_path)
        # valid_idx = (pc[:, 0] >= PC_RANGE[0]) & (pc[:, 0] < PC_RANGE[3]) & (pc[:, 1] >= PC_RANGE[1]) & (pc[:, 1] < PC_RANGE[4]) & (pc[:, 2] >= PC_RANGE[2]) & (pc[:, 2] < PC_RANGE[5])
        # pc = pc[valid_idx, :]
        for box in info[idx]['gt_boxes']:
            utils.draw_gt_box(utils.boxToCorners(box), fig, color=(1, 0,0))

        vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
        mlab.show()
    
    

