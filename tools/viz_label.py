#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import PyQt5 # for Mayavi
from mayavi import mlab
import utils
import os
import argparse
import signal
import glob
from pathlib import Path

def handler(signum, frame):
    exit(1)

if __name__ == '__main__':
    dataset_path = "/media/ganoufa/GAnoufaSSD/datasets/generated_datasets/unigineGround1/label/0.npy"
    # parser = argparse.ArgumentParser()
    # parser.add_argument('index', type=int, help="index")
    # args = parser.parse_args()
    # sample_idx = args.index
    
    # box_path = dataset_path + "box_" + str(sample_idx) + ".npy"

    # # base_pc = utils.load_pc("/media/ganoufa/GAnoufaSSD/datasets/generated_datasets/unigine7/lidar/" + str(sample_idx) + ".npy")

    # boxes = np.load(box_path)
    # fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
    # for file in os.listdir(dataset_path):
    #     if "pc" in file and str(sample_idx) in file:
    #         pc = utils.load_pc(dataset_path+file)
    #         break
    # # draw gt
    # for box in boxes:
    #     fig = utils.draw_gt_box(utils.boxToCorners(box), fig, color=(1, 0,0))

    # vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
    # mlab.view(distance=25)
    # mlab.show()
    
    signal.signal(signal.SIGINT, handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="file path")
    args = parser.parse_args()
    file_path = Path(args.file_path)
    data_file_list = glob.glob(str(file_path / f'*.npy')) if file_path.is_dir() else [file_path]
    
    for file in data_file_list:
        pc = utils.load_pc(str(file))
        print(pc.shape)
        utils.viz_labels(pc)
        mlab.show()