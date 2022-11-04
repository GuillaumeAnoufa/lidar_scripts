#!/usr/bin/env python3
# coding=utf-8
from __future__ import absolute_import, print_function

from numba.core.errors import (NumbaDeprecationWarning, 
                                NumbaPendingDeprecationWarning,
                                NumbaPerformanceWarning)
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

import pandas as pd
import numpy as np
import PyQt5 # for Mayavi
from mayavi import mlab
import tools.utils as utils
import pickle
import argparse
import numba
import sys
import torch
import time
from pytorch3d.ops import box3d_overlap as box3d_overlap_pytorch3d

sys.path.append('/home/ganoufa/workSpace/deep_learning/lidar_detection/pvrcnn_agnostic/pcdet_agn/datasets/kitti/kitti_object_eval_python/')
from rotate_iou import rotate_iou_gpu_eval

sys.path.append('/home/ganoufa/workSpace/deep_learning/lidar_detection/pvrcnn_agnostic/pcdet_agn/ops/iou3d_nms')
from iou3d_nms_utils import boxes_iou3d_gpu

@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0

def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]], qboxes[:, [0, 2, 3, 5, 6]], 2)
    # rotation around Z instead of Y
    # Criterion 2 = area
    # rinc = rotate_iou_gpu_eval(boxes[:, [0, 1, 3, 4, 6]], qboxes[:, [0, 1, 3, 4, 6]], -1)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc

PC_RANGE = [0, -39.68, -1, 69.12, 39.68, 7]
if __name__ == '__main__':
    dataset_path = "/media/ganoufa/GAnoufaSSD/datasets/generated_datasets/new_misc/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=2, help='image idx')
    args = parser.parse_args()
    sample_idx = args.idx
    pred_path = "/home/ganoufa/workSpace/deep_learning/lidar_detection/pvrcnn_agnostic/output/unigine_models"
    pred_path += "/pointpillar3/default/eval/epoch_50/val/default/result.pkl"
    with open(pred_path, 'rb') as pickle_file:
        pred = pickle.load(pickle_file)
    
    pc_path = dataset_path + "lidar/" + str(sample_idx) + ".npy"
    info_path = dataset_path +  "unigine_infos_train.pkl"
    fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
    
    with open(info_path, 'rb') as pickle_file:
        info = pickle.load(pickle_file)
    pc = np.load(pc_path)
    valid_idx = (pc[:, 0] >= PC_RANGE[0]) & (pc[:, 0] < PC_RANGE[3]) & (pc[:, 1] >= PC_RANGE[1]) & (pc[:, 1] < PC_RANGE[4]) & (pc[:, 2] >= PC_RANGE[2]) & (pc[:, 2] < PC_RANGE[5])
    pc = pc[valid_idx, :]
    gt_boxes = info[sample_idx]['gt_boxes']
    dt_boxes = pred[sample_idx]['boxes_lidar']
    
    # Methode 1
    start=time.perf_counter()
    ret = d3_box_overlap(gt_boxes, dt_boxes, criterion=-1).astype(np.float64)
    end=time.perf_counter()
    print("Méthode 1 | d3_box_overlap ({:.3f}ms) : \n IOU = {}".format( (end-start)*1000, ret))

    # Methode 2
    start=time.perf_counter()
    gt_corners = torch.tensor(np.stack([utils.boxToCorners(gt_box) for gt_box in gt_boxes], axis=0), dtype=torch.float32)
    dt_corners =  torch.tensor(np.stack([utils.boxToCorners(dt_box) for dt_box in dt_boxes], axis=0), dtype=torch.float32)
    intersection_vol, iou_3d = box3d_overlap_pytorch3d(gt_corners, dt_corners)
    end=time.perf_counter()
    print("Méthode 2 | pytorch3d ({:.3f}ms) : \n IOU = {}".format( (end-start)*1000, iou_3d))
    
    # Methode 3
    start=time.perf_counter()
    iou_3d_boxes = boxes_iou3d_gpu(torch.from_numpy(gt_boxes).cuda(), torch.from_numpy(dt_boxes).cuda())
    end=time.perf_counter()
    print("Méthode 3 | boxes_iou3d_gpu({:.3f} ms) : \n IOU = {}".format( (end-start)*1000, iou_3d_boxes))
    
    gt_color = (1, 1, 1)
    dt_color = (1, 0, 0)
    # draw gt
    for box in info[sample_idx]['gt_boxes']:
        fig = utils.draw_gt_box(utils.boxToCorners(box), fig, color=gt_color)
    for box in pred[sample_idx]['boxes_lidar']:
        fig = utils.draw_gt_box(utils.boxToCorners(box), fig, color=dt_color)
        
    for i in range(gt_corners.shape[0]):
        pt = gt_corners[i][0]
        mlab.text3d(pt[0], pt[1], pt[2], '%d' % (i), scale=(0.8,0.8,0.8), color=gt_color, figure=fig) 
        
    for i in range(dt_corners.shape[0]):
        pt = dt_corners[i][0]
        mlab.text3d(pt[0], pt[1], pt[2], '%d' % (i), scale=(0.8,0.8,0.8), color=dt_color, figure=fig)

    vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
    mlab.view(distance=25)
    mlab.show()