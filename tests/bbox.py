#!/usr/bin/env python3
# coding=utf-8

from __future__ import absolute_import, print_function

import pandas as pd
import cv2
import numpy as np
import PyQt5 # for Mayavi
from mayavi import mlab
import math
import utils

boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

BEV_HEIGHT = 608
BEV_WIDTH = 608
DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT

def get_filtered_lidar(lidar, labels=None):
    minX = boundary['minX']
    maxX = boundary['maxX']
    minY = boundary['minY']
    maxY = boundary['maxY']
    minZ = boundary['minZ']
    maxZ = boundary['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
                    (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
                    (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
    lidar = lidar[mask]
    lidar[:, 2] = lidar[:, 2] - minZ

    if labels is not None:
        label_x = (labels[:, 1] >= minX) & (labels[:, 1] < maxX)
        label_y = (labels[:, 2] >= minY) & (labels[:, 2] < maxY)
        label_z = (labels[:, 3] >= minZ) & (labels[:, 3] < maxZ)
        mask_label = label_x & label_y & label_z
        labels = labels[mask_label]
        return lidar, labels
    else:
        return lidar

def makeBEVMap(PointCloud_):
    Height = BEV_HEIGHT + 1
    Width = BEV_WIDTH + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / DISCRETIZATION))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / DISCRETIZATION) + Width / 2)

    # sort-3times
    sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[sorted_indices]
    _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[unique_indices]

    # Height Map, Intensity Map & Density Map
    heightMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[:BEV_HEIGHT, :BEV_WIDTH]  # r_map
    RGB_Map[1, :, :] = heightMap[:BEV_HEIGHT, :BEV_WIDTH]  # g_map
    RGB_Map[0, :, :] = intensityMap[:BEV_HEIGHT, :BEV_WIDTH]  # b_map

    return RGB_Map

# Corners à partir d'une box sous format kitti
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # front left
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw
    return bev_corners

def drawRotatedBox(img, x, y, w, l, yaw, color):
    bev_corners = get_corners(x, y, w, l, yaw)
    print(bev_corners)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2)
    cv2.line(img, (int(corners_int[0, 0]), int(corners_int[0, 1])), (int(corners_int[3, 0]), int(corners_int[3, 1])), (255, 255, 0), 2)


if __name__ == '__main__':
    img = cv2.imread("/media/ganoufa/GAnoufaSSD/datasets/generated_datasets/test_box/test/anomaly/1.png")
    pc = pd.read_csv("/media/ganoufa/GAnoufaSSD/datasets/generated_datasets/test_box/test/anomaly/1.csv", usecols=["X", "Y", "Z", "Reflectivity", "image_x", "image_y"])

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

    box = np.array([
[40.679427470233804, 2.7246839143590478, 1.9406961526616442],
[40.549275185775059, 4.6592936497373785, 1.8973120668891852],
[38.772854061046473, 4.5265047539841134, 1.305171815460767],
[38.903006345505219, 2.5918950186056691, 1.348555901233226],
[41.077143542904651, 2.7246839143590478, 0.74754776080192187],
[40.946991258445905, 4.6592936497373785, 0.70416367502946287],
[39.17057013371732, 4.5265047539841134, 0.11202342360104467],
[39.300722418176065, 2.5918950186056691, 0.15540750937350367],
    ])


    # Centers (either works)
    # print((box[1] + box[4]) /2)
    # print((box[2] + box[5]) /2)
    # print((box[3] + box[6]) /2)

    # Pour l'angle, et les dimensions
    # On suppose que l'arête [0, 1] soit dirigée sur l'axe X (avant rotation)
    angle = math.atan2(box[1, 1] - box[0, 1], box[1, 0] - box[0, 0])
    
    length = math.sqrt((box[1, 0] - box[0, 0])**2 
        + (box[1, 1] - box[0, 1])**2 + (box[1, 2] - box[0, 2])**2)
    width = math.sqrt((box[5, 0] - box[0, 0])**2 
        + (box[5, 1] - box[0, 1])**2 + (box[5, 2] - box[0, 2])**2) 
    height = math.sqrt((box[3, 0] - box[0, 0])**2 
        + (box[3, 1] - box[0, 1])**2 + (box[3, 2] - box[0, 2])**2)

    dimensions = [length, width, height]
    center = (box[0] + box[7]) /2

    print('center: ', center)
    print('dimensions: ', dimensions)
    print('angle: ', angle)

    # get_corners()
    filtered_pc = get_filtered_lidar(pc)
    bev = makeBEVMap(filtered_pc)
    bev = cv2.UMat(np.moveaxis(bev, [0, 1, 2], [2, 0, 1]))

    print(type(bev))
    print(type(img))
    drawRotatedBox(bev, center[0], center[1], width, length, angle, (0, 255, 255))
    cv2.imshow("bev", bev)
    fig = utils.draw_gt_box(box, fig)

    mlab.show()
