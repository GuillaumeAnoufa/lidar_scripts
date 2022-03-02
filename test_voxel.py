import time

import numpy as np, sys, os, cv2, yaml
import PyQt5 # for Mayavi
# import pandas as pd
import utils

sys.path.append('/home/ganoufa/workSpace/lidar/catkin_ws/src/lidar_camera_processing/scripts/algos/')
import projections as proj

DX = 0.2
DY = 0.2
DZ = 0.2

X_MIN = -89.6
X_MAX = 89.6

Y_MIN = -22.4
Y_MAX = 22.4

Z_MIN = -3.0
Z_MAX = 3.0

GPU_INDEX = 0
NMS_THRESHOLD = 0.1
BOX_THRESHOLD = 0.6
overlap = 11.2
HEIGHT = round((X_MAX - X_MIN+2*overlap) / DX)
WIDTH = round((Y_MAX - Y_MIN) / DY)
CHANNELS = round((Z_MAX - Z_MIN) / DZ)


def data2voxel(pclist):

    data = [i*0 for i in range(HEIGHT*WIDTH*CHANNELS)]

    for line in pclist:
        X = float(line[0])
        Y = float(line[1])
        Z = float(line[2])
        if( Y > Y_MIN and Y < Y_MAX and
            X > X_MIN and X < X_MAX and
            Z > Z_MIN and Z < Z_MAX):
            channel = int((-Z + Z_MAX)/DZ)
            if abs(X)<3 and abs(Y)<3:
                continue
            if (X > -overlap):
                pixel_x = int((X - X_MIN + 2*overlap)/DX)
                pixel_y = int((-Y + Y_MAX)/DY)
                data[pixel_x*WIDTH*CHANNELS+pixel_y*CHANNELS+channel] = 1
            if (X < overlap):
                pixel_x = int((-X + overlap)/DX)
                pixel_y = int((Y + Y_MAX)/DY)
                data[pixel_x*WIDTH*CHANNELS+pixel_y*CHANNELS+channel] = 1
    voxel = np.reshape(data, (HEIGHT, WIDTH, CHANNELS))
    return voxel


if __name__ == '__main__':

    livox_example = 'data/1.npy' # npy load is faster (by far (1000x))
    pc = utils.load_pc_npy(livox_example)
    print(pc.shape)

    pc_list = pc.tolist()

    voxel = data2voxel(pc_list)
    print(voxel.shape)



