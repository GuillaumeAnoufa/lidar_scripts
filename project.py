import time

import numpy as np, sys, os, cv2, yaml
import PyQt5 # for Mayavi
# import pandas as pd
import utils

import ground_removal_ext

# sys.path.append('/home/ganoufa/workSpace/lidar/catkin_ws/src/lidar_camera_processing/scripts/algos/')
sys.path.append('/home/ganoufa/workSpace/lidar/useful_scripts/')
import algos.projections as proj

configs = yaml.load(open("/home/ganoufa/workSpace/lidar/catkin_ws/src/lidar_camera_processing/calibration/calibration.yaml", 'r').read(), Loader=yaml.FullLoader)
root = configs['data']['root']
intrinsic_matrix = np.loadtxt(os.path.join(root, 'intrinsic'))
distortion = np.loadtxt(os.path.join(root, 'distortion'))
extrinsic_matrix = np.loadtxt(os.path.join(root, 'extrinsic'))

def project_pc_to_img(pc, img):
    view_img = np.copy(img)
    # 8ms
    view_img = cv2.undistort(view_img, intrinsic_matrix, distortion)
    view_img = cv2.cvtColor(view_img, cv2.COLOR_BGR2RGB)

    # 60ms -> peut-être ne pas projeter toute l'image ? Seulement dans la zone de l'image interessante
    board = proj.pc_to_img(pc, view_img, extrinsic_matrix, intrinsic_matrix)
    cv2.imshow('Projection', board)
    cv2.waitKey(0)
    return board

def project_img_to_pc(pc, img):
    img = cv2.undistort(img, intrinsic_matrix, distortion)
    pc = ground_removal_ext.ground_removal_kernel(pc, 0.2, 200)

    # 60ms -> peut-être ne pas projeter toute l'image ? Seulement dans la zone de l'image interessante
    pc = proj.img_to_pc(pc, img, extrinsic_matrix, intrinsic_matrix)
    utils.visualize_colored_pointcloud(pc)

if __name__ == '__main__':

    # For perf, convert pcd to npy. npy load is faster (by far (1000x))
    # pcd_path = "/media/ganoufa/GAnoufaSSD/datasets/vols_24_02/record3/197.pcd"
    # img_path = "/media/ganoufa/GAnoufaSSD/datasets/vols_24_02/record3/197.png"
    pcd_path = "/media/ganoufa/GAnoufaSSD/datasets/vols_24_02/record2/2644.pcd"
    img_path = "/media/ganoufa/GAnoufaSSD/datasets/vols_24_02/record2/2644.png"

    pc = utils.load_pc(pcd_path)
    img = cv2.imread(img_path)
    print(img.shape)

    # En python cv2 ellipse prend a/2 et b/2 pour une raison inconnue ...
    # img = cv2.ellipse(img, (677, 926), (465, 254), 174.821, 0, 360, (255,0,0), 2)
    img = cv2.ellipse(img, (619, 591), (390, 147), 176.938, 0, 360, (255,0,0), 6)

    cv2.imshow('img', img)
    cv2.imwrite("2644.png", img)

    alpha = 1. # [1, 3]
    beta = 40 # [0, 100]
    # new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # cv2.imshow("img", img)
    # cv2.imshow("new_img", new_img)
    project_img_to_pc(pc, img)
    