import time

import numpy as np, sys, os, cv2, yaml
import PyQt5 # for Mayavi
# import pandas as pd
import utils

sys.path.append('/home/ganoufa/workSpace/lidar/catkin_ws/src/lidar_camera_processing/scripts/algos/')
import projections as proj

configs = yaml.load(open("/home/ganoufa/workSpace/lidar/catkin_ws/src/lidar_camera_processing/calibration/calibration.yaml", 'r').read(), Loader=yaml.FullLoader)
root = configs['data']['root']
intrinsic_matrix = np.loadtxt(os.path.join(root, 'intrinsic'))
distortion = np.loadtxt(os.path.join(root, 'distortion'))
extrinsic_matrix = np.loadtxt(os.path.join(root, 'extrinsic'))


if __name__ == '__main__':

    livox_example = 'data/1.npy' # npy load is faster (by far (1000x))
    pc = utils.load_pc_npy(livox_example)
    print(pc.shape)

    img = cv2.imread("data/1.png")

    projection_points = proj.undistort_projection(pc[:, :3], intrinsic_matrix, extrinsic_matrix)
    projection_points_normal = np.column_stack([np.squeeze(projection_points), pc[:, 3:]])

    projection_points_test = np.column_stack([np.squeeze(projection_points), pc[:, 3:]])


    # crop
    projection_points_normal = projection_points_normal[np.where(
        (projection_points_normal[:, 0] > 0) &
        (projection_points_normal[:, 0] < img.shape[1]) &
        (projection_points_normal[:, 1] > 0) &
        (projection_points_normal[:, 1] < img.shape[0])
    )]

    # projection_points : (y_img, x_img, X,_pc, Y_pc, Z_pc, Reflectivity)

    # depth map projection
    depth_map = np.zeros_like(img, dtype=np.float32)
    # depth_map[np.int_(projection_points[:, 1]), np.int_(projection_points[:, 0]), 0] = 1#projection_points[:, 2]
    # depth_map[np.int_(projection_points[:, 1]), np.int_(projection_points[:, 0]), 1] = 2#projection_points[:, 3]
    # Todo attribuer une couleur en fct de la distance

    print(depth_map[:100, 100])

    print(depth_map.shape)


    cv2.imshow("depth_map", depth_map)
    cv2.waitKey(0)