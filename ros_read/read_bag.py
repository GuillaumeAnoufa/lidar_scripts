#!/usr/bin/env python3
# coding=utf-8

from __future__ import absolute_import, print_function, division

import argparse
from os import preadv
import sys

import cv2
import numpy as np
from cv_bridge import CvBridge
import rosbag
import ros_numpy
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
import threading
import PyQt5 # for Mayavi
from mayavi import mlab
import pcl.pcl_visualization

global_pc = np.empty([1, 4])
global_img = None

def read_camera_bag(bag_name):
    global global_img
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    camera_bag = rosbag.Bag(bag_name)
    for topic, msg, t in camera_bag.read_messages(topics=['/camera/image_raw/compressed']):
        global_img = bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
        cv2.imshow("preview", global_img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

def read_lidar_bag(bag_name):
    global global_pc
    lidar_bag = rosbag.Bag(bag_name)
    for topic, msg, t in lidar_bag.read_messages(topics=['/livox/to_integrate']):
        # Ros time is in µsec
        msg.__class__ = PointCloud2
        pc_array = ros_numpy.numpify(msg)
        # parsing point cloud
        pc = np.zeros((pc_array.shape[0], 4))
        pc[:, 0] = pc_array['x']
        pc[:, 1] = pc_array['y']
        pc[:, 2] = pc_array['z']
        pc[:, 3] = pc_array['intensity']
        # print("new_data {}, time = {}, last_time = {}".format(pc.shape, t.nsecs, prev))
        global_pc = np.concatenate([global_pc, pc])[-len(pc) * 2:]


if __name__ == '__main__':
    print('Init......')
    bridge = CvBridge()

    # for _, msg, t in camera_bag.read_messages(topics=['/camera/image_raw/compressed']):
    #     global_img = bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
    #     print("T = {}".format(t))
    #     cv2.imshow("img", global_img)
    #     cv2.waitKey(1)

    # for topic, msg, t in lidar_bag.read_messages(topics=['/livox/to_integrate']):
    #     msg.__class__ = PointCloud2
    #     pc_array = ros_numpy.numpify(msg)
    #     # parsing point cloud
    #     pc = np.zeros((pc_array.shape[0], 4))
    #     pc[:, 0] = pc_array['x']
    #     pc[:, 1] = pc_array['y']
    #     pc[:, 2] = pc_array['z']
    #     pc[:, 3] = pc_array['intensity']
    #     print("hello")

    #     fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
    #     print(pc.shape)
    #     if pc.shape[1] == 4:
    #         vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
    #     else:
    #         vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], 0., mode='point', figure=fig)

    #     mlab.view(distance=25)
    #     mlab.show()


    #### LIDAR THREAD
    bag_name = "/media/ganoufa/GAnoufaSSD/vols_24_02/camera_lidar_vol_0.bag"
    lidarThread = threading.Thread(target=read_lidar_bag, args=(bag_name,), daemon=True)


    #### CAM THREAD
    # cameraThread = threading.Thread(target=read_camera_bag, args=(bag_name,), daemon=True)
    # cameraThread.start()

    lidarThread.start()
    visual = pcl.pcl_visualization.CloudViewing()
    visual.ShowMonochromeCloud(pcl.PointCloud(global_pc, 'XYZI'), b'cloud')

    v = True
    while v:
        v = not(visual.WasStopped())

    # fig = mlab.figure('pc', size=(960,540), bgcolor=(0.05, 0.05, 0.05))
    # vis = mlab.points3d(0, 0, 0, 0, mode='point', figure=fig)
    # mlab.view(distance=25)

    # @mlab.animate(delay=10)
    # def anim():
    #     while True:
    #         vis.mlab_source.reset(x=global_pc[:, 0], y=global_pc[:, 1], z=global_pc[:, 2], scalars=global_pc[:, 3])
    #         # mlab.savefig(f'{i}.png', figure=fig)
    #         yield
    #     mlab.close(all=True)
    # anim()
    # mlab.show()

