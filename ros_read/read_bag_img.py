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

global_pc = np.empty([1, 4])
global_img = None


if __name__ == '__main__':
    print('Init......')
    bridge = CvBridge()

    bag_name = "/home/ganoufa/data/rosbags/camera_2021-12-10-16-46-42.bag"
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    camera_bag = rosbag.Bag(bag_name)
    for topic, msg, t in camera_bag.read_messages(topics=['/camera/image_raw/compressed']):
        global_img = bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
        cv2.imshow("preview", global_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

