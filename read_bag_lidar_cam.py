#!/usr/bin/env python3
# coding=utf-8

from __future__ import absolute_import, print_function

import cv2
import numpy as np
import ros_numpy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, PointCloud2
import rosbag
import threading
import PyQt5 # for Mayavi
from mayavi import mlab

# Global variables
global_pc = np.empty([1, 4])
global_img = None

def read_bag(bag_name):
    global global_pc, global_img
    with rosbag.Bag(bag_name, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/livox/to_integrate', '/camera/image_raw/compressed']):
            if topic == '/livox/to_integrate':
                # Ros time is in µsec
                msg.__class__ = PointCloud2
                pc_array = ros_numpy.numpify(msg)
                # parsing point cloud
                pc = np.zeros((pc_array.shape[0], 4))
                pc[:, 0] = pc_array['x']
                pc[:, 1] = pc_array['y']
                pc[:, 2] = pc_array['z']
                pc[:, 3] = pc_array['intensity']
                print("new_data PC")
                global_pc = np.concatenate([global_pc, pc])[-len(pc) * 2:]
            if topic == '/camera/image_raw/compressed':
                global_img = bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
                print("new_data image")
                # cv2.imshow("preview", global_img)
                # cv2.waitKey(1)

if __name__ == '__main__':
    bridge = CvBridge()
    bag_name = "/media/ganoufa/GAnoufaSSD/vols_24_02/camera_lidar_vol_0.bag"

    bagThread = threading.Thread(target=read_bag, args=(bag_name,), daemon=True)
    bagThread.start()

    fig = mlab.figure('pc', size=(960,540), bgcolor=(0.05, 0.05, 0.05))
    vis = mlab.points3d(0, 0, 0, 0, mode='point', figure=fig)
    mlab.view(distance=25)

    @mlab.animate(delay=10)
    def anim():
        while bagThread.is_alive():
            # print("display ")
            vis.mlab_source.reset(x=global_pc[:, 0], y=global_pc[:, 1], z=global_pc[:, 2], scalars=global_pc[:, 3])
            # mlab.savefig(f'{i}.png', figure=fig)
            yield
        mlab.close(all=True)
    anim()
    mlab.show()
    cv2.destroyAllWindows()
