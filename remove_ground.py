import array
import time

import numpy as np
import ground_removal_ext
import PyQt5 # for Mayavi
import pandas as pd
import utils

def load_pc_velodyne(bin_file_path):
    """
    load pointcloud file (KITTI format)
    :param bin_file_path:
    :return:
    """
    with open(bin_file_path, 'rb') as bin_file:
        pc = array.array('f')
        pc.fromstring(bin_file.read()) # replace fromstring to frombytes for python3
        pc = np.array(pc).reshape(-1, 4)
        return pc

if __name__ == '__main__':

    velodyne_example = '/home/ganoufa/workSpace/lidar/ViktorTsoi/pointcloud_ground_removal/2011_09_28/2011_09_28_drive_0034_sync/velodyne_points/data/0000000000.bin'

    # livox_example = '/home/ganoufa/data/pcds/plane_ground_500.csv'
    livox_example = '/media/ganoufa/GAnoufaSSD/datasets/vols_24_02/record1/4460.pcd'

    # pc = load_pc_velodyne(velodyne_example)

    pc = utils.load_pc(livox_example)

    print(pc.shape)
    # ground removal
    start=time.perf_counter()
    segmentation = ground_removal_ext.ground_removal_kernel(pc, 0.2, 200)
    end=time.perf_counter()
    print('processing time : {}ms'.format( (end-start)*1000 ) )

    # Nx5, x, y, z, intensity, is_ground
    print(segmentation.shape)
    print(segmentation)

    try:
        from mayavi import mlab

        segmentation[..., 3] = segmentation[..., 4]
        fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
        vis= mlab.points3d(segmentation[:, 0], segmentation[:, 1], segmentation[:, 2], segmentation[:, 3], mode='point', figure=fig)
        mlab.view(distance=25)
        mlab.show()
    except ImportError:
        print('mayavi not installed, skip visualization')
