import utils
import PyQt5 # for Mayavi
import argparse
from mayavi import mlab
import numpy as np

PC_RANGE = [0, -39.68, -1, 69.12, 39.68, 7]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--p', dest='file_path', required=True, type=str, help="pcd file path")
    parser.add_argument('file_path', type=str, help="file path")
    args = parser.parse_args()
    file = args.file_path

    pc = utils.load_pc(args.file_path)
    # valid_idx = (pc[:, 0] >= PC_RANGE[0]) & (pc[:, 0] < PC_RANGE[3]) & (pc[:, 1] >= PC_RANGE[1]) & (pc[:, 1] < PC_RANGE[4]) & (pc[:, 2] >= PC_RANGE[2]) & (pc[:, 2] < PC_RANGE[5])
    # pc = pc[valid_idx, :]
    fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))

    print(pc.shape)
    if pc.shape[1] == 4:
        vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
    else:
        vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], 0., mode='point', figure=fig)

    mlab.view(distance=25)
    print('min_z = {} , max_z = {}'.format(np.min(pc, axis=0)[2], np.max(pc, axis=0)[2]))
    mlab.show()