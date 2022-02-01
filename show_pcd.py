import utils
import PyQt5 # for Mayavi
import argparse
from mayavi import mlab
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--p', dest='file_path', required=True, type=str, help="pcd file path")
    parser.add_argument('file_path', type=str, help="file path")
    args = parser.parse_args()
    file = args.file_path

    pc = utils.load_pc(args.file_path)

    fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))

    print(pc.shape)
    if pc.shape[1] == 4:
        vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
    else:
        vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], 0., mode='point', figure=fig)

    mlab.view(distance=25)
    mlab.show()