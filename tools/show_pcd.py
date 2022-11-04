import utils
import PyQt5 # for Mayavi
import argparse
from mayavi import mlab
import numpy as np
import glob
from pathlib import Path
import signal

PC_RANGE = [0, -39.68, -1, 69.12, 39.68, 7]

def handler(signum, frame):
    exit(1)

def visualize_colored_pointcloud(pc, aux=False):
    import PyQt5 # for Mayavi
    from mayavi import mlab
    mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
    
    # mlab.pipeline.user_defined(data, filter=tvtk.CubeAxesActor())
    if pc.shape[1] == 4:
        p3d = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point')
    else:
        lut_idx = np.arange(len(pc))
        if aux:
            # aux color
            lut = np.column_stack([255-pc[:, 7:10], np.ones_like(pc[:, 0]) * 255])
        else:
            # actual color
            lut = np.column_stack([pc[:, 4:7], np.ones_like(pc[:, 0]) * 255])

        # plot
        p3d = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], lut_idx, mode='point')
        p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut_idx)
        p3d.module_manager.scalar_lut_manager.lut.table = lut
        # mlab.axes()
    mlab.show()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="file path")
    parser.add_argument('-a', '--auxiliary', dest='show_auxiliary', action='store_true',help='Show auxiliary color instead of albedo')
    args = parser.parse_args()
    file_path = Path(args.file_path)

    data_file_list = glob.glob(str(file_path / f'*.npy')) if file_path.is_dir() else [file_path]
    
    for file in data_file_list:
        pc = utils.load_pc(str(file))
        print('file: {}\nshape: {}'.format(str(file), pc.shape))
        # valid_idx = (pc[:, 0] >= PC_RANGE[0]) & (pc[:, 0] < PC_RANGE[3]) & (pc[:, 1] >= PC_RANGE[1]) & (pc[:, 1] < PC_RANGE[4]) & (pc[:, 2] >= PC_RANGE[2]) & (pc[:, 2] < PC_RANGE[5])
        # pc = pc[valid_idx, :]
        # fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
        # if pc.shape[1] == 4:
        #     vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
        # else:
        #     vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], 0., mode='point', figure=fig)

        # mlab.view(distance=25)
        # print('min_z = {} , max_z = {}'.format(np.min(pc, axis=0)[2], np.max(pc, axis=0)[2]))
        # mlab.show()
        visualize_colored_pointcloud(pc, args.show_auxiliary)