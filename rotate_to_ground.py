import utils
import PyQt5 # for Mayavi
import argparse
from mayavi import mlab
import numpy as np
import time
import anno_read
PC_RANGE = [0, -40., -30., 150.4, 40., 10.]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="file path")
    args = parser.parse_args()
    file = args.file_path
    pc = utils.load_pc(args.file_path)
    # Suppression des points à l'origine sinon ça marche pas
    pc = pc[(pc[:, 0] != 0) | (pc[:, 1] != 0) | (pc[:, 2] != 0)]
    start=time.perf_counter()
    rot_mat, translation = utils.rotate_plane(pc[:, :3])
    pc[:, :3] = np.add(pc[:, :3] @ rot_mat.T, translation)
    end=time.perf_counter()

    # CROP
    # valid_idx = (pc[:, 0] >= PC_RANGE[0]) & (pc[:, 0] < PC_RANGE[3]) & (pc[:, 1] >= PC_RANGE[1]) & (pc[:, 1] < PC_RANGE[4]) & (pc[:, 2] >= PC_RANGE[2]) & (pc[:, 2] < PC_RANGE[5])
    # pc = pc[valid_idx, :]
    print('processing time : {}ms'.format( (end-start)*1000 ) )

    print('min = {} , max = {}'.format(np.min(pc, axis=0), np.max(pc, axis=0)))

    fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
    # # label_file = file.replace('.pcd', '.txt').replace('lidar', 'label')
    # # for obj in anno_read.get_objects_from_label(label_file):
    # #     box = np.add(obj.corners @ rot_mat.T, translation)
    # #     fig = utils.draw_gt_box(box, fig, color=(0, 1,1))
    if pc.shape[1] == 4:
        vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
    else:
        vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], mode='point', figure=fig)

    mlab.view(distance=25)
    mlab.show()
    
    