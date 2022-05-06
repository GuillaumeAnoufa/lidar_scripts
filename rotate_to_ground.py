import utils
import PyQt5 # for Mayavi
import argparse
from mayavi import mlab
import numpy as np
import pyransac3d as pyrsc
import time
import anno_read

PC_RANGE = [0, -40., -30., 150.4, 40., 10.]

def rotate_plane(pc):
    plane1 = pyrsc.Plane()
    best_eq, _ = plane1.fit(pc, 0.2)

    n_sign = np.sign(best_eq[2])
    n = np.array(best_eq[:3]) * n_sign # normale au plan orientée vers le haut (z)

    # On défini n = [-sin(alp)*cos(eps), sin(eps), cos(alp)*cos(eps)]
    # On défini la base u1, u2, n
    eps = np.arcsin(n[1])
    alp = np.arccos(n[2] / np.cos(eps))
    u1 = [np.cos(alp), 0, np.sin(alp)]
    u2 = [np.sin(eps) * np.sin(alp), np.cos(eps), np.sin(eps)*np.cos(alp)]

    # Rotation then Translation
    rot_mat = np.stack([u1, u2, n])
    translation = np.array([0, 0, n_sign * best_eq[3]])
    return rot_mat, translation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="file path")
    args = parser.parse_args()
    file = args.file_path
    pc = utils.load_pc(args.file_path)
    print(pc.shape)
    start=time.perf_counter()
    rot_mat, translation = rotate_plane(pc[:, :3])
    pc[:, :3] = np.add(pc[:, :3] @ rot_mat.T, translation)
    end=time.perf_counter()

    # CROP
    valid_idx = (pc[:, 0] >= PC_RANGE[0]) & (pc[:, 0] < PC_RANGE[3]) & (pc[:, 1] >= PC_RANGE[1]) & (pc[:, 1] < PC_RANGE[4]) & (pc[:, 2] >= PC_RANGE[2]) & (pc[:, 2] < PC_RANGE[5])
    pc = pc[valid_idx, :]
    print('processing time : {}ms'.format( (end-start)*1000 ) )

    print('min = {} , max = {}'.format(np.min(pc, axis=0), np.max(pc, axis=0)))

    fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
    # label_file = file.replace('.pcd', '.txt').replace('lidar', 'label')
    # for obj in anno_read.get_objects_from_label(label_file):
    #     box = np.add(obj.corners @ rot_mat.T, translation)
    #     fig = utils.draw_gt_box(box, fig, color=(0, 1,1))
    if pc.shape[1] == 4:
        vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
    else:
        vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], mode='point', figure=fig)

    mlab.view(distance=25)
    mlab.show()
    
    