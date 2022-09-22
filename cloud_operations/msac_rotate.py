import utils
import PyQt5 # for Mayavi
import argparse
from mayavi import mlab
import numpy as np
import time
import anno_read
import pcl
PC_RANGE = [0, -40., -30., 150.4, 40., 10.]

def rotate_plane(pc):
    segment = pc.make_segmenter()
    segment.set_optimize_coefficients(True)
    segment.set_model_type(pcl.SACMODEL_PLANE)
    segment.set_method_type(pcl.SAC_RANSAC)
    # SAC_RANSAC - RANdom SAmple Consensus
    # SAC_LMEDS - Least Median of Squares
    # SAC_MSAC - M-Estimator SAmple Consensus
    # SAC_RRANSAC - Randomized RANSAC
    # SAC_RMSAC - Randomized MSAC
    # SAC_MLESAC - Maximum LikeLihood Estimation SAmple Consensus
    # SAC_PROSAC - PROgressive SAmple Consensus
    segment.set_distance_threshold(0.2)
    _, best_eq = segment.segment()
    print(best_eq)

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
    
    pc = pcl.load(args.file_path)
    start=time.perf_counter()
    # ATTENTION ÇA NE MARCHE PAS SI ON NE SUPPRIME PAS LES POINTS [0,0,0] AVANT
    rot_mat, translation = rotate_plane(pc)
    pc = pc.to_array()
    pc[:, :3] = np.add(pc[:, :3] @ rot_mat.T, translation)
    end=time.perf_counter()
    print('processing time : {}ms'.format( (end-start)*1000 ) )

    print('min = {} , max = {}'.format(np.min(pc, axis=0), np.max(pc, axis=0)))

    fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
    if pc.shape[1] == 4:
        vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
    else:
        vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], mode='point', figure=fig)

    mlab.view(distance=25)
    mlab.show()
    