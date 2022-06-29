import utils
import PyQt5 # for Mayavi
import argparse
from mayavi import mlab
import numpy as np
import time
from scipy.spatial import cKDTree
from ifp import ifp_sample
# PC_RANGE = [0, -40., -30., 150.4, 40., 10.]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="file path")
    args = parser.parse_args()
    file = args.file_path
    pc = utils.load_pc(args.file_path)

    # Suppression des points à l'origine sinon ça marche pas
    pc = pc[(pc[:, 0] != 0) | (pc[:, 1] != 0) | (pc[:, 2] != 0)]
    print(pc.shape)

    # Rotation ground
    rot_mat, translation = utils.rotate_plane(pc[:, :3])
    pc[:, :3] = np.add(pc[:, :3] @ rot_mat.T, translation)

    start=time.perf_counter()
    dists, indices = cKDTree(pc[:, :3]).query(pc[:, :3], 8)
    # sparsify to 3000 points
    sampled_indices = ifp_sample(dists, indices, 3000)
    pc = pc[sampled_indices]
    print(pc.shape)
    end=time.perf_counter()

    print('processing time : {}ms'.format( (end-start)*1000 ) )

    fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))

    if pc.shape[1] == 4:
        vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
    else:
        vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], mode='point', figure=fig)

    mlab.view(distance=25)
    mlab.show()
    
    