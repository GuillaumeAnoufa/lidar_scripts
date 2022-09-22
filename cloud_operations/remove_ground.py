import time
import argparse
import numpy as np
import ground_removal_ext
import PyQt5 # for Mayavi
import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="file path")
    args = parser.parse_args()
    file = args.file_path
    pc = utils.load_pc(args.file_path)
    # Suppression des points à l'origine sinon ça marche pas
    pc = pc[(pc[:, 0] != 0) | (pc[:, 1] != 0) | (pc[:, 2] != 0)]
    rot_mat, translation = utils.rotate_plane(pc[:, :3])
    pc[:, :3] = np.add(pc[:, :3] @ rot_mat.T, translation)

    # ground removal
    start=time.perf_counter()
    segmentation = ground_removal_ext.ground_removal_kernel(pc, 0.2, 200)
    end=time.perf_counter()
    print('processing time : {}ms'.format( (end-start)*1000 ) )

    # Nx5, x, y, z, intensity, is_ground
    print(segmentation.shape)

    try:
        from mayavi import mlab

        segmentation[..., 3] = segmentation[..., 4]
        fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
        vis= mlab.points3d(segmentation[:, 0], segmentation[:, 1], segmentation[:, 2], segmentation[:, 3], mode='point', figure=fig)
        mlab.view(distance=25)
        mlab.show()
    except ImportError:
        print('mayavi not installed, skip visualization')
