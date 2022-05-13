import utils
import PyQt5 # for Mayavi
import argparse
from mayavi import mlab
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, MeanShift
PC_RANGE = [0, -40., 0.1, 150.4, 40., 10.]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="file path")
    args = parser.parse_args()
    file = args.file_path
    pc = utils.load_pc(args.file_path)
    pc = pc[(pc[:, 0] != 0) | (pc[:, 1] != 0) | (pc[:, 2] != 0)]
    rot_mat, translation = utils.rotate_plane(pc[:, :3])
    pc[:, :3] = np.add(pc[:, :3] @ rot_mat.T, translation)
    valid_idx = (pc[:, 0] >= PC_RANGE[0]) & (pc[:, 0] < PC_RANGE[3]) & (pc[:, 1] >= PC_RANGE[1]) & (pc[:, 1] < PC_RANGE[4]) & (pc[:, 2] >= PC_RANGE[2]) & (pc[:, 2] < PC_RANGE[5])
    pc = pc[valid_idx, :]
    # model = KMeans(n_clusters=3).fit(pc[:, :3])
    # model = MeanShift().fit(pc[:, :3])
    model = DBSCAN(eps=.7, min_samples=10, algorithm='auto').fit(pc[:, 3:4])
    colors = [255//(klabel+1) for klabel in model.labels_]
    
    fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
    vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], colors, mode='point', figure=fig)

    mlab.view(distance=25)
    mlab.show()
    
    
    
    