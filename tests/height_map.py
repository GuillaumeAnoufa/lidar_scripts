import utils
import PyQt5 # for Mayavi
import argparse
from mayavi import mlab
import numpy as np
import glob
from pathlib import Path
import signal

import sys
np.set_printoptions(threshold=sys.maxsize)

PC_RANGE = [0, -50.0, -4.0, 100.0, 50.0, 4.0]

def handler(signum, frame):
    exit(1)

def visualize_colored_pointcloud(pc):
    import PyQt5 # for Mayavi
    from mayavi import mlab
    fig = mlab.figure('pc', size=(1000, 1000), bgcolor=(0.05, 0.05, 0.05))
    lut_idx = np.arange(len(pc))
    # actual color
    # lut = np.column_stack([pc[:, 4:7], np.ones_like(pc[:, 0]) * 255])
    # aux color
    lut = np.column_stack([255-pc[:, 7:10], np.ones_like(pc[:, 0]) * 255])
    # plot
    p3d = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], lut_idx, mode='point', figure=fig)
    p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut_idx)
    p3d.module_manager.scalar_lut_manager.lut.table = lut
    # mlab.axes()
    
def visualize_z_pointcloud(pc):
    fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 2], mode='point', figure=fig)
    mlab.view(distance=25)
    mlab.show()
    
def viz_labels_plt(gnd_label):
    import matplotlib.pyplot as plt
    plt.interactive(True) # Non blocking
    hf = plt.figure()
    cs = plt.imshow(gnd_label.T, interpolation='nearest')
    cbar = hf.colorbar(cs)
    ha = hf.add_subplot(111, projection='3d')
    ha.set_xlabel('$X$', fontsize=20)
    ha.set_ylabel('$Y$')
    X = np.arange(0, 100, 1)
    Y = np.arange(0, 100, 1)
    X, Y = np.meshgrid(X, Y)  # `plot_surface` expects `x` and `y` data to be 2D
    # R = np.sqrt(X**2 + Y**2)
    ha.plot_surface(Y, X, gnd_label)
    ha.set_zlim(-10, 10)
    plt.draw()
    plt.show()
    # plt.pause(0.01)
    # hf.clf()
    
def viz_labels(gnd_label):
    fig = mlab.figure('height_map', size=(1000, 1000), bgcolor=(0.05, 0.05, 0.05))
    mlab.surf(gnd_label, figure=fig)
    
def height_map(pc, K=2):
    label = np.zeros((100, 100))
    size = np.zeros((100, 100))
    print(label.shape)
    
    # On rempli la map de hauteur en faisant une moyenne de la hauteur des points dans chaque m²
    # On retire x_min et y_min pour avoir une height map allant de 0 à 100
    for pt in pc:
        label[int(pt[0] - PC_RANGE[0]), int(pt[1] - PC_RANGE[1])] += pt[2]
        size[int(pt[0] - PC_RANGE[0]), int(pt[1] - PC_RANGE[1])] += 1
    # Si aucun point dans la zone on set à 1 pour éviter les div/0
    label[size>0] /= size[size>0]


    # ? Dans cette boucle on rempli un nouveau tableau new_label
    # ? qui aura pour valeur la moyenne des voisins étant à distance K et non nuls
    # ? en tout indice de label où on a aucune valeur de hauteur
    # for i in range(0+K, 100-K):
    #     for j in range(0+K, 100-K):
    # On parcourt des indices négatifs mais ça a pas l'air de l'emmerder... (de -K à 0)
    new_label = np.zeros((100, 100))
    for i in range(0, 100):
        for j in range(0, 100):
            if size[i, j] == 0:
                new_label[i, j] = np.mean(label[i-K:i+K+1, j-K:j+K+1][size[i-K:i+K+1, j-K:j+K+1] > 0])
                # print(" label[{}, {}] = {}".format(i, j, label[i-K:i+K+1, j-K:j+K+1][size[i-K:i+K+1, j-K:j+K+1] > 0]))
    label[size==0] = new_label[size==0]
    return label

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="file path")
    args = parser.parse_args()
    file_path = Path(args.file_path)
    data_file_list = glob.glob(str(file_path / f'*.csv')) if file_path.is_dir() else [file_path]
    
    for file in data_file_list:
        pc = utils.load_pc(str(file))
        # pc = np.array(pd.read_csv(data_path))
        print(pc.shape)
        rot_mat, translation = utils.rotate_plane(pc[:, :3])
        pc[:, :3] = np.add(pc[:, :3] @ rot_mat.T, translation)
        valid_idx = (pc[:, 0] >= PC_RANGE[0]) & (pc[:, 0] < PC_RANGE[3]) & (pc[:, 1] >= PC_RANGE[1]) & (pc[:, 1] < PC_RANGE[4]) & (pc[:, 2] >= PC_RANGE[2]) & (pc[:, 2] < PC_RANGE[5])
        pc = pc[valid_idx, :]
        
        label = height_map(pc, 2)
        viz_labels(label)
        # viz_labels_plt(label)
        # visualize_colored_pointcloud(pc)
        # visualize_z_pointcloud(pc)
        mlab.show()
        