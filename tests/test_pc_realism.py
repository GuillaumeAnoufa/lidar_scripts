import tools.utils as utils
import PyQt5 # for Mayavi
import argparse
from mayavi import mlab
import numpy as np
import signal

# ! Voir manuel livox Avia page 10 pour l'impact de l'angle sur la reduction de dmax

def handler(signum, frame):
    exit(1)

import sys
np.set_printoptions(threshold=sys.maxsize)
def visualize_colored_pointcloud(pc):
    import PyQt5 # for Mayavi
    from mayavi import mlab
    mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
    
    # ! Les angles sont centrés en Zero
    # ! La perte induite par l'angle est:
    # ! Differente pour l'azimuth et l'elevation
    # ! Plus forte pour des faibles distances
    # ! -> Depend de alpha
    
    # ? 
    # Pour L'azimuth (horizontal) 
    # le champ de vision est de 70.4°
    # la réduction entre centre et bordure est de 15% 
    # (320m au centre, 272m au bord sur du 80% reflect)
    
    # Pour Le zenith (vertical) 
    # le champ de vision est de 77.2°
    # la réduction entre centre et bordure est de 19.375% 
    # (320m au centre, 258m au bord sur du 80% reflect)
    
    # On va arrondir prendre cos(angle) pour le moment quelque soit alpha, ça fonctionne pas trop mal (environ 20% à +- 30°)
    
    print(np.mean(pc[:, 11]))
    print(np.mean(pc[:, 12]))
    dmax = 320*np.abs(np.cos(pc[:, 11]) * np.cos(pc[:, 12]))
    # dmax = 200
    d = np.minimum(np.sqrt(pc[:, 0]**2 + pc[:, 1]**2 + pc[:, 2]**2), dmax)
    alpha = pc[:, 3]/255.
    prob = np.ones_like(pc[:, 0]) * (dmax-d)/dmax * alpha ** 2
    
    rdmvals = np.random.rand(pc.shape[0])
    
    cloud = pc[prob > rdmvals]
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], prob, mode='point')
    # mlab.points3d(cloud[:, 0], cloud[:, 1], cloud[:, 2], prob[prob > rdmvals], mode='point')
    
    # mlab.axes()
    print(pc.shape)
    print(cloud.shape)
    mlab.show()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="file path")
    parser.add_argument('-a', '--auxiliary', dest='show_auxiliary', action='store_true',help='Show auxiliary color instead of albedo')
    args = parser.parse_args()
    pc = utils.load_pc(str(args.file_path))
    
    visualize_colored_pointcloud(pc)
    