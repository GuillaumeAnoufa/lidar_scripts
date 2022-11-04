import numpy as np
import PyQt5 # for Mayavi
from mayavi import mlab
import utils
import signal
import argparse
import anno_read
import glob
from pathlib import Path

def handler(signum, frame):
    exit(1)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="file path")
    parser.add_argument('-a', '--auxiliary', dest='show_auxiliary', action='store_true',help='Show auxiliary color instead of albedo')
    args = parser.parse_args()
    file_path = Path(args.file_path)

    data_file_list = glob.glob(str(file_path / f'*.csv')) if file_path.is_dir() else [file_path]
    
    for pc_file in data_file_list:
        pc_file = str(pc_file)
        print("loading: ", pc_file)
        label_file = pc_file.replace('lidar', 'label').rsplit( ".", 1 )[ 0 ] + '.txt'
        
        pc = utils.load_pc(pc_file)
        fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))

        # draw gt
        for obj in anno_read.get_objects_from_label(label_file):
            utils.draw_gt_box(obj.corners, fig, color=(1, 0,0))
        
        vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
        mlab.view(distance=25)
        mlab.show()
    
    

