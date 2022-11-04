import tools.utils as utils
import argparse
import numpy as np
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="file path")
    parser.add_argument('-a', '--auxiliary', dest='show_auxiliary', action='store_true',help='Show auxiliary color instead of albedo')
    args = parser.parse_args()
    pc = utils.load_pc(args.file_path)
    pc = pc[(pc[:, 0] != 0) | (pc[:, 1] != 0) | (pc[:, 2] != 0)]
    
    example_csv = utils.load_pc("_data/csv_files/long.csv")
    print(pc.shape)
    print(example_csv.shape)
    
    csv_path = args.file_path.replace('.pcd', '.csv')
    csv_path = "/home/ganoufa/data/test.csv"
    header = "Version,Slot ID,LiDAR Index,Rsvd,Error Code,Timestamp Type,Data Type,Timestamp,X,Y,Z,Reflectivity,Tag,Ori_x,Ori_y,Ori_z,Ori_radius,Ori_theta,Ori_phi"
    cloud = example_csv[:pc.shape[0]]
    cloud[:, 8:12] = pc
    # cloud[:, 11] = int(cloud[:, 11])
    cloud[:, 11] *=2
    fmt = ('%d,%d,%d,%d,%s,%d,%d,%ld,%.8e,%.8e,%.8e,%d,%d,%d,%d,%d,%.8e,%.8e,%.8e')
    np.savetxt(csv_path, cloud, delimiter=",", header=header, comments='', fmt=fmt)
    