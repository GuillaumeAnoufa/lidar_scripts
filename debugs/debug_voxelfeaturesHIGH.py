import torch
import torch.nn as nn
import numpy as np
import glob
from pathlib import Path

# Test de VFE

if __name__ == '__main__':
    
    # batch_dict = {}
    # batch_dict['batch_size'] = 1
    # grid_size = np.array([864, 864, 40])
    
    # voxel_features = torch.load("data/voxel_features_VFE.pt")
    # voxel_num_points = torch.load("data/voxel_num_points_VFE.pt")

    # points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
    # normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
    # points_mean = points_mean / normalizer
    # output_features = points_mean.contiguous()

    # print("MAX: ",torch.max(voxel_num_points))

    # torch.set_printoptions(profile="full")
    # print(batch_dict['voxel_features'])
    
    # good_coord = torch.load("data/voxel_features_good.pt")
    # print(good_coord)
    
    dataset_path = "/media/ganoufa/GAnoufaSSD/datasets/generated_datasets/unigineClass0/lidar/"
    file_path = Path(dataset_path)
    data_file_list = glob.glob(str(file_path / f'*.npy')) if file_path.is_dir() else [file_path]
    
    for file in data_file_list:
        pc = np.load(file)
        pc_max = np.max(pc)
        print(pc.shape)
        
        if pc_max > 100:
            print(pc_max)