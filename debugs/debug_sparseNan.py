import torch
import torch.nn as nn
from functools import partial
try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv
import numpy as np

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

class VoxelBackBone8x(nn.Module):
    def __init__(self, input_channels, grid_size, **kwargs):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        print("sparse_shape: ", self.sparse_shape)
        
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)

        # print("voxel_coords: ", voxel_coords.shape)
        # print("voxel_coords NAN VALUES ?2: ", torch.isnan(voxel_coords).any())
        
        # print("voxel_features: ", voxel_features.shape)
        # print("voxel_features NAN VALUES ?2: ", torch.isnan(voxel_features).any())
        
        print("input_sp_tensor: ", input_sp_tensor.spatial_shape)
        print("input_sp_tensor NAN VALUES ", torch.isnan(input_sp_tensor.dense()).any())

        print("x: ", x.spatial_shape)
        print("x NAN VALUES ?2: ", torch.isnan(x.dense()).any())
        
        print("x_conv1: ", x_conv1.spatial_shape)
        print("x_conv1 NAN VALUES ?2: ", torch.isnan(x_conv1.dense()).any())
        
        print("x_conv2: ", x_conv2.spatial_shape)
        print("x_conv2 NAN VALUES ?2: ", torch.isnan(x_conv2.dense()).any())
        
        print("x_conv3: ", x_conv3.spatial_shape)
        print("x_conv3 NAN VALUES ?2: ", torch.isnan(x_conv3.dense()).any())

        print("x_conv4: ", x_conv4.spatial_shape)
        print("x_conv4 NAN VALUES ?2: ", torch.isnan(x_conv4.dense()).any())
        
        print("out: ", out.spatial_shape)
        print("out NAN VALUES ?2: ", torch.isnan(out.dense()).any())
        
        return out

if __name__ == '__main__':
    
    batch_dict = {}
    batch_dict['batch_size'] = 1
    grid_size = np.array([864, 864, 40])
    
    # Works, without colors
    # input_channels = 4
    # batch_dict['voxel_features'] = torch.load("data/voxel_features_good.pt")
    # batch_dict['voxel_coords'] = torch.load("data/voxel_coords_good.pt")
    
    # Dont work with colors
    input_channels = 7
    batch_dict['voxel_features'] = torch.load("data/voxel_features_2.pt")#.clamp(min=0., max=10.)
    batch_dict['voxel_coords'] = torch.load("data/voxel_coords_2.pt")

    model = VoxelBackBone8x(input_channels=input_channels, grid_size=grid_size).cuda()
    # print(model)

    y = model(batch_dict)
    
    # torch.set_printoptions(profile="full")
    # print(batch_dict['voxel_features'])
    
    # good_coord = torch.load("data/voxel_features_good.pt")
    # print(good_coord)
    