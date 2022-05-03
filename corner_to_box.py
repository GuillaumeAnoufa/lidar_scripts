import PyQt5 # for Mayavi
from mayavi import mlab
import utils
import numpy as np
import math

def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point. 
    
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32 
    
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners. 
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = 3
    dtype = np.float32
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    # corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.reshape(-1, 1, ndim) * corners_norm.reshape(1, 2**ndim, ndim)
    return corners

def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")

    return np.einsum('aij,jka->aik', points, rot_mat_T)


def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=(0.5, 0.5, 0.5),
                           axis=2):
    """convert kitti locations, dim and angles to corners
    
    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dim in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners

def generate_corners3d(center, dims, angle):
    """
    generate corners3d representation for this object
    :return corners_3d: (8, 3) corners of box3d in camera coord
    """
    l, w, h = dims
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]

    # R = np.array([[np.cos(angle), 0, np.sin(angle)],
    #                 [0, 1, 0],
    #                 [-np.sin(angle), 0, np.cos(angle)]])
    R = np.stack([[np.cos(angle), np.sin(angle), 0],
                        [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    corners3d = np.dot(R, corners3d).T
    corners3d = corners3d + center
    return corners3d

if __name__ == '__main__':
    box = np.array([
[31.254395123330596, 3.3917501290679866, 0.10679355282792358],
[33.093268435291293, 3.082108743552908, 0.71975125342646606],
[33.381078435952986, 4.9812449109035697, 0.81568790934903745],
[31.542205123992289, 5.2908862964186483, 0.20273020875055181],
[31.652111237112194, 3.3917501245155108, -1.0863548128342586],
[33.490984549072891, 3.0821087390004323, -0.47339711223571612],
[33.778794549734585, 4.981244906351094, -0.37746045631314473],
[31.939921237773888, 5.2908862918661725, -0.99041815691163038],
    ])
    
    # Angle entre le [2, 1] et x
    angle = math.atan2(box[1, 1] - box[2, 1], box[1, 0] - box[2, 0])
    
    length = math.sqrt((box[1, 0] - box[0, 0])**2 
        + (box[1, 1] - box[0, 1])**2 + (box[1, 2] - box[0, 2])**2)
    width = math.sqrt((box[5, 0] - box[0, 0])**2 
        + (box[5, 1] - box[0, 1])**2 + (box[5, 2] - box[0, 2])**2) 
    height = math.sqrt((box[3, 0] - box[0, 0])**2 
        + (box[3, 1] - box[0, 1])**2 + (box[3, 2] - box[0, 2])**2)
    dim = [length, width, height]
    print('dim: ', dim)
    
    dim = np.max(box, axis=0) - np.min(box, axis=0) # max sur les colonnes
    center = (box[0] + box[6]) /2

    print('center: ', center)
    print('dim: ', dim)
    print('angle: ', angle)
    
    
    # new_box = center_to_corner_box3d(np.expand_dims(center,0), np.expand_dims(dim,0), np.expand_dims(angle,0)).squeeze()
    
    upleft_corner = np.array(center - [dim[0]/2, dim[1]/2, dim[2]/2])
    autre_box = np.stack([upleft_corner + [0, 0, dim[2]], 
                 upleft_corner + [dim[0], 0, dim[2]],
                 upleft_corner + dim,
                 upleft_corner + [0, dim[1], dim[2]],
                 upleft_corner,
                 upleft_corner + [dim[0], 0, 0],
                 upleft_corner + [dim[0], dim[1], 0],
                 upleft_corner + [0, dim[1], 0]
            ])
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat = np.stack([[rot_cos, rot_sin, 0],
                        [-rot_sin, rot_cos, 0], [0, 0, 1]])
    # autre_box = np.stack([(rot_mat @ (x - center)) + center for x in autre_box])


    cool_box = generate_corners3d(center, dim, angle)

    pc = utils.load_pc("/media/ganoufa/GAnoufaSSD/datasets/generated_datasets/test_box/test/anomaly/4.csv")
    fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
    fig = utils.draw_gt_box(box, fig, color=(1, 0,0))
    # fig = utils.draw_gt_box(autre_box, fig, color=(1, 1,0))
    fig = utils.draw_gt_box(cool_box, fig, color=(1, 1,1))
    vis= mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point', figure=fig)
    mlab.view(distance=25)
    mlab.show()

