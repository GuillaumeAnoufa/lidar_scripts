import pandas as pd
import os,sys,numpy as np
from pypcd import pypcd
import open3d as o3d
import PyQt5 # for Mayavi
from mayavi import mlab

def visualize_colored_pointcloud(pc):
    try:
        from mayavi import mlab
    except ImportError:
        print('mayavi not found, skip visualize')
        return
        # plot rgba points
    mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
    # 构建lut 将RGB颜色索引到点
    lut_idx = np.arange(len(pc))
    lut = np.column_stack([pc[:, 4:][:, ::-1], np.ones_like(pc[:, 0]) * 255])
    # plot
    p3d = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], lut_idx, mode='point')
    p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut_idx)
    p3d.module_manager.scalar_lut_manager.lut.table = lut
    # mlab.axes()
    mlab.show()

def visualize_pcd(data_path):
    pointcloud = o3d.io.read_point_cloud(data_path)
    o3d.visualization.draw_geometries([pointcloud])

def load_pc_pcd(file):
    cloud = pypcd.PointCloud.from_path(file)
    cloud = cloud.pc_data.view(np.float32).reshape(cloud.pc_data.shape + (-1,))
    return cloud

def load_pc_npy(file):
    cloud = np.load(file)
    return cloud


def load_pc_csv(file):
    """Read files cloud points csv.
    Args:
        self: The object pointer.
        file (str): path to files cloud points (csv).
    Returns:
        cloud (str): cloud points (x,y,z,d).
    """
    # data = pd.read_csv(file, usecols=["X", "Y", "Z", "Reflectivity"])
    data = pd.read_csv(file)
    cloud = np.array(data)
    return cloud

def load_pc(file):
    if (not os.path.exists(file)):
        print("file {} doesn't exist".format(file))
        exit()

    if file.endswith(".pcd"):
        pc = load_pc_pcd(file)
        return pc
    elif file.endswith(".csv"):
        pc = load_pc_csv(file)
        return pc    
    elif file.endswith(".bin"):
        pc = load_pc_velodyne(file)
        return pc
    elif file.endswith(".npy"):
        pc = load_pc_npy(file)
        return pc
    else:
        print("unknown file extension")

def load_pc_velodyne(bin_file_path):
    import array
    """
    load pointcloud file (KITTI format)
    :param bin_file_path:
    :return:
    """
    with open(bin_file_path, 'rb') as bin_file:
        pc = array.array('f')
        pc.fromstring(bin_file.read()) # replace fromstring to frombytes for python3
        pc = np.array(pc).reshape(-1, 4)
        return pc


def write_pcd(file, cloud):
    """Write Files pcd.
    Args:
        self: The object pointer.
        file (str): name file.
        cloud (array): data cloud point.
    Returns:
        clouds (str): save cloud points founded.
    """

    X = cloud[:, 0]
    Y = cloud[:, 1]
    Z = cloud[:, 2]
    I = cloud[:, 3]

    xyzi = np.column_stack((X, Y, Z, I))

    md = {'version': .7,
      'fields': ['x', 'y', 'z', 'intensity'],
      'size': [4, 4, 4, 4],
      'type': ['F', 'F', 'F', 'F'],
      'count': [1, 1, 1, 1],
      'width': len(xyzi),
      'height': 1,
      'viewpoint': [0, 0, 0, 1, 0, 0, 0],
      'points': len(xyzi),
      'data': 'ascii'}

    xyzi = xyzi.astype(np.float32)
    pc_data = xyzi.view(np.dtype([('x', np.float32),
                                 ('y', np.float32),
                                 ('z', np.float32),
                                 ('intensity', np.float32)])).squeeze()
    new_cloud = pypcd.PointCloud(md, pc_data)

    new_cloud.save_pcd(file)

def csv_to_pcd(csv_file):
    if (not os.path.exists(csv_file)) or (not csv_file.endswith(".csv")):
        print("ERROR: file {} is not a csv".format(csv_file) )
        sys.exit()
    new_pcd_path = os.path.splitext(csv_file)[0] + ".pcd"

    print("converting {} to {}".format(csv_file, new_pcd_path) )
    cloud = load_pc_csv(csv_file)
    write_pcd(new_pcd_path, cloud)

def npy_to_pcd(npy_file):
    if (not os.path.exists(npy_file)) or (not npy_file.endswith(".npy")):
        print("ERROR: file {} is not a npy".format(npy_file) )
        sys.exit()
    new_pcd_path = os.path.splitext(npy_file)[0] + ".pcd"
    print("converting {} to {}".format(npy_file, new_pcd_path) )
    cloud = np.load(npy_file)
    write_pcd(new_pcd_path, cloud)

def npy_to_csv(npy_file):
    if (not os.path.exists(npy_file)) or (not npy_file.endswith(".npy")):
        print("ERROR: file {} is not a npy".format(npy_file) )
        sys.exit()
    new_csv_path = os.path.splitext(npy_file)[0] + ".csv"
    print("converting {} to {}".format(npy_file, new_csv_path) )
    cloud = np.load(npy_file)
    np.savetxt(new_csv_path, cloud[:, :4], delimiter=",", header="X,Y,Z,Reflectivity", comments='') # comments='' -> else the header is commented (a # is added)

def pcd_to_csv(pcd_file):
    if (not os.path.exists(pcd_file)) or (not pcd_file.endswith(".pcd")):
        print("ERROR: file {} is not a pcd".format(pcd_file) )
        sys.exit()
    new_csv_path = os.path.splitext(pcd_file)[0] + ".csv"
    print("converting {} to {}".format(pcd_file, new_csv_path) )
    cloud = load_pc_pcd(pcd_file)
    np.savetxt(new_csv_path, cloud, delimiter=",", header="X,Y,Z,Reflectivity", comments='')

def draw_gt_box(b, fig, color=(1,1,1), line_width=1, text_scale=(0.2,0.2,0.2), color_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    ''' 

    for k in range(0,4):
        # Axe X
        i,j=k,(k+1)%4
        mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

        # Axe Z
        i,j=k+4,(k+1)%4 + 4
        mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

        # Axe Y
        i,j=k,k + 4
        mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    
    for i, pt in enumerate(b):
        mlab.text3d(pt[0], pt[1], pt[2], '%d' % (i), scale=text_scale, color=color, figure=fig)
    return fig

def boxToCorners(box):#center, dims, angle):
    """
    Args:
        pred_box: List (cx, cy, cz, l, w, h, rz)
    Returns:
        corners_3d: (8, 3) corners of box3d
    """
    center = box[0:3]
    l, w, h = box[3:6]
    angle = box[6]
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

import pyransac3d as pyrsc
def rotate_plane(pc):
    plane1 = pyrsc.Plane()
    best_eq, _ = plane1.fit(pc, 0.2)

    n_sign = np.sign(best_eq[2])
    n = np.array(best_eq[:3]) * n_sign # normale au plan orientée vers le haut (z)

    # On défini n = [-sin(alp)*cos(eps), sin(eps), cos(alp)*cos(eps)]
    # On défini la base u1, u2, n
    eps = np.arcsin(n[1])
    alp = np.arccos(n[2] / np.cos(eps))
    u1 = [np.cos(alp), 0, np.sin(alp)]
    u2 = [np.sin(eps) * np.sin(alp), np.cos(eps), np.sin(eps)*np.cos(alp)]

    # Rotation then Translation
    rot_mat = np.stack([u1, u2, n])
    translation = np.array([0, 0, n_sign * best_eq[3]])
    return rot_mat, translation
