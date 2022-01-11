import pandas as pd
import os,sys,numpy as np
from pypcd import pypcd
import open3d as o3d

def visualize_pcd(data_path):
    pointcloud = o3d.io.read_point_cloud(data_path)
    o3d.visualization.draw_geometries([pointcloud])

def load_pc_pcd(file):
    pc = pypcd.PointCloud.from_path(file)
    pc_data = pc.pc_data
    cloud = np.column_stack((pc_data["x"], pc_data["y"], pc_data["z"], pc_data["intensity"]))
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
    data = pd.read_csv(file, usecols=["X", "Y", "Z", "Reflectivity"])
    cloud = np.array(data)
    return cloud

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

