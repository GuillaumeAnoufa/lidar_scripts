import pandas as pd
import os,sys,csv,numpy as np
from pypcd import pypcd

def readCSV(file):
    """Read files cloud points csv.
    Args:
        self: The object pointer.
        file (str): path to files cloud points (csv).
    Returns:
        cloud (str): cloud points (x,y,z,d).
    """
    data = pd.read_csv(file)
    cloud = np.array(data)
    return cloud

def writePCD(file, cloud):
    """Write Files PCD.
    Args:
        self: The object pointer.
        file (str): name file.
        cloud (array): data cloud point.
    Returns:
        clouds (str): save cloud points founded.
    """
    X = cloud[:, 8]
    Y = cloud[:, 9]
    Z = cloud[:, 10]
    I = cloud[:, 11]

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

if __name__ == "__main__":
    dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = dir + "/" + sys.argv[1]

    if (not os.path.exists(csv_path)) or (not csv_path.endswith(".csv")):
        print("ERROR: file {} is not a csv".format(csv_path) )
        sys.exit()

    new_pcd_path = os.path.splitext(csv_path)[0] + ".pcd"

    print("converting {} to {}".format(csv_path, new_pcd_path) )
    cloud = readCSV(csv_path)
    writePCD(new_pcd_path, cloud)