import numpy as np, sys, os, cv2, yaml
# import PyQt5 # for Mayavi
import utils

sys.path.append('/home/ganoufa/workSpace/lidar/useful_scripts/')
import algos.projections as proj
import numpy.linalg as LA

configs = yaml.load(open("/home/ganoufa/workSpace/lidar/catkin_ws/src/lidar_camera_processing/calibration/calibration.yaml", 'r').read(), Loader=yaml.FullLoader)
root = configs['data']['root']
intrinsic_matrix = np.loadtxt(os.path.join(root, 'intrinsic'))
distortion = np.loadtxt(os.path.join(root, 'distortion'))
extrinsic_matrix = np.loadtxt(os.path.join(root, 'extrinsic'))

# Dla merde proj 2D -> 3D impossible, on regarde les pts du pc qui sont pas loins une fois projetés dans le repere img
# def camToWorld(u, v):
#     uv_1=np.array([[u,v,1]], dtype=np.float32)
#     print(uv_1)
#     xyz_c= LA.inv(intrinsic_matrix).dot(uv_1.T)
#     print(xyz_c)
#     # XYZ=LA.inv(extrinsic_matrix).dot(xyz_c)
#     R, T = extrinsic_matrix[:3, :3], extrinsic_matrix[:3, 3]
#     XYZ = np.matmul(LA.inv(R), xyz_c - T.reshape(-1, 1)).T
#     return XYZ

def camToLidar(pc, x, y, d=10):
    projection_points = proj.undistort_projection(pc[:, :3], intrinsic_matrix, extrinsic_matrix)
    projection_points = np.column_stack([np.squeeze(projection_points), pc[:, 3:]])

    # crop
    projection_points = projection_points[np.where(
        (projection_points[:, 0] >= y - d) &
        (projection_points[:, 0] < y + d) &
        (projection_points[:, 1] > x - d) &
        (projection_points[:, 1] <  x + d)
    )]
    print(projection_points.shape)
    projection_points = proj.back_projection(projection_points, intrinsic_matrix, extrinsic_matrix)
    return np.mean(projection_points, axis=0)[:3]

if __name__ == '__main__':

    pcd_path = "/home/ganoufa/workSpace/lidar/project_cpp/data/1099.pcd"
    pc = utils.load_pc(pcd_path)
    img = cv2.imread(pcd_path.replace('.pcd', '.png'))

    print(camToLidar(pc, x=300, y=591, d=15))


