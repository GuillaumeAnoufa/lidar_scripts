import time
import numpy as np, sys, os, cv2, yaml
import PyQt5 # for Mayavi
import utils

sys.path.append('/home/ganoufa/workSpace/lidar/useful_scripts/')
import algos.projections as proj

configs = yaml.load(open("/home/ganoufa/workSpace/lidar/catkin_ws/src/lidar_camera_processing/calibration/calibration.yaml", 'r').read(), Loader=yaml.FullLoader)
root = configs['data']['root']
intrinsic_matrix = np.loadtxt(os.path.join(root, 'intrinsic'))
distortion = np.loadtxt(os.path.join(root, 'distortion'))
extrinsic_matrix = np.loadtxt(os.path.join(root, 'extrinsic'))

def circle_crop_function_basic(pc, radius, center):
    cropped_pc_list = []
    xc, yc = center
    for i in range(len(pc)):
        if (((pc[i,0] - xc)*(pc[i,0] - xc))) + (((pc[i,1] - yc)*(pc[i,1] - yc))) <= (radius*radius):
            cropped_pc_list.append(pc[i])
    cropped_pc = np.array(cropped_pc_list)
    return cropped_pc

def sphere_crop_function_opti(pc, radius, center):
    temp = pc[:, :3] - np.repeat(center, len(pc), axis=0)
    # On recup tous les indices des points qui sont dans le cercle (selon x et y)
    valid_idx = np.where(np.linalg.norm(temp, axis=1) < radius)
    cropped_pc = pc[valid_idx, :].squeeze(0)
    print(cropped_pc.shape)
    return cropped_pc

# (X forward, Y facing left, Z upward) -> X in image is -Y in lidar coordinates, Y in image is -Z in lidar coordinates
####
####           LIDAR

####               (Z) ^
####           (X)^    |
####               \   |
####                \  |
####                 \ |
####                  \|
####    (Y)<----------(.)
####

####           IMAGE

####    (.) ------------> (X)
####     |
####     |
####     |
####     |
####     \/ (Y)
####
# Crop selon -Y et -Z
def bbox_crop(pc, bbox_min, bbox_max):
    temp = pc[:, (2, 3)] + np.repeat(bbox_min, len(pc), axis=0)
    # On recup tous les indices des points qui sont dans le cercle (selon x et y)
    valid_idx = np.where( temp < bbox_max )
    cropped_pc = pc[valid_idx, :].squeeze(0)
    print(cropped_pc.shape)
    return cropped_pc

def test(pc):
    temp = pc[:5, :]
    print(temp)

    val = np.where((temp < -0.395).any(axis=0))  # On recupère les indices des points dont le x est inférieur à -0.395
    print(val)

    temp = temp[val, :].squeeze(0)
    print(temp)

def project(pc, img):
    view_img = np.copy(img)
    # 8ms
    view_img = cv2.undistort(view_img, intrinsic_matrix, distortion)
    view_img = cv2.cvtColor(view_img, cv2.COLOR_BGR2RGB)

    # 60ms -> peut-être ne pas projeter toute l'image ? Seulement dans la zone de l'image interessante
    board = proj.pc_to_img(pc, view_img, extrinsic_matrix, intrinsic_matrix)
    return board

if __name__ == '__main__':

    pcd_path = "/media/ganoufa/GAnoufaSSD/datasets/vols_24_02/record1/281.pcd"
    img_path = "/media/ganoufa/GAnoufaSSD/datasets/vols_24_02/record1/281.png"
    
    pc = utils.load_pc(pcd_path)
    img = cv2.imread(img_path)

    board = project(pc, img)
    img_roi = cv2.selectROI("selectROI", board)

    start=time.perf_counter()

    # convert to pc_roi
    # project pointcloud to image
    projection_points = proj.undistort_projection(pc[:, :3], intrinsic_matrix, extrinsic_matrix)
    projection_points = np.column_stack([np.squeeze(projection_points), pc[:, 3:]])

    # crop
    projection_points = projection_points[np.where(
        (projection_points[:, 0] > 0) &
        (projection_points[:, 0] < img.shape[1]) &
        (projection_points[:, 1] > 0) &
        (projection_points[:, 1] < img.shape[0])
    )]

    projection_points_selected = projection_points[np.where(
        (projection_points[:, 1] > img_roi[1]) &
        (projection_points[:, 1] < img_roi[1] + img_roi[3]) &
        (projection_points[:, 0] > img_roi[0]) &
        (projection_points[:, 0] < img_roi[0] + img_roi[2])
    )]

    # Optimisation: 
    # On aurait pu sauvegarder les coordonnées dans le point cloud des projections points en rajoutant 3 colonnes 
    # au lieu de faire la projection inverse
    pc_selected = proj.back_projection(projection_points_selected, intrinsic_matrix, extrinsic_matrix)

    proj.img_to_pc(pc_selected, img, extrinsic_matrix, intrinsic_matrix)

    end=time.perf_counter()
    print('processing time : {}ms'.format( (end-start)*1000 ) )

    try:
        from mayavi import mlab
        fig = mlab.figure('pc', size=(1440, 810), bgcolor=(0.05, 0.05, 0.05))
        vis= mlab.points3d(pc_selected[:, 0], pc_selected[:, 1], pc_selected[:, 2], pc_selected[:, 3], mode='point', figure=fig)
        mlab.view(distance=25)
        mlab.show()
    except ImportError:
        print('mayavi not installed, skip visualization')
