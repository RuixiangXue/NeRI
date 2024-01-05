import numpy as np
from dataset.transformer import PCTransformer
import open3d as o3d
import struct
import os
from scipy.spatial.transform import Rotation
def load_files(folder,start_idx,number_frames):
    """Load all files in a folder and sort."""
    file_paths = os.listdir(folder)
    file_paths.sort()
    file_paths = file_paths[start_idx:]
    file_paths = file_paths[:number_frames]
    return file_paths


def load_data(file,data):
    # return N x 3 point cloud
    data_type = file.split('.')[-1]
    if data_type == 'txt':
        point_cloud = np.loadtxt(file)
    elif data_type == 'bin':
        point_cloud = np.fromfile(file, dtype=np.float32)
        if data=='KITTI':
            point_cloud = point_cloud.reshape((-1, 4))
        elif data == 'nuScenes':
            point_cloud = point_cloud.reshape((-1, 5))
    elif data_type == 'npy' or data_type == 'npz':
        point_cloud = np.load(file)
    elif data_type == 'ply':
        pcd = o3d.io.read_point_cloud(file)
        point_cloud = np.asarray(pcd.points)
    elif data_type == 'pcd':
        pcd = o3d.io.read_point_cloud(file)
        point_cloud = np.asarray(pcd.points)
    else:
        assert False, 'File type not correct: ' + file

    point_cloud = point_cloud[:, :3]
    return point_cloud

def load_range_image_points_from_file(file,transformer,data):
    original_point_cloud = load_data(file,data)
    range_image = transformer.point_cloud_to_range_image(original_point_cloud)
    range_image = np.expand_dims(range_image, -1)
    point_cloud = transformer.range_image_to_point_cloud(range_image)
    return point_cloud, range_image, original_point_cloud

def save_point_cloud_to_file(file, point_cloud, color=None):
    data_type = file.split('.')[-1]
    valid_idx = np.where(np.sum(point_cloud, -1) != 0)
    point_cloud = point_cloud[valid_idx]
    if data_type == 'txt':
        point_cloud = np.concatenate((point_cloud, np.zeros((point_cloud.shape[0], 1))), -1)
        np.savetxt(file, point_cloud)
    elif data_type == 'bin':
        point_cloud = np.concatenate((point_cloud, np.zeros((point_cloud.shape[0], 1))), -1)
        point_cloud.astype(np.float32).tofile(file)
    elif data_type == 'npy' or data_type == 'npz':
        point_cloud = np.concatenate((point_cloud, np.zeros((point_cloud.shape[0], 1))), -1)
        np.save(file, point_cloud)
    elif data_type == 'ply':
        point_cloud = point_cloud[:, :3]
        # Write header of .ply file
        with open(file, 'wb') as fid:
            fid.write(bytes('ply\n', 'utf-8'))
            fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
            fid.write(bytes('element vertex %d\n' % point_cloud.shape[0], 'utf-8'))
            fid.write(bytes('property float x\n', 'utf-8'))
            fid.write(bytes('property float y\n', 'utf-8'))
            fid.write(bytes('property float z\n', 'utf-8'))
            fid.write(bytes('end_header\n', 'utf-8'))

            # Write 3D points to .ply file
            for i in range(point_cloud.shape[0]):
                fid.write(bytearray(struct.pack("fff", point_cloud[i, 0], point_cloud[i, 1], point_cloud[i, 2])))
    elif data_type == 'pcd':
        points_o3d = o3d.geometry.PointCloud()
        points_o3d.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        if color is not None:
            color_vec = color[valid_idx]
            points_o3d.colors = o3d.utility.Vector3dVector(color_vec)
        o3d.io.write_point_cloud(file, points_o3d)
    else:
        assert False, 'File type not correct.'
def load_calib(calib_path):
    """Load calibrations (T_cam_velo) from file."""
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Tr:" in line:
                    line = line.replace("Tr:", "")
                    T_cam_velo = np.fromstring(line, dtype=float, sep=" ")
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print("Calibrations are not avaialble.")

    return np.array(T_cam_velo)
def load_poses(pose_path):
    """Load ground truth poses (T_w_cam0) from file.
    Args:
      pose_path: (Complete) filename for the pose file
    Returns:
      A numpy array of size nx4x4 with n poses as 4x4 transformation
      matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if ".txt" in pose_path:
            with open(pose_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=" ")

                    if len(T_w_cam0) == 12:
                        T_w_cam0 = T_w_cam0.reshape(3, 4)
                        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    elif len(T_w_cam0) == 16:
                        T_w_cam0 = T_w_cam0.reshape(4, 4)
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)["arr_0"]

    except FileNotFoundError:
        print("Ground truth poses are not avaialble.")

    return np.array(poses)
def normalize_pose(poses):
    
    translations = np.array([pose[:3, 3] for pose in poses])
    max_translation = np.max(translations)
    min_translation = np.min(translations)

    rotations = np.array([Rotation.from_matrix(pose[:3, :3]).as_rotvec() for pose in poses])
    max_rotation = np.max(rotations)
    min_rotation = np.min(rotations)
        
    normal_translations = list((translations-min_translation)/(max_translation-min_translation))
    normal_rotations = list((rotations-min_rotation)/(max_rotation-min_rotation))
    normal = [[normal_rot,normal_trans] for normal_rot, normal_trans in zip(normal_rotations, normal_translations) ]
    return  normal