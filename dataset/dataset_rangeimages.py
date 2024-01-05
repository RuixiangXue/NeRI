import sys
sys.path.append("..")
import numpy as np
import torch
import os
import torchvision.transforms as transforms
import yaml
from dataset.dataset_utils import load_range_image_points_from_file,load_files,load_poses,load_calib,normalize_pose
import random
from dataset.transformer import PCTransformer

class RangeImgaes_Kitti_Dataset(torch.utils.data.Dataset):
    def __init__(self,cfg,purpose):
        self.cfg = cfg
        self.purpose = purpose
        self.dir = self.cfg["DATA"]["DIR"]
        self.seq = self.cfg["DATA"]["SEQ"]
        self.start_idx = self.cfg["DATA"]["START_IDX"]
        self.number_frames = self.cfg["DATA"]["NUMBER_FRAMES"]
        self.frame_gap = self.cfg["DATA"]["FRAME_GAP"]
        
        self.aug_normalization = self.cfg["DATA"]["AUG_NORMALIZATION"]
        if self.aug_normalization:
            self.means = self.cfg["DATA"]["MEANS"]
            self.std = self.cfg["DATA"]["STD"]
        self.normalization = self.cfg["DATA"]["NORMALIZATION"]
        if self.normalization:
            self.max_range = self.cfg["DATA"]["MAX_RANGE"]
        
        self.transform_toTensor = transforms.ToTensor()
        self.transformer = PCTransformer(lidar_cfg=self.cfg["DATA"]["LiDAR_CFG_YAML"])
        frame_idx, self.frame_path = [], []
        accum_frame_num = []
        
        seq_str = "{0:02d}".format(int(self.seq))
        path_to_seq = os.path.join(self.dir, seq_str)
        self.scan_path = os.path.join(path_to_seq, "velodyne")
        self.frame_path = load_files(self.scan_path,self.start_idx,self.number_frames)
        self.frame_pose = self.read_poses(path_to_seq,self.start_idx,self.number_frames)
        num_frame = 0
        for frame_id in self.frame_path:
            frame_idx.append(num_frame)
            num_frame += 1
        accum_frame_num.append(num_frame)
        self.frame_idx = [float(x) / len(frame_idx) for x in frame_idx]
        self.accum_frame_num = np.asfarray(accum_frame_num)
    def read_poses(self, path_to_seq,start_idx,number_frames):
        pose_file = os.path.join(path_to_seq, "pose.txt")
        calib_file = os.path.join(path_to_seq, "calib.txt")

        poses = np.array(load_poses(pose_file))
        inv_frame0 = np.linalg.inv(poses[0])
        # load calibrations
        T_cam_velo = load_calib(calib_file)
        T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
        T_velo_cam = np.linalg.inv(T_cam_velo)
        # convert kitti poses from camera coord to LiDAR coord
        new_poses = []
        for pose in poses:
            new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
        pose = np.asarray(new_poses) #[n,4,4]
        
        normalized_rot_trans = normalize_pose(pose) #a list,every element is a list [rot,trans]
        normalized_rot_trans = normalized_rot_trans[start_idx:]
        normalized_rot_trans = normalized_rot_trans[:number_frames]
        
        return normalized_rot_trans
    def __len__(self):
            return len(self.frame_idx) // self.frame_gap
    
    def __getitem__(self, idx):
        

        valid_idx = idx * self.frame_gap

        frame_id = self.frame_path[valid_idx]
        
        frame_name = os.path.join(self.scan_path, frame_id)
        frame_idx = torch.tensor(self.frame_idx[valid_idx])
        frame_pose = torch.tensor(self.frame_pose[valid_idx])
        if self.purpose == "train":
            _, range_image, _ = load_range_image_points_from_file(frame_name,self.transformer,'KITTI')
            tensor_range_image = self.transform_toTensor(range_image)
            if self.aug_normalization:
                 tensor_range_image = transforms.Normalize(self.means,self.std)(tensor_range_image)
            if self.normalization:
                tensor_range_image = torch.div(tensor_range_image,self.max_range)
            return [tensor_range_image,frame_idx,frame_pose]
        elif self.purpose == "test":
            point_cloud, range_image, original_point_cloud = load_range_image_points_from_file(frame_name,self.transformer,'KITTI')
            pc_seg = (np.where(point_cloud[..., 0] != 0, 1, 0))
            tensor_range_image = self.transform_toTensor(range_image)
            if self.aug_normalization:
                 tensor_range_image = transforms.Normalize(self.means,self.std)(tensor_range_image)
            if self.normalization:
                tensor_range_image = torch.div(tensor_range_image,self.max_range)
            return [tensor_range_image,frame_idx,frame_pose,point_cloud,original_point_cloud,pc_seg]

def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return
