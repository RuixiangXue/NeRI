import numpy as np
from easydict import EasyDict
import yaml
import math
import cv2
def write_ri(filename,ri):
    ri=ri.permute(1,2,0).cpu().numpy()
    range_max=255
    range_min=0
    pixel_max=ri.max()
    pixel_min=ri.min()
    ri=np.round((range_max-range_min)*(ri-pixel_min)/(pixel_max-pixel_min)+range_min)
    cv2.imwrite(filename,ri)
def create_transform_map(cfg):
    lidar_cfg=cfg["DATA"]["LiDAR_CFG_YAML"]
    with open(lidar_cfg, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    lidar_config = EasyDict(config)
    horizontal_FOV = lidar_config.HORIZONTAL_FOV * (np.pi / 180)
    vertical_max = lidar_config.VERTICAL_ANGLE_MAX * (np.pi / 180)
    vertical_min = lidar_config.VERTICAL_ANGLE_MIN * (np.pi / 180)
    vertical_FOV = vertical_max - vertical_min
    H = lidar_config.RANGE_IMAGE_HEIGHT
    W = lidar_config.RANGE_IMAGE_WIDTH
    transform_map = np.zeros((H, W, 3))
    for h in range(H):
        for w in range(W):
            altitude = vertical_FOV * (h / (H - 1)) + vertical_min
            azimuth = horizontal_FOV * (w / W)
            transform_map[h, w, 0] = math.cos(altitude) * math.cos(azimuth)  # * depth
            transform_map[h, w, 1] = math.cos(altitude) * math.sin(azimuth)  # * depth
            transform_map[h, w, 2] = math.sin(altitude)  # * depth
    return transform_map.astype(np.float32)

def range2pc(output,cfg):
    cfg = yaml.safe_load(open(cfg))  
    transform_map = create_transform_map(cfg)
    output = (output.cpu().numpy()*cfg["DATA"]["MAX_RANGE"]).reshape(cfg["DATA"]["V_Res"],cfg["DATA"]["H_Res"],1)
    if len(output.shape) == 2:
        point_cloud = np.expand_dims(output, -1) * transform_map
    elif len(output.shape) == 3:
        point_cloud = output * transform_map
    else:
        assert False
    return point_cloud# (64,2000,3)
    