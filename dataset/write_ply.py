import numpy as np
import open3d as o3d

def write_ply_o3d_geo(filedir, coords, dtype='float32'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(dtype))
    o3d.io.write_point_cloud(filedir, pcd, write_ascii=True)
    f = open(filedir)
    lines = f.readlines()
    lines[4] = 'property float x\n'
    lines[5] = 'property float y\n'
    lines[6] = 'property float z\n'
    fo = open(filedir, "w")
    fo.writelines(lines)

    return

def write_ply_o3d_normal(filedir, coords, dtype='float32', knn=20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(dtype))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    o3d.io.write_point_cloud(filedir, pcd, write_ascii=True)
    f = open(filedir)
    lines = f.readlines()
    lines[4] = 'property float x\n'
    lines[5] = 'property float y\n'
    lines[6] = 'property float z\n'
    lines[7] = 'property float nx\n'
    lines[8] = 'property float ny\n'
    lines[9] = 'property float nz\n'
    fo = open(filedir, "w")
    fo.writelines(lines)

    return