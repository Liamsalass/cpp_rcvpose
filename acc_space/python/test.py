import open3d as o3d
import numpy as np


lm_cls_names = ["ape", "cam", "can", "cat", "driller", "duck", "eggbox", "glue", "holepuncher", "iron", "lamp"]

add_threshold = {
    "eggbox": 0.019735770122546523,
    "ape": 0.01421240983190395,
    "cat": 0.018594838977253875,
    "cam": 0.02222763033276377,
    "duck": 0.015569664208967385,
    "glue": 0.01930723067998101,
    "can": 0.028415044264086586,
    "driller": 0.031877906042,
    "holepuncher": 0.019606109985
}

linemod_K = np.array([
    [572.4114, 0.0, 325.2611],
    [0.0, 573.57043, 242.04899],
    [0.0, 0.0, 1.0]
])

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    #pointc->actual scene
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    actual_xyz=xyz
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy,actual_xyz

def rgbd_to_point_cloud(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    print(zs.min())
    print(zs.max())
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts
    
def rgbd_to_color_point_cloud(K, depth, rgb):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    r = rgb[vs,us,0]
    g = rgb[vs,us,1]
    b = rgb[vs,us,2]
    print(zs.min())
    print(zs.max())
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs, r, g, b]).T
    return pts

def rgbd_to_point_cloud_no_depth(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    zs_min = zs.min()
    zs_max = zs.max()
    iter_range = int(zs_max*1000)+1-int(zs_min*1000)
    pts=[]
    for i in range(iter_range):
        if(i%1==0):
            z_tmp = np.empty(zs.shape) 
            z_tmp.fill(zs_min+i*0.001)
            xs = ((us - K[0, 2]) * z_tmp) / float(K[0, 0])
            ys = ((vs - K[1, 2]) * z_tmp) / float(K[1, 1])
            if(i == 0):
                pts = np.expand_dims(np.array([xs, ys, z_tmp]).T, axis=0)
                #print(pts.shape)
            else:
                pts = np.append(pts, np.expand_dims(np.array([xs, ys, z_tmp]).T, axis=0), axis=0)
                #print(pts.shape)
    print(pts.shape)
    return pts


if __name__ == "__main__":
    print("Testing acc space")
    proj_func = False
    rgbd_to_pc = True


    if proj_func:
        xyz = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0]])

        print("xyz:\n", xyz)

        K = np.array([[1000.0, 0.0, 500.0],
                      [0.0, 1000.0, 300.0],
                      [0.0, 0.0, 1.0]])

        print("K:\n", K)

        RT = np.array([[2.0, 0.0, 0.0, 0.0],
                       [0.0, 2.0, 0.0, 5.0],
                       [0.0, 0.0, 2.0, 0.0]])

        print("RT:\n", RT)

        xy, actual_xyz = project(xyz, K, RT)

        print("Projected xy coordinates:\n", xy)
        print("Actual XYZ:\n", actual_xyz)

    if rgbd_to_pc:
        pc_load = o3d.io.read_point_cloud(r"C:\Users\User\.cw\work\datasets\test\LINEMOD\ape\ape.ply")
        
        o3d.visualization.draw_geometries([pc_load])

        