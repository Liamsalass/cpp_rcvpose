import open3d as o3d
import numpy as np

pointCloud = o3d.io.read_point_cloud('C:/Users/User/.cw/work/datasets/test/LINEMOD/phone/phone.ply')

print(pointCloud)

# o3d.visualization.draw_geometries([pointCloud])

print(np.asarray(pointCloud.points).dtype)

print(np.asarray(pointCloud.points)[:10,:])

# Store the point cloud as a numpy array
np.save('C:/Users/User/.cw/work/datasets/test/LINEMOD/phone/pc_phone.npy', np.asarray(pointCloud.points))