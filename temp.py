import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
import os
from numba import jit, prange

linemod_cls_names = ['ape','benchvise','cam','can','cat','driller','duck','eggbox','glue','holepuncher','iron','lamp','phone']

linemod_K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]])

depthGeneration = False

linemod_path = "datasets/LINEMOD/"
original_linemod_path = "datasets/LINEMOD_ORIG/"
              
def project(xyz, K, RT):
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    actual_xyz=xyz
    xyz = np.dot(xyz, K.T)
    
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy,actual_xyz

def rgbd_to_point_cloud(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts, vs, us

@jit(nopython=True, parallel=True)   
def fast_for_map(yList, xList, xyz, distance_list, Radius3DMap):
    for i in prange(len(xList)):
        Radius3DMap[yList[i],xList[i]] = distance_list[i]
    return Radius3DMap
 
def linemod_pose(path, i):
    R = open("{}/data/rot{}.rot".format(path, i))
    R.readline()
    R = np.float32(R.read().split()).reshape((3, 3))

    t = open("{}/data/tra{}.tra".format(path, i))
    t.readline()
    t = np.float32(t.read().split())
    
    return R, t


def read_depth(path):
    if (path[-3:] == 'dpt'):
        with open(path) as f:
            h,w = np.fromfile(f,dtype=np.uint32,count=2)
            data = np.fromfile(f,dtype=np.uint16,count=w*h)
            depth = data.reshape((h,w))
    else:
        depth = np.asarray(Image.open(path)).copy()
    return depth
    
@jit(nopython=True, parallel=True)   
def fast_for(pixel_coor, xy, actual_xyz, distance_list, Radius3DMap):
    z_mean = np.mean(actual_xyz[:,2])
    for coor in pixel_coor:
        iter_count=0
        z_loc = 0
        z_min = 99999999999999999
        for xy_single in xy:
            if(coor[0]==xy_single[1] and coor[1]==xy_single[0]):
              
                if(actual_xyz[iter_count,2]<z_min):
                    z_loc = iter_count
                    z_min = actual_xyz[iter_count,2]
                
            iter_count+=1
        
        
        if(z_min<=z_mean):
            if depthGeneration:
                Radius3DMap[xy[z_loc][1],xy[z_loc][0]]=actual_xyz[z_loc,2]
                pre_z_loc = z_loc
            else:
                Radius3DMap[xy[z_loc][1],xy[z_loc][0]]=distance_list[z_loc]
                pre_z_loc = z_loc
        else:
            if depthGeneration:
                Radius3DMap[xy[z_loc][1],xy[z_loc][0]]=actual_xyz[pre_z_loc,2]
            else:
                Radius3DMap[xy[z_loc][1],xy[z_loc][0]]=distance_list[pre_z_loc]
    return Radius3DMap
    
z_min = 999999999999999999
z_max = 0
depth_list=[]



if __name__=='__main__':
    for class_name in linemod_cls_names:
        print(class_name)
        
        pcd_load = o3d.io.read_point_cloud(linemod_path+class_name+"/"+class_name+".ply")
    
        xyz_load = np.asarray(pcd_load.points)
        print(xyz_load)
        
        keypoints=np.load(linemod_path+class_name+"/"+"Outside9.npy")
        points_count = 1
        
        for keypoint in keypoints:
            
            print(keypoint)
        
            x_mean = keypoint[0]   
            y_mean = keypoint[1] 
            z_mean = keypoint[2]
            
            rootDict = original_linemod_path+class_name+"/" 
            GTDepthPath = rootDict+'FakeDepth/'
            if depthGeneration:
                saveDict = original_linemod_path+class_name+"/FakeDepth/"
            else:
                saveDict = original_linemod_path+class_name+"/Out_pt"+str(points_count)+"_dm/"    
            if(os.path.exists(saveDict)==False):
                os.mkdir(saveDict)
            points_count+=1
            iter_count = 0
            dataDict = rootDict + "data/"
            for filename in os.listdir(dataDict):
                if filename.endswith(".dpt"):
                    print(filename)
                    
                    realdepth = read_depth(dataDict+filename)
                    mask = np.asarray(Image.open(linemod_path+class_name+"/mask/"+os.path.splitext(filename)[0][5:].zfill(4)+".png"), dtype=int)
                    mask = mask[:,:,0]        
               
                    realdepth[np.where(mask==0)] = 0
               
                    Radius3DMap = np.zeros(mask.shape)
                    RT = np.load(linemod_path+class_name+"/pose/pose"+os.path.splitext(filename)[0][5:]+".npy")
                    print(RT)
                    print(linemod_pose(rootDict,os.path.splitext(filename)[0][5:]))
                    pixel_coor = np.argwhere(mask==255)
                    xyz,y,x = rgbd_to_point_cloud(linemod_K, realdepth)
                    print(xyz)
                    print(RT)
                    dump, transformed_kpoint = project(np.array([keypoint]),linemod_K,RT)
                    transformed_kpoint = transformed_kpoint[0]*1000
                    print(transformed_kpoint)
                    distance_list = ((xyz[:,0]-transformed_kpoint[0])**2+(xyz[:,1]-transformed_kpoint[1])**2+(xyz[:,2]-transformed_kpoint[2])**2)**0.5
                    Radius3DMap = fast_for_map(y, x, xyz, distance_list, Radius3DMap)
                   
                    iter_count+=1
                
                    plt.imshow(Radius3DMap)
                    plt.show()
                    mean = 0.84241277810665
                    std = 0.12497967663932731
                
                    if depthGeneration:
                        np.save(saveDict+os.path.splitext(filename)[0][5:].zfill(6)+'.npy',Radius3DMap)
                    else:
                        np.save(saveDict+os.path.splitext(filename)[0][5:].zfill(6)+'.npy',Radius3DMap*10)
                   
                    if(Radius3DMap.max()>z_max):
                        z_max = Radius3DMap.max()
                   
            if depthGeneration:
                break
           