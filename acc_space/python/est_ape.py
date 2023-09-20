from models.fcnresnet import DenseFCNResNet152, ResFCNResNet152
from util.horn import HornPoseFitting
import utils
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numba import jit,njit,cuda
import os
import open3d as o3d
import time
from numba import prange
import math
from sklearn import metrics
import scipy
import struct


lm_cls_names = ['ape', 'cam', 'can', 'cat', 'duck', 'driller', 'eggbox', 'glue', 'holepuncher','iron','lamp']
lmo_cls_names = ['ape', 'can', 'cat', 'duck', 'driller',  'eggbox', 'glue', 'holepuncher']
ycb_cls_names={1:'002_master_chef_can',
           2:'003_cracker_box',
           3:'004_sugar_box',
           4:'005_tomato_soup_can',
           5:'006_mustard_bottle',
           6:'007_tuna_fish_can',
           7:'008_pudding_box',
           8:'009_gelatin_box',
           9:'010_potted_meat_can',
           10:'011_banana',
           11:'019_pitcher_base',
           12:'021_bleach_cleanser',
           13:'024_bowl',
           14:'025_mug',
           15:'035_power_drill',
           16:'036_wood_block',
           17:'037_scissors',
           18:'040_large_marker',
           19:'051_large_clamp',
           20:'052_extra_large_clamp',
           21:'061_foam_brick'}
lm_syms = ['eggbox', 'glue']
ycb_syms = ['024_bowl','036_wood_block','051_large_clamp','052_extra_large_clamp','061_foam_brick']
add_threshold = {
                  'eggbox': 0.019735770122546523,
                  'ape': 0.01421240983190395,
                  'cat': 0.018594838977253875,
                  'cam': 0.02222763033276377,
                  'duck': 0.015569664208967385,
                  'glue': 0.01930723067998101,
                  'can': 0.028415044264086586,
                  'driller': 0.031877906042,
                  'holepuncher': 0.019606109985}

linemod_K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]])

#IO function from PVNet
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
    #print(zs.min())
    #print(zs.max())
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
    #print(zs.min())
    #print(zs.max())
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
    
def FCResBackbone(model, input_img_path):
    """
    This is a funciton runs through a pre-trained FCN-ResNet checkpoint
    Args:
        model: model obj
        input_img_path: input image to the model
    Returns:
        output_map: feature map estimated by the model
                    radial map output shape: (1,h,w)
                    vector map output shape: (2,h,w)
    """
    #model = DenseFCNResNet152(3,2)
    #model = torch.nn.DataParallel(model)
    #checkpoint = torch.load(model_path)
    #model.load_state_dict(checkpoint)
    #optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    #model, _, _, _ = utils.load_checkpoint(model, optim, model_path)
    #model.eval()
    input_image = Image.open(input_img_path).convert('RGB')
    #plt.imshow(input_image)
    #plt.show()
    img = np.array(input_image, dtype=np.float64)
    img /= 255.
    img -= np.array([0.485, 0.456, 0.406])
    img /= np.array([0.229, 0.224, 0.225])
    img = img.transpose(2, 0, 1)
    #dpt = np.load(normalized_depth)
    #img = np.append(img,np.expand_dims(dpt,axis=0),axis=0)
    input_tensor = torch.from_numpy(img).float()

    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    # use gpu if available
    if torch.cuda.is_available():
         input_batch = input_batch.to('cuda')
         model.to('cuda')
    with torch.no_grad():
        sem_out, radial_out = model(input_batch)
    sem_out, radial_out = sem_out.cpu(), radial_out.cpu()

    sem_out, radial_out = np.asarray(sem_out[0]),np.asarray(radial_out[0])
    return sem_out[0], radial_out[0]

#@jit(nopython=True)
def coords_inside_image(rr, cc, shape, val=None):
    """
    Modified based on https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/draw/draw.py#L484-L544
    Return the coordinates inside an image of a given shape.
    Parameters
    ----------
    rr, cc : (N,) ndarray of int
        Indices of pixels.
    shape : tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates.  Must be at least length 2. Only the first two values
        are used to determine the extent of the input image.
    val : (N, D) ndarray of float, optional
        Values of pixels at coordinates ``[rr, cc]``.
    Returns
    -------
    rr, cc : (M,) array of int
        Row and column indices of valid pixels (i.e. those inside `shape`).
    val : (M, D) array of float, optional
        Values at `rr, cc`. Returned only if `val` is given as input.
    """
    mask = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    if val is None:
        return rr[mask], cc[mask]
    else:
        return rr[mask], cc[mask], val[mask]

#@jit(nopython=True)        
def circle_perimeter(r_o, c_o, radius, method, shape):
    """
    Modified based on https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/draw/draw.py#L484-L544
    Generate circle perimeter coordinates.
    Parameters
    ----------
    r, c : int
        Centre coordinate of circle.
    radius : int
        Radius of circle.
    method : {'bresenham', 'andres'}
        bresenham : Bresenham method (default)
        andres : Andres method
    shape : tuple
        Image shape which is used to determine the maximum extent of output pixel
        coordinates. This is useful for circles that exceed the image size.
        If None, the full extent of the circle is used.
    Returns
    -------
    rr, cc : (N,) ndarray of int
        Bresenham and Andres' method:
        Indices of pixels that belong to the circle perimeter.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    Notes
    -----
    Andres method presents the advantage that concentric
    circles create a disc whereas Bresenham can make holes. There
    is also less distortions when Andres circles are rotated.
    Bresenham method is also known as midpoint circle algorithm.
    Anti-aliased circle generator is available with `circle_perimeter_aa`.
    References
    ----------
    .. [1] J.E. Bresenham, "Algorithm for computer control of a digital
           plotter", IBM Systems journal, 4 (1965) 25-30.
    .. [2] E. Andres, "Discrete circles, rings and spheres", Computers &
           Graphics, 18 (1994) 695-706.
    """

    rr = []
    cc = []

    c = 0
    r = radius
    d = 0

    dceil = 0
    dceil_prev = 0


    if method == 'bresenham':
        d = 3 - 2 * radius
    elif method == 'andres':
        d = radius - 1
    else:
        raise ValueError('Wrong method')

    while r >= c:
        rr.extend([r, -r, r, -r, c, -c, c, -c])
        cc.extend([c, c, -c, -c, r, r, -r, -r])

        if method == 'bresenham':
            if d < 0:
                d += 4 * c + 6
            else:
                d += 4 * (c - r) + 10
                r -= 1
            c += 1
        elif method == 'andres':
            if d >= 2 * (c - 1):
                d = d - 2 * c
                c = c + 1
            elif d <= 2 * (radius - r):
                d = d + 2 * r - 1
                r = r - 1
            else:
                d = d + 2 * (r - c - 1)
                r = r - 1
                c = c + 1

    if shape is not None:
        return coords_inside_image(np.array(rr, dtype=np.intp) + r_o,
                                    np.array(cc, dtype=np.intp) + c_o,
                                    shape)
    return (np.array(rr, dtype=np.intp) + r,
            np.array(cc, dtype=np.intp) + c)

#@jit(nopython=True,parallel=True)
def draw_sphere(x_o, r_o, c_o, radius, method, shape):
    xx=[]
    yy=[]
    zz=[]
    circle_shape = (shape[1],shape[2])
    radius_tmp = radius
    radius_decrement = 1
    # from x center to 0
    for i in range(x_o,0,-1):
        yy_tmp, zz_tmp = circle_perimeter(r_o, c_o, radius_tmp, method, circle_shape)
        radius_tmp =int(np.around((radius**2-radius_decrement**2)*0.5))
        radius_decrement+=1
        xx_tmp = np.full(yy_tmp.shape, i)
        yy.append(yy_tmp)
        zz.append(zz_tmp)
        xx.append(xx_tmp)
    radius_tmp = radius
    radius_decrement = 1
    #from x center to up boundary
    for i in range(x_o,shape[0]-1,1):
        yy_tmp, zz_tmp = circle_perimeter(r_o, c_o, radius_tmp, method, circle_shape)
        radius_tmp =int(np.around((radius**2-radius_decrement**2)*0.5))
        radius_decrement+=1
        xx_tmp = np.full(yy_tmp.shape, i)
        yy.append(yy_tmp)
        zz.append(zz_tmp)
        xx.append(xx_tmp)
    return (xx,yy,zz)

def parallel_for(count,xyz_mm,radial_list_mm,VoteMap_3D):
    xx=[]
    yy=[]
    zz=[]
    xyz = xyz_mm[count]
    radius = radial_list_mm[count]
    radius = int(np.around(radial_list_mm[count]))
    shape = VoteMap_3D.shape
    
    xx_,yy_,zz_=draw_sphere(int(np.around(xyz[0])),int(np.around(xyz[1])),int(np.around(xyz[2])),radius,'andres',shape)
    xx.append(xx_)
    yy.append(yy_)
    zz.append(zz_)
    return xx,yy,zz

@jit(nopython=True,parallel=True)
#@jit(parallel=True)     
def fast_for(xyz_mm,radial_list_mm,VoteMap_3D):  
    factor = (3**0.5)/4
    #print ('XYZ shape' , xyz_mm.shape[0])
    for count in prange(xyz_mm.shape[0]):
        xyz = xyz_mm[count]
        radius = radial_list_mm[count]
        radius = int(np.around(radial_list_mm[count]))
        shape = VoteMap_3D.shape
        for i in prange(VoteMap_3D.shape[0]):
            for j in prange(VoteMap_3D.shape[1]):
                for k in prange(VoteMap_3D.shape[2]):
                    distance = ((i-xyz[0])**2+(j-xyz[1])**2+(k-xyz[2])**2)**0.5
                    #print ('Count: ',count,' Distance: ',distance,' Radius: ',radius,' Factor: ',factor,' VoteMap: ',VoteMap_3D[i,j,k],' i: ',i,' j: ',j,' k: ',k,' xyz: ',xyz)
                    if radius - distance < factor and radius - distance>0:
                        VoteMap_3D[i,j,k]+=1
        
    return VoteMap_3D


@cuda.jit
def cuda_internal1(VoteMap_3D,xyz,radius):
    m, i,j,k=cuda.grid(4)
    if m<xyz.shape[0] and i<VoteMap_3D.shape[0] and j<VoteMap_3D.shape[1] and k<VoteMap_3D.shape[2]:
        distance = ((i-xyz[m,0])**2+(j-xyz[m,1])**2+(k-xyz[m,2])**2)**0.5
        if radius - distance < factor and radius - distance >=0:
            VoteMap_3D[i,j,k]+=1

@cuda.jit
def cuda_internal(xyz_mm, radial_list_mm, VoteMap_3D):
    m = cuda.grid(1)
    if m<xyz_mm.shape[0]:
        threadsperblock = (8, 8, 8)
        blockspergrid_x = math.ceil(VoteMap_3D.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(VoteMap_3D.shape[1] / threadsperblock[1])
        blockspergrid_z = math.ceil(VoteMap_3D.shape[2] / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        cuda_internal1[blockspergrid, threadsperblock](VoteMap_3D,xyz[m],radius[m])

@cuda.jit
def fast_for_cuda(xyz_mm,radial_list_mm,VoteMap_3D): 
    threadsperblock = (8, 8, 8, 8)
    blockspergrid_w = math.ceil(xyz_mm.shape[0] / threadsperblock[0])
    blockspergrid_x = math.ceil(VoteMap_3D.shape[0] / threadsperblock[1])
    blockspergrid_y = math.ceil(VoteMap_3D.shape[1] / threadsperblock[2])
    blockspergrid_z = math.ceil(VoteMap_3D.shape[2] / threadsperblock[3])
    blockspergrid = (blockspergrid_w, blockspergrid_x, blockspergrid_y, blockspergrid_z)
    cuda_internal1[blockspergrid, threadsperblock](VoteMap_3D,xyz_mm,radial_list_mm)

def Accumulator_3D(xyz, radial_list, kpt_count):
    acc_unit = 5
    # unit 5mm 

    #print('Input data: ')
    #print('xyz shape: ', xyz.shape)
    #for i in range(20):
    #    print('xyz[',i,']',xyz[i])
    #print('radial_list shape: ', radial_list.shape)
    #for i in range(20):
    #    print('radial_list[',i,']',radial_list[i])

    xyz_mm = xyz*1000/acc_unit #point cloud is in meter

    #print('xyz_mm shape: ', xyz_mm.shape)
    #for i in range(20):
    #    print('xyz_mm[',i,']',xyz_mm[i])

    #print(xyz_mm.shape)
    #recenter the point cloud
    x_mean_mm = np.mean(xyz_mm[:,0])
    y_mean_mm = np.mean(xyz_mm[:,1])
    z_mean_mm = np.mean(xyz_mm[:,2])

    xyz_mm[:,0] -= x_mean_mm
    xyz_mm[:,1] -= y_mean_mm
    xyz_mm[:,2] -= z_mean_mm


    radial_list_mm = radial_list*100/acc_unit  #radius map is in decimetre for training purpose

    xyz_mm_min = xyz_mm.min()
    xyz_mm_max = xyz_mm.max()
    radius_max = radial_list_mm.max()

    zero_boundary = int(xyz_mm_min-radius_max)+1

    if(zero_boundary<0):
        xyz_mm -= zero_boundary
       
        #length of 3D vote map 
    length = int(xyz_mm.max())


  
    VoteMap_3D = np.zeros((length+int(radius_max),length+int(radius_max),length+int(radius_max)))

    VoteMap_3D = fast_for(xyz_mm,radial_list_mm,VoteMap_3D)
                        
    center = np.argwhere(VoteMap_3D==VoteMap_3D.max())

    if len(center) > 1:
        print("Multiple centers located.")
   # print("debug center raw: ",center)
    center = center.astype("float64")
    if(zero_boundary<0):
        center = center+zero_boundary
        
    #return to global coordinate
    center[0,0] = (center[0,0]+x_mean_mm+0.5)*acc_unit
    center[0,1] = (center[0,1]+y_mean_mm+0.5)*acc_unit
    center[0,2] = (center[0,2]+z_mean_mm+0.5)*acc_unit
    
    #center = center*acc_unit+((3**0.5)/2)

    return center

@jit(nopython=True, parallel=True)    
def fast_for_no_depth(xyz_mm,radial_list_mm,VoteMap_3D):
    factor = (3**0.5)/4
    for xyz_son in xyz_mm:    
        for count in prange(xyz_son.shape[0]):
            xyz = xyz_son[count]
            radius = radial_list_mm[count]
            for i in prange(VoteMap_3D.shape[0]):
                for j in prange(VoteMap_3D.shape[1]):
                    for k in prange(VoteMap_3D.shape[2]):
                        distance = ((i-xyz[0])**2+(j-xyz[1])**2+(k-xyz[2])**2)**0.5
                        if radius - distance < factor and radius - distance>0:
                            VoteMap_3D[i,j,k]+=1
    return VoteMap_3D
    
def Accumulator_3D_no_depth(xyz, radial_list, pixel_coor):
    # unit 5mm 
    xyz_mm = xyz*200 #point cloud is in meter
    #recenter the point cloud
    x_mean_mm = np.mean(xyz_mm[:,0])
    y_mean_mm = np.mean(xyz_mm[:,1])
    z_mean_mm = np.mean(xyz_mm[:,2])
    xyz_mm[:,0] -= x_mean_mm
    xyz_mm[:,1] -= y_mean_mm
    xyz_mm[:,2] -= z_mean_mm
    
    radial_list_mm = radial_list*20 #radius map is in decimetre for training purpose
    
    xyz_mm_min = xyz_mm.min()
    xyz_mm_max = xyz_mm.max()
    radius_max = radial_list_mm.max()
    
    zero_boundary = int(xyz_mm_min-radius_max)+1
    #print("debug zero boundary: ",zero_boundary)
    
    if(zero_boundary<0):
        xyz_mm = xyz_mm-zero_boundary
        #length of 3D vote map 
    length = int(xyz_mm.max())+1
    
    VoteMap_3D = np.zeros((length,length,length))
    
    #print(length)
    
    VoteMap_3D = fast_for_no_depth(xyz_mm,radial_list_mm,VoteMap_3D)
    
                        
    center = np.argwhere(VoteMap_3D==VoteMap_3D.max())
    if(zero_boundary<0):
        center = center+zero_boundary
        
    #return to global coordinate
    center[0,0] += x_mean_mm
    center[0,1] += y_mean_mm
    center[0,2] += z_mean_mm
    
    center = center*5

    return center

#for original linemod depth
def read_depth(path):
    if (path[-3:] == 'dpt'):
        with open(path) as f:
            h,w = np.fromfile(f,dtype=np.uint32,count=2)
            data = np.fromfile(f,dtype=np.uint16,count=w*h)
            depth = data.reshape((h,w))
    else:
        depth = np.asarray(Image.open(path)).copy()
    return depth
depthList=[]


def fileToTensor(filename):
    with open(filename, "rb") as file:
        num_dimensions = struct.unpack("Q", file.read(8))[0]
        shape = struct.unpack(f"{num_dimensions}q", file.read(8 * num_dimensions))
        num_elements = struct.unpack("Q", file.read(8))[0]
        tensor_data = struct.unpack(f"{num_elements}f", file.read(4 * num_elements))
        tensor = torch.tensor(tensor_data).reshape(shape)

    return tensor


#==============================================================================
# 6D pose estimation
def estimate_6d_pose_lm():
    print ("Start pose estimation")
    horn = HornPoseFitting()
    root_dataset = 'C:/Users/User/.cw/work/datasets/test/'
    class_name = 'ape'

    rootPath = root_dataset + "LINEMOD_ORIG/"+class_name+"/" 
    rootpvPath = root_dataset + "LINEMOD/"+class_name+"/" 
    
    print('rootPath: ',rootPath)

    test_list = open(root_dataset + "LINEMOD/"+class_name+"/" +"Split/val.txt","r").readlines()
    test_list = [ s.replace('\n', '') for s in test_list]

    # test = 'C:/Users/User/.cw/work/cpp_rcvpose/acc_space/python/kpt' + str(2) + '_t/score_' + str(0) +'.txt'
    # sem_out = fileToTensor(test)
    # print('Testing shape of kpt2 score0: ')
    # print (sem_out.shape)
    
    pcd_load = o3d.io.read_point_cloud(root_dataset + "LINEMOD/"+class_name+"/"+class_name+".ply")
    
    #time consumption
    net_time = 0
    acc_time = 0
    general_counter = 0
    
    #counters
    bf_icp = 0
    af_icp = 0
    model_list=[]
    
    #h5 save keypoints
    #h5f = h5py.File(class_name+'PointPairsGT.h5','a')
    
    filenameList = []
    
    xyz_load = np.asarray(pcd_load.points)
    #print(xyz_load.shape)
    
    keypoints=np.load(root_dataset + "LINEMOD/"+class_name+"/"+"Outside9.npy")
    #print(keypoints)
    
    dataPath = rootpvPath + 'JPEGImages/'

    img_avg_time = 0
        
    for filename in os.listdir(dataPath):
        # Only 25 first images
        count = 0


        if filename.endswith(".jpg"):
            filename_without_ext = os.path.splitext(filename)[0]

            if filename_without_ext not in test_list:
                continue

            if filename_without_ext in test_list:
                print(filename)

                img_start_timer = time.time_ns()

                estimated_kpts = np.zeros((3,3))
                RTGT = np.load(root_dataset + "LINEMOD/"+class_name+"/pose/pose"+os.path.splitext(filename)[0][5:]+'.npy')
             


                keypoint_count = 1

                for keypoint in keypoints:
                    keypoint=keypoints[keypoint_count]

                    
                    iter_count = 0
                    centers_list = []
                    
                    GTRadiusPath = rootpvPath+'Out_pt'+str(keypoint_count)+'_dm/'

                    transformed_gt_center_mm = (np.dot(keypoints, RTGT[:, :3].T) + RTGT[:, 3:].T)*1000

                    transformed_gt_center_mm = transformed_gt_center_mm[keypoint_count]

                    input_path = dataPath +filename

                    normalized_depth = []
                    tic = time.time_ns()
                    
                    #Cpp backend
                    #sem_out_path = 'C:/Users/User/.cw/work/cpp_rcvpose/gpu_models/' + class_name + '/test_tensors/semantic/' + filename_without_ext + '.txt'
                    #rad_out_path = 'C:/Users/User/.cw/work/cpp_rcvpose/gpu_models/' + class_name + '/test_tensors/radial' + str(keypoint_count) + '/' + filename_without_ext + '.txt'
                    #radial_out = fileToTensor(rad_out_path)
                    #sem_out = fileToTensor(sem_out_path)

                    

                    #sem_out = (sem_out - sem_out.min()) / (sem_out.max() - sem_out.min())

                    #sem_out = np.where(sem_out>0.8,1,0).transpose(1,0)
                    #radial_out = np.array(radial_out).transpose(1,0)


                    #Python backend      
                    #sem_out_path = 'C:/Users/User/.cw/work/cpp_rcvpose/acc_space/python/kpt' +str(keypoint_count) +'/tensors_py/score_' + str(count) +'.npy'
                    #rad_out_path = 'C:/Users/User/.cw/work/cpp_rcvpose/acc_space/python/kpt' +str(keypoint_count) + '/tensors_py/score_rad_' + str(count) +'.npy'
                    #sem_out = np.load(sem_out_path)
                    #radial_out = np.load(rad_out_path)
                    #sem_out = np.where(sem_out>0.8,1,0)
                    #radial_out = np.array(radial_out)

                    # Paper output
                    radial_path = root_dataset + "/LINEMOD/"+class_name+"/"+"Out_pt"+str(keypoint_count)+"_dm/"+os.path.splitext(filename)[0][:] + '.npy'
                    #print(radial_path)
                    radial_out = np.load(radial_path)   

                    sem_out = np.where(radial_out!=0,1,0)

                    toc = time.time_ns()
                    net_time += toc-tic
                    #print("Network time consumption: ", network_time_single)
                    depth_map1 = read_depth(root_dataset + "LINEMOD/"+class_name+'/data/depth'+os.path.splitext(filename)[0][5:]+'.dpt')

                    #for i in range(20):
                    #    print(depth_map1[i])
                    #input("Press Enter to continue...")
                    

                    depth_map = depth_map1*sem_out 

                    pixel_coor = np.where(sem_out==1)

                    radial_list = radial_out[pixel_coor]
      
                    xyz_mm = rgbd_to_point_cloud(linemod_K,depth_map)
                    xyz = xyz_mm/1000   


                    dump, xyz_load_transformed=project(xyz_load, linemod_K, RTGT)

                    tic = time.time_ns()
                    
                    #gt_list = np.load(GTRadiusPath+ os.path.splitext(filename)[0] + '.npy')
                    #gt_list = gt_list[pixel_coor]         

                    # if the radial_list or xyz is empty then skip
                    if (len(radial_list) == 0 or len(xyz) == 0):
                        print("Empty Prediction for keypoint: ", keypoint_count)
                        keypoint_count+=1
                        if (keypoint_count == 4):
                            break
                        continue


                    center_mm_s = Accumulator_3D(xyz, radial_list, keypoint_count)
                    #gt_center = Accumulator_3D(xyz, gt_list)
                    #center_mm_s = Accumulator_3D_no_depth(xyz, radial_list, pixel_coor)
                    toc = time.time_ns()
                    acc_time += toc-tic
                    #print acc time in ms
                    #print("Accumulator time consumption: ", (toc-tic)/1000000, " ms")
                    
                    pre_center_off_mm = math.inf
                    
                    estimated_center_mm = center_mm_s[0]

                    center_off_mm = ((transformed_gt_center_mm[0]-estimated_center_mm[0])**2+
                                    (transformed_gt_center_mm[1]-estimated_center_mm[1])**2+
                                    (transformed_gt_center_mm[2]-estimated_center_mm[2])**2)**0.5
                    
                    #save estimations
                    '''
                    index 0: original keypoint
                    index 1: applied gt transformation keypoint
                    index 2: network estimated keypoint
                    '''
                    centers = np.zeros((1,3,3))
                    centers[0,0] = keypoint
                    centers[0,1] = transformed_gt_center_mm*0.001
                    centers[0,2] = estimated_center_mm*0.001
                    
                    estimated_kpts[keypoint_count - 1] = estimated_center_mm
                    
                    if(iter_count==0):
                        centers_list = centers
                        if keypoint_count==0:
                            f_save = int(os.path.splitext(filename)[0][5:])
                            filenameList=np.array([f_save])
                            #filenameList = np.expand_dims(filenameList,axis=0)
                            #print(filenameList)
                    else:
                        centers_list = np.append(centers_list, centers, axis=0)
                        if keypoint_count==0:
                            filenameList = np.append(filenameList, np.array([int(os.path.splitext(filename)[0][5:])]), axis=0)
                            #print(filenameList)
                    #print(centers_list.shape)
                    iter_count += 1

                    keypoint_count+=1
                    if (keypoint_count == 4):
                        break
                    
                
                kpts = keypoints[1:4,:]*1000
                RT = np.zeros((4, 4))

                horn.lmshorn(kpts, estimated_kpts, 3, RT)

                dump, xyz_load_est_transformed=project(xyz_load*1000, linemod_K, RT[0:3,:])
           

                input_image = np.asarray(Image.open(input_path).convert('RGB'))
                input_image_copy = np.copy(input_image)

                for coor in dump:
                    y, x = int(coor[1]), int(coor[0])
                    if 0 <= y < input_image_copy.shape[0] and 0 <= x < input_image_copy.shape[1]:
                        input_image_copy[y, x] = [255, 0, 0]

                # plt.imshow(input_image_copy)
                # plt.show()

                sceneGT = o3d.geometry.PointCloud()
                sceneEst = o3d.geometry.PointCloud()
                sceneGT.points=o3d.utility.Vector3dVector(xyz_load_transformed*1000)
                sceneEst.points=o3d.utility.Vector3dVector(xyz_load_est_transformed)
                sceneGT.paint_uniform_color(np.array([0,0,1]))
                sceneEst.paint_uniform_color(np.array([1,0,0]))
            
                #o3d.visualization.draw_geometries([sceneGT, sceneEst],window_name='gt vs est before icp')
                distance_bf_icp = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).mean()
                min_distance_bf_icp = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).min()
                max_distance_bf_icp = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).max()
                median_distance_bf_icp = np.median(np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)))
                std_dev_distance_bf_icp = np.std(np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)))

                if class_name in lm_syms:
                    if min_distance_bf_icp <= add_threshold[class_name]*1000:
                        bf_icp+=1   
                else:
                    if distance_bf_icp <= add_threshold[class_name]*1000:
                        bf_icp+=1
                     
                
                trans_init = np.asarray([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0], 
                                        [0, 0, 0, 1]])
                threshold = distance_bf_icp
                criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000000)
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    sceneGT, sceneEst,  threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria)
                sceneGT.transform(reg_p2p.transformation)
                
                # o3d.visualization.draw_geometries([sceneGT, sceneEst],window_name='gt vs est after icp')
                distance_af_icp = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).mean()
                min_distance_af_icp = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).min()
                max_distance_af_icp = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).max()
                median_distance_af_icp = np.median(np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)))
                std_dev_distance_af_icp = np.std(np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)))
                #print('ADD(s) point distance after ICP: ', distance)
                if class_name in lm_syms:
                    if min_distance_af_icp <= add_threshold[class_name]*1000:
                        af_icp+=1
                else:
                    if distance_af_icp <= add_threshold[class_name]*1000:
                        af_icp+=1                   
                general_counter += 1

                img_end_timer = time.time_ns()

                img_avg_time += img_end_timer - img_start_timer
                img_avg_time_ms = img_avg_time/general_counter/1000000

                print ('Before ICP: ', bf_icp)
                print ('After ICP: ', af_icp)
                print ('Current ADDs before ICP: ', bf_icp/general_counter)
                print ('Current ADDs after ICP: ', af_icp/general_counter)
                print ('Average time per image: ', img_avg_time_ms ,'ms\n')
                print ('Avg Acc space time ' , acc_time/general_counter/1000000 ,'ms\n')
                print ('Avg Network time ' , net_time/general_counter/1000000 ,'ms\n')
                print ('Avg ICP time ' , (img_avg_time_ms - acc_time/general_counter/1000000 - net_time/general_counter/1000000) ,'ms\n')

                #Save data distances to file
                with open('data/' + filename_without_ext + '.txt', 'a') as f:
                    f.write('Before ICP: \n')
                    f.write('Mean distance: ' + str(distance_bf_icp) + '\n')
                    f.write('Min distance: ' + str(min_distance_bf_icp) + '\n')
                    f.write('Max distance: ' + str(max_distance_bf_icp) + '\n')
                    f.write('Median distance: ' + str(median_distance_bf_icp) + '\n')
                    f.write('Standard deviation distance: ' + str(std_dev_distance_bf_icp) + '\n')
                    f.write('After ICP: \n')
                    f.write('Mean distance: ' + str(distance_af_icp) + '\n')
                    f.write('Min distance: ' + str(min_distance_af_icp) + '\n')
                    f.write('Max distance: ' + str(max_distance_af_icp) + '\n')
                    f.write('Median distance: ' + str(median_distance_af_icp) + '\n')
                    f.write('Standard deviation distance: ' + str(std_dev_distance_af_icp) + '\n')
                    f.write('-------------------------------------------------------------------\n')
                    f.close()



            
            count = count + 1
        #os.system("pause")
        
    print('ADD\(s\) of '+class_name+' before ICP: ', bf_icp/general_counter)
    print('ADD\(s\) of '+class_name+' after ICP: ', af_icp/general_counter)   
    print('Accumulator time: ', acc_time/general_counter)
          


if __name__ == "__main__":
    estimate_6d_pose_lm() 
