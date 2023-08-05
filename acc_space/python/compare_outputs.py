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
import concurrent.futures


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




def fileToTensor(filename):
    with open(filename, "rb") as file:
        num_dimensions = struct.unpack("Q", file.read(8))[0]
        shape = struct.unpack(f"{num_dimensions}q", file.read(8 * num_dimensions))
        num_elements = struct.unpack("Q", file.read(8))[0]
        tensor_data = struct.unpack(f"{num_elements}f", file.read(4 * num_elements))
        tensor = torch.tensor(tensor_data).reshape(shape)

    return tensor

def FCResBackbone(model, input_img_path, normalized_depth):
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

root_dataset = 'C:/Users/User/.cw/work/datasets/test/'
class_name = 'ape'
rootPath = root_dataset + "LINEMOD_ORIG/"+class_name+"/" 
rootpvPath = root_dataset + "LINEMOD/"+class_name+"/" 

test_list = open(root_dataset + "LINEMOD/"+class_name+"/" +"Split/val.txt","r").readlines()
test_list = [ s.replace('\n', '') for s in test_list]

pcd_load = o3d.io.read_point_cloud(root_dataset + "LINEMOD/"+class_name+"/"+class_name+".ply")

keypoints=np.load(root_dataset + "LINEMOD/"+class_name+"/"+"Outside9.npy")

dataPath = rootpvPath + 'JPEGImages/'

save_path = 'C:/Users/User/.cw/work/cpp_rcvpose/acc_space/python/'

model_dir = 'C:/Users/User/.cw/work/cpp_rcvpose/acc_space/python/pretrained/'

for kpt in range(3):
    print('Running for keypoint ' + str(kpt + 1))
    ground_truth_path = rootpvPath + 'Out_pt' + str(kpt + 1) +'_dm/'
    cpp_out = 'C:/Users/User/.cw/work/cpp_rcvpose/acc_space/python/kpt' + str(kpt+1) + '/tensors_cpp/'
    python_out = 'C:/Users/User/.cw/work/cpp_rcvpose/acc_space/python/kpt' + str(kpt+1) + '/tensors_py/'
    
    cpp_mean = []
    cpp_median = []
    cpp_std = []
    cpp_max = []
    cpp_min_val = []

    py_mean = []
    py_median = []
    py_std = []
    py_max = []
    py_min_val = []

    for img in test_list:
        count = int(os.path.splitext(img)[0])
        if ( kpt > 0 and count > 49):
            break
        cpp_radial_tensor_path = cpp_out + 'score_rad_' + str(count) + '.txt'
        cpp_semantic_tensor_path = cpp_out + 'score_' + str(count) + '.txt'
        py_radial_tensor_path = python_out + 'score_rad_' + str(count) + '.npy'
        py_semantic_tensor_path = python_out + 'score_' + str(count) + '.npy'
        radial_ground_truth_path = ground_truth_path + img + '.npy'


        cpp_radial_tensor = fileToTensor(cpp_radial_tensor_path)
        cpp_semantic_tensor = fileToTensor(cpp_semantic_tensor_path)
        cpp_semantic_tensor = np.where(cpp_semantic_tensor>0.8,1,0).squeeze(2).transpose(1,0)
        cpp_radial_tensor = np.array(cpp_radial_tensor).squeeze(2).transpose(1,0)

        py_radial_tensor = np.load(py_radial_tensor_path)
        py_semantic_tensor = np.load(py_semantic_tensor_path)
        py_semantic_tensor = np.where(py_semantic_tensor>0.8,1,0)
        py_radial_tensor = np.array(py_radial_tensor)

        radial_ground_truth = np.load(radial_ground_truth_path)
      
        cpp_radial_w_sem = cpp_radial_tensor * cpp_semantic_tensor
        py_radial_w_sem = py_radial_tensor * py_semantic_tensor

        # Flatten data
        cpp_radial_w_sem = cpp_radial_w_sem.flatten()
        py_radial_w_sem = py_radial_w_sem.flatten()
        radial_ground_truth = radial_ground_truth.flatten()

        # Remove zeros
        cpp_radial_w_sem = cpp_radial_w_sem[cpp_radial_w_sem != 0]
        py_radial_w_sem = py_radial_w_sem[py_radial_w_sem != 0]
        radial_ground_truth = radial_ground_truth[radial_ground_truth != 0]

        # Check if sizes match
        if cpp_radial_w_sem.shape != py_radial_w_sem.shape or cpp_radial_w_sem.shape != radial_ground_truth.shape:
            min_size = min(cpp_radial_w_sem.shape[0], py_radial_w_sem.shape[0], radial_ground_truth.shape[0])
            cpp_radial_w_sem = cpp_radial_w_sem[:min_size]
            py_radial_w_sem = py_radial_w_sem[:min_size]
            radial_ground_truth = radial_ground_truth[:min_size]

        if cpp_radial_w_sem.size == 0:
            print('cpp_radial_w_sem is empty')
            continue
        
        if py_radial_w_sem.size == 0:
            print('py_radial_w_sem is empty')
            continue

        cpp_diff_w_sem = np.abs(cpp_radial_w_sem - radial_ground_truth)
        py_diff_w_sem = np.abs(py_radial_w_sem - radial_ground_truth)
        
        cpp_diff_w_sem = np.divide(cpp_diff_w_sem,radial_ground_truth)
        py_diff_w_sem = np.divide(py_diff_w_sem,radial_ground_truth)

        # Save the mean, median, std deviation, max and min
        cpp_mean.append(np.mean(cpp_diff_w_sem))
        cpp_median.append(np.median(cpp_diff_w_sem))
        cpp_std.append(np.std(cpp_diff_w_sem))
        cpp_max.append(np.max(cpp_diff_w_sem))
        cpp_min_val.append(np.min(cpp_diff_w_sem))

        py_mean.append(np.mean(py_diff_w_sem))
        py_median.append(np.median(py_diff_w_sem))
        py_std.append(np.std(py_diff_w_sem))
        py_max.append(np.max(py_diff_w_sem))
        py_min_val.append(np.min(py_diff_w_sem))

        if(count == 1049):
            break
        

    # Print the average mean, median, std deviation, max and min
    print('Printing average values for keypoint ' + str(kpt + 1))
    print('CPP:')
    print('\tMean: ' + str(np.mean(cpp_mean)))
    print('\tMedian: ' + str(np.mean(cpp_median)))
    print('\tStd: ' + str(np.mean(cpp_std)))
    print('\tMax: ' + str(np.mean(cpp_max)))
    print('\tMin: ' + str(np.mean(cpp_min_val)))
    print('Python:')
    print('\tMean: ' + str(np.mean(py_mean)))
    print('\tMedian: ' + str(np.mean(py_median)))
    print('\tStd: ' + str(np.mean(py_std)))
    print('\tMax: ' + str(np.mean(py_max)))
    print('\tMin: ' + str(np.mean(py_min_val)))
    print('')


        