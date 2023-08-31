import numpy as np
import os

path = 'C:/Users/User/.cw/work/datasets/test/LINEMOD/duck/pose/'

save_path = 'C:/Users/User/.cw/work/datasets/test/LINEMOD/duck/pose_txt/'

for filename in os.listdir(path):
    pose = path + filename
    pose = np.load(pose)
    filename = filename[:-4]
    np.savetxt(save_path + filename + '.txt', pose, delimiter=',', fmt='%f')
    print('Saved ' + filename + '.txt')


    

