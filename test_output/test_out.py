import struct
import torch
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def fileToTensor(filename):
    with open(filename, "rb") as file:
        num_dimensions = struct.unpack("Q", file.read(8))[0]
        shape = struct.unpack(f"{num_dimensions}q", file.read(8 * num_dimensions))
        num_elements = struct.unpack("Q", file.read(8))[0]
        tensor_data = struct.unpack(f"{num_elements}f", file.read(4 * num_elements))
        tensor = torch.tensor(tensor_data).reshape(shape)
    return tensor

input_img = fileToTensor(r'epoch_11\input_img0.txt')
input_img = input_img[0]
input_img = input_img.permute(1, 2, 0)

rad_target = fileToTensor(r'epoch_11\input_rad0.txt')
rad_target = rad_target[0]
rad_target = rad_target.permute(1, 2, 0)

sem_target = fileToTensor(r'epoch_11\input_sem0.txt')
sem_target = sem_target[0]
sem_target = sem_target.permute(1, 2, 0)

rad_output = fileToTensor(r'epoch_11\output_rad.txt')
rad_output = rad_output[0]
rad_output = rad_output.permute(1, 2, 0)

sem_output = fileToTensor(r'epoch_11\output_sem.txt')
sem_output = sem_output[0]
sem_output = sem_output.permute(1, 2, 0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(rad_target)
axes[0].set_title("Radial Target")
axes[1].imshow(sem_target)
axes[1].set_title("Semantic Target")
axes[2].imshow(input_img)
axes[2].set_title("Input Image")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(rad_output)
axes[0].set_title("Radial Output")
axes[1].imshow(sem_output)
axes[1].set_title("Semantic Output")
axes[2].imshow(input_img)
axes[2].set_title("Input Image")
plt.show()
