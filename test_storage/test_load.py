import torch
import struct

def fileToTensor(filename):
    with open(filename, "rb") as file:
        # Read the number of dimensions from the file
        num_dimensions = struct.unpack("Q", file.read(8))[0]

        # Read the shape of the tensor from the file
        shape = struct.unpack(f"{num_dimensions}q", file.read(8 * num_dimensions))

        # Read the number of elements from the file
        num_elements = struct.unpack("Q", file.read(8))[0]

        # Read the tensor data from the file
        tensor_data = struct.unpack(f"{num_elements}f", file.read(4 * num_elements))

        # Create a torch.Tensor with the retrieved shape and data
        tensor = torch.tensor(tensor_data).reshape(shape)

    return tensor

# Specify the filename of the file generated by the C++ code
filename = "data.bin"

# Read the tensor from the file
tensor = fileToTensor(filename)

# Print the retrieved tensor
print(tensor)
