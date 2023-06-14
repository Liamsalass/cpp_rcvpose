
import torch
import torchvision


def main():
    #read in tensor.pt file, it was stored by c++
    tensor = torch.load("tensor.pt")
    print("Tensor Info: ")
    print(tensor)

# Call to main
if __name__ == '__main__':
    main()
