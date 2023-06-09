import torch as to
from torch import optim
from res_net import DenseFCNResNet152

class ModelLoader:
    def __init__(self, path):
        input_archive = to.serialization.load(path + "/model.pt")
        print(input_archive)
        self.model = DenseFCNResNet152(3,2)
        self.model = input_archive.load(path + "/model.pt")
        print(self.model)

        

def main():
    path = "C:/Users/User/.cw/work/cpp_rcvpose/rcvpose/out/build/x64-debug/train_kpt1/current"
    loader = ModelLoader(path)

if __name__ == "__main__":
    main()