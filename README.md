# cpp_rcvpose
Liam Salass

## Task

Port python pytorch models from https://github.com/aaronWool/rcvpose and https://github.com/aaronWool/rcvpose3d into a C++. 

## Libraries

```
conda create -n port python=3.9 tqdm pandas numpy matplotlib torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
```
to activate 
```
conda activate port
```
required pip library
```
pip install pytorch-model-summary torchsummary
```
Delete the library
```
conda deactivate
conda env remove --name port
```