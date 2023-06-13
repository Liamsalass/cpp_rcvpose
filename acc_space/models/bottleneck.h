#ifndef BOTTLENECK_H
#define BOTTLENECK_H

#include <torch/torch.h>

class BottleneckImpl : public torch::nn::Module {
public:

    BottleneckImpl(int in_channels, int channels, int stride = 1, bool upsample = false);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1, conv2, conv3;
    torch::nn::BatchNorm2d bn1, bn2, bn3;
    torch::nn::Sequential upsample_;
    torch::nn::ReLU relu;
    bool upsample;
};

TORCH_MODULE(Bottleneck);

#endif // BOTTLENECK_H
