#pragma once
#ifndef BOTTLENECK_H
#define BOTTLENECK_H

#include <torch/torch.h>

class BottleneckImpl : public torch::nn::Module {
public:
    BottleneckImpl(int in_channels, int channels, int stride = 1, bool upsample = false);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv3{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr }, bn2{ nullptr }, bn3{ nullptr };
    torch::nn::Sequential upsample_{ nullptr };
    torch::nn::ReLU relu{ nullptr };
    bool upsample;
};

TORCH_MODULE(Bottleneck);

#endif // BOTTLENECK_H
