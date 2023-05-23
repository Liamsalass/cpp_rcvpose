
#include "bottleneck.h"


BottleneckImpl::BottleneckImpl(int in_channels, int channels, int stride, bool upsample) :
    // Initialize class variables
    upsample(upsample),
    conv1(torch::nn::Conv2dOptions(in_channels, channels, 1).bias(false)),
    bn1(channels),
    conv2(torch::nn::Conv2dOptions(channels, channels, 3).stride(stride).padding(1)),
    bn2(channels),
    conv3(torch::nn::Conv2dOptions(channels, channels * 4, 1).bias(false)),
    bn3(channels * 4),
    relu(torch::nn::ReLUOptions().inplace(true)),
    upsample_(nullptr)
{
    // Initialize layers
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    register_module("conv3", conv3);
    register_module("bn3", bn3);
    register_module("relu", relu);

    // If upsample is true, initialize the upsampling block
    if (upsample) {
        upsample_ = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, channels * 4, 1).stride(stride).bias(false)),
            torch::nn::BatchNorm2d(channels * 4)
        );
        register_module("upsample_", upsample_);
    }
}

torch::Tensor BottleneckImpl::forward(torch::Tensor x) {
    torch::Tensor residual = x;

    x = conv1(x);
    x = bn1(x);
    x = relu(x);

    x = conv2(x);
    x = bn2(x);
    x = relu(x);

    x = conv3(x);
    x = bn3(x);

    if (upsample) {
        residual = upsample_->forward(residual);
    }

    x += residual;
    x = relu(x);

    return x;
}
