#ifndef BOTTLENECK_H
#define BOTTLENECK_H

#include <torch/torch.h>

// This code defines a BottleneckImpl class that inherits from the PyTorch torch::nn::Module class.
// The class implements a ResNet bottleneck block used in deep neural networks for image classification and other computer vision tasks.
// The constructor takes four arguments : in_channels(integer), channels(integer), stride(integer), and upsample(boolean).
// The upsample argument is used to determine if an upsampling operation needs to be performed on the input tensor before the skip connection is added back to the main path.
// Inside the constructor, the class initializes several PyTorch layers, including three convolutional layers(conv1, conv2, and conv3), three batch normalization layers(bn1, bn2, and bn3), and a ReLU activation layer(relu).
// The register_module() method is called to register each layer as a sub - module of the BottleneckImpl class.
// If upsample is true, an additional convolutional layer and batch normalization layer are initialized to create the upsampling block.
// The forward() method takes a tensor x as input and applies the three convolutional layers(conv1, conv2, and conv3) with batch normalization(bn1, bn2, and bn3) and ReLU activation(relu) to x in sequence.
// If upsample is true, the input tensor residual is passed through the upsample_ block to perform upsampling, and the result is added back to the main path. 
// Finally, the output tensor is passed through the ReLU activation function and returned.


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
