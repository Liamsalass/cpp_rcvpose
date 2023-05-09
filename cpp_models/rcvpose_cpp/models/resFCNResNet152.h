
#pragma once

#include <torch/torch.h>
#include "bottleneck.h"

class ResFCNResNet152Impl : public torch::nn::Module {
	public:
		ResFCNResNet152Impl(int input_channels = 3, int output_channels = 2);
		std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

	private:
		int input_channels, output_channels;
		// Encoders
		torch::nn::Conv2d conv1;
		torch::nn::BatchNorm2d bn1;
		torch::nn::ReLU relu;
		torch::nn::MaxPool2d maxpool;
		Bottleneck block1up;
		torch::nn::Sequential block1;
		Bottleneck block2up;
		torch::nn::Sequential block2;
		Bottleneck block3up;
		torch::nn::Sequential block3;
		Bottleneck block4up;
		torch::nn::Sequential block4;
		// Decoders
		torch::nn::Sequential conv_up4;
		torch::nn::Conv2d conv_up4_1;
		torch::nn::Upsample up4;
		torch::nn::Sequential conv_up3;
		torch::nn::Conv2d conv_up3_1;
		torch::nn::Upsample up3;
		torch::nn::Sequential conv_up2;
		torch::nn::Conv2d conv_up2_1;
		torch::nn::Upsample up2;
		torch::nn::Sequential conv_up1;
		torch::nn::Conv2d conv_up1_1;
		torch::nn::Upsample up1;
		torch::nn::Sequential conv7;
		torch::nn::Conv2d conv8;
};

TORCH_MODULE(ResFCNResNet152);
