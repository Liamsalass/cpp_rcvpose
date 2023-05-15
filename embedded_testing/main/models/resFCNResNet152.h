
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
	torch::nn::Conv2d conv1{ nullptr };
	torch::nn::BatchNorm2d bn1{ nullptr };
	torch::nn::ReLU relu{ nullptr };
	torch::nn::MaxPool2d maxpool{ nullptr };
	Bottleneck block1up{ nullptr };
	torch::nn::Sequential block1{ nullptr };
	Bottleneck block2up{ nullptr };
	torch::nn::Sequential block2{ nullptr };
	Bottleneck block3up{ nullptr };
	torch::nn::Sequential block3{ nullptr };
	Bottleneck block4up{ nullptr };
	torch::nn::Sequential block4{ nullptr };
	// Decoders
	torch::nn::Sequential conv_up4{ nullptr };
	torch::nn::Conv2d conv_up4_1{ nullptr };
	torch::nn::Upsample up4{ nullptr };
	torch::nn::Sequential conv_up3{ nullptr };
	torch::nn::Conv2d conv_up3_1{ nullptr };
	torch::nn::Upsample up3{ nullptr };
	torch::nn::Sequential conv_up2{ nullptr };
	torch::nn::Conv2d conv_up2_1{ nullptr };
	torch::nn::Upsample up2{ nullptr };
	torch::nn::Sequential conv_up1{ nullptr };
	torch::nn::Conv2d conv_up1_1{ nullptr };
	torch::nn::Upsample up1{ nullptr };
	torch::nn::Sequential conv7{ nullptr };
	torch::nn::Conv2d conv8{ nullptr };
};

TORCH_MODULE(ResFCNResNet152);
