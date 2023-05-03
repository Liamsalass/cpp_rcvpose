#include "denseFCNResNet152.h"

DenseFCNResNet152Impl::DenseFCNResNet152Impl(int input_channels, int output_channels) :
	input_channels(input_channels),
	output_channels(output_channels)
{
	// Initialize resnet encoders
	// conv1 deviates from the original implementation by having input channels as a parameter, instead of being hardcoded to 3
	conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, 64, 7).stride(2).padding(3).bias(false));
	bn1(64);
	relu(torch::nn::ReLUOptions().inplace(true));
	maxpool(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
	block1up(64, 64, 1, true);
	block1(torch::nn::Sequential(Bottleneck(256, 64, 1), Bottleneck(256, 64, 1), Bottleneck(256, 64, 1)));
	block2up(256, 128, 2, true);
	block2(torch::nn::Sequential(Bottleneck(512, 128, 1), Bottleneck(512, 128, 1),
		Bottleneck(512, 128, 1), Bottleneck(512, 128, 1), Bottleneck(512, 128, 1),
		Bottleneck(512, 128, 1), Bottleneck(512, 128, 1), Bottleneck(512, 128, 1)));
	block3up(512, 256, 2, true);
	block3(torch::nn::Sequential(Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1),
		Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1),
		Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1),
		Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1),
		Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1),
		Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1),
		Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1),
		Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1),
		Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1),
		Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1),
		Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1),
		Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1),
		Bottleneck(1024, 256, 1)));
	block4up(1024, 512, 2, true);
	block4(torch::nn::Sequential(Bottleneck(2048, 512, 1), Bottleneck(2048, 512, 1), Bottleneck(2048, 512, 1)));
	conv6(torch::nn::Conv2dOptions(2048, 1024, 3).stride(1).padding(1));
	bn6(1024);
	// Initialize decoders
	conv_up5(torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(2048 + 1024, 1024, 3).padding(1)),
		torch::nn::BatchNorm2d(1024),
		torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
	));
	//Why does scale factor have to be a vector?
	up5(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2 })).mode(torch::kBilinear).align_corners(false));
	conv_up4(torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(1024 + 1024, 512, 3).padding(1).stride(1)),
		torch::nn::BatchNorm2d(512),
		torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
	));
	up4(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2 })).mode(torch::kBilinear).align_corners(false));
	conv_up3(torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512+512, 256, 3).padding(1).stride(1)),
		torch::nn::BatchNorm2d(256),
		torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
	));
	up3(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2 })).mode(torch::kBilinear).align_corners(false));

	

}

torch::Tensor DenseFCNResNet152Impl::forward(torch::Tensor x)
{

	return torch::Tensor();
}
