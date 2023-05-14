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
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512 + 512, 256, 3).padding(1).stride(1)),
		torch::nn::BatchNorm2d(256),
		torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
	));
	up3(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2 })).mode(torch::kBilinear).align_corners(false));
	conv_up2(torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256 + 256, 128, 3).padding(1).stride(1)),
		torch::nn::BatchNorm2d(128),
		torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
	));
	up2(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(false));
	conv_up1(torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(64 + 128, 64, 3).padding(1).stride(1)),
		torch::nn::BatchNorm2d(64),
		torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
	));
	up1(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2 })).mode(torch::kBilinear).align_corners(false));
	conv7(torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 3).padding(1).stride(1)),
		torch::nn::BatchNorm2d(32),
		torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
	));
	conv8(torch::nn::Conv2dOptions(32, output_channels, 1).padding(0).stride(1));
}

std::tuple<torch::Tensor, torch::Tensor> DenseFCNResNet152Impl::forward(torch::Tensor x)
{
	x = conv1->forward(x);
	x = bn1->forward(x);
	auto x2s = relu(x);
	x2s = maxpool(x2s);

	x2s = block1up->forward(x);
	x2s = block1->forward(x);

	auto x4s = block2up->forward(x2s);
	x4s = block2->forward(x4s);

	auto x8s = block3up->forward(x4s);
	x8s = block3->forward(x8s);

	auto x16s = block4up->forward(x8s);
	x16s = block4->forward(x16s);

	auto x32s = conv6->forward(x16s);
	x32s = bn6->forward(x32s);
	x32s = relu(x32s);

	auto up = conv_up5->forward(torch::cat({ x32s, x16s }, 1));
	up = up5->forward(up);

	up = conv_up4->forward(torch::cat({ up, x8s }, 1));
	up = up4->forward(up);

	up = conv_up3->forward(torch::cat({ up, x4s }, 1));
	up = up3->forward(up);

	up = conv_up2->forward(torch::cat({ up, x2s }, 1));
	up = up2->forward(up);

	up = conv_up1->forward(torch::cat({ up, x }, 1));
	up = up1->forward(up);

	up = conv7->forward(up);

	auto out = conv8->forward(up);
	auto seg_pred = out.index({ torch::indexing::Slice(), 0, torch::indexing::Slice(), torch::indexing::Slice() });
	auto radial_pred = out.index({ torch::indexing::Slice(), torch::indexing::Slice(1, nullptr), torch::indexing::Slice(), torch::indexing::Slice() });

	return std::make_tuple(seg_pred, radial_pred);

}