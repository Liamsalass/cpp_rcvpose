
#include "resFCNResNet152.h"
namespace nn = torch::nn;

ResFCNResNet152Impl::ResFCNResNet152Impl(int input_channels, int output_channels) :
	input_channels(input_channels),
	output_channels(output_channels)
{
	// Encoders
	nn::Conv2d conv1(nn::Conv2dOptions(input_channels, 64, 7).stride(2).padding(3).bias(false));
	nn::BatchNorm2d bn1(64);
	nn::ReLU relu(nn::ReLUOptions().inplace(true));
	nn:: MaxPool2d maxpool(nn::MaxPool2dOptions(3).stride(2).padding(1));
	Bottleneck block1up(64, 64, 1, true);
	nn::Sequential block1(nn::Sequential(Bottleneck(256, 64, 1), Bottleneck(256, 64, 1), Bottleneck(256, 64, 1)));
	Bottleneck block2up(256, 128, 2, true);
	nn::Sequential block2(nn::Sequential(Bottleneck(512, 128, 1), Bottleneck(512, 128, 1),
		Bottleneck(512, 128, 1), Bottleneck(512, 128, 1), Bottleneck(512, 128, 1),
		Bottleneck(512, 128, 1), Bottleneck(512, 128, 1), Bottleneck(512, 128, 1)));
	Bottleneck block3up(512, 256, 2, true);
	nn::Sequential block3(nn::Sequential(Bottleneck(1024, 256, 1), Bottleneck(1024, 256, 1),
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
	Bottleneck block4up(1024, 512, 2, true);
	nn::Sequential block4(nn::Sequential(Bottleneck(2048, 512, 1), Bottleneck(2048, 512, 1), Bottleneck(2048, 512, 1)));

	// Decoders
	nn::Conv2d conv_up4_1(nn::Conv2dOptions(1024, 2048, 1));
	nn::Sequential conv_up4(nn::Sequential(
		nn::Conv2d(nn::Conv2dOptions(2048, 1024, 3).stride(1).padding(1)),
		nn::BatchNorm2d(1024),
		nn::ReLU(true)
	));
	nn::Upsample up4(nn::UpsampleOptions().scale_factor(std::vector<double>({ 2 })).mode(torch::kBilinear).align_corners(false));

	nn::Conv2d conv_up3_1(nn::Conv2dOptions(512, 1024, 1));
	nn::Sequential conv_up3(nn::Sequential(
		nn::Conv2d(nn::Conv2dOptions(1024, 512, 3).stride(1).padding(1)),
		nn::BatchNorm2d(512),
		nn::ReLU(true)
	));
	nn::Upsample up3(nn::UpsampleOptions().scale_factor(std::vector<double>({ 2 })).mode(torch::kBilinear).align_corners(false));

	nn::Conv2d conv_up2_1(nn::Conv2dOptions(256, 512, 1));
	nn::Sequential conv_up2(nn::Sequential(
		nn::Conv2d(nn::Conv2dOptions(512, 256, 3).stride(1).padding(1)),
		nn::BatchNorm2d(256),
		nn::ReLU(true)
	));
	nn::Upsample up2(nn::UpsampleOptions().scale_factor(std::vector<double>({ 2 })).mode(torch::kBilinear).align_corners(false));

	nn::Conv2d conv_up1_1(nn::Conv2dOptions(64, 256, 1));
	nn::Sequential conv_up1(nn::Sequential(
		nn::Conv2d(nn::Conv2dOptions(256, 128, 3).stride(1).padding(1)),
		nn::BatchNorm2d(128),
		nn::ReLU(true)
	));
	nn::Upsample up1(nn::UpsampleOptions().scale_factor(std::vector<double>({ 2 })).mode(torch::kBilinear).align_corners(false));

	nn::Sequential conv7(nn::Sequential(
		nn::Conv2d(nn::Conv2dOptions(128, 64, 3).stride(1).padding(1)),
		nn::BatchNorm2d(64),
		nn::ReLU(true)
	));

	nn::Conv2d conv8(nn::Conv2dOptions(64, output_channels, 1).stride(1));

	//Register Components
	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("relu", relu);
	register_module("maxpool", maxpool);
	register_module("block1up", block1up);
	register_module("block1", block1);
	register_module("block2up", block2up);
	register_module("block2", block2);
	register_module("block3up", block3up);
	register_module("block3", block3);
	register_module("block4up", block4up);
	register_module("block4", block4);
	register_module("conv_up4_1", conv_up4_1);
	register_module("conv_up4", conv_up4);
	register_module("up4", up4);
	register_module("conv_up3_1", conv_up3_1);
	register_module("conv_up3", conv_up3);
	register_module("up3", up3);
	register_module("conv_up2_1", conv_up2_1);
	register_module("conv_up2", conv_up2);
	register_module("up2", up2);
	register_module("conv_up1_1", conv_up1_1);
	register_module("conv_up1", conv_up1);
	register_module("up1", up1);
	register_module("conv7", conv7);
	register_module("conv8", conv8);
}

std::tuple<torch::Tensor, torch::Tensor> ResFCNResNet152Impl::forward(torch::Tensor x)
{
	x = conv1->forward(x);
	x = bn1->forward(x);
	auto x2s = relu(x);
	x2s = maxpool(x2s);

	x2s = block1up->forward(x2s);
	x2s = block1->forward(x2s);

	auto x4s = block2up->forward(x2s);
	x4s = block2->forward(x4s);

	auto x8s = block3up->forward(x4s);
	x8s = block3->forward(x8s);

	auto x16s = block4up->forward(x8s);
	x16s = block4->forward(x16s);

	auto up = up4->forward(x16s);
	up = conv_up4->forward(up + conv_up4_1(x8s));

	up = up3->forward(up);
	up = conv_up3->forward(up + conv_up3_1(x4s));

	up = up2->forward(up);
	up = conv_up2->forward(up + conv_up2_1(x2s));

	up = up1->forward(up);
	up = conv_up1->forward(up + conv_up1_1(x));
	up = up1->forward(up);

	up = conv7->forward(up);
	auto out = conv8->forward(up);

	// seg_pred = out[:,:1,:,:]
	// radial_pred = out[:,1:,:,:]

	auto seg_pred = out.index({ 
		torch::indexing::Slice(),
		0,
		torch::indexing::Slice(), 
		torch::indexing::Slice()
	});

	auto radial_pred = out.index({ 
		torch::indexing::Slice(), 
		torch::indexing::Slice(1, torch::indexing::None),
		torch::indexing::Slice(), 
		torch::indexing::Slice() 
	});

	return std::make_tuple(seg_pred, radial_pred);

}
