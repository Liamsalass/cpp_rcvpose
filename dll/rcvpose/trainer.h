#pragma once

#include "options.hpp"
#include "torch/torch.h"
#include "data_loader.h"
#include <iostream>


class Trainer {
public:
	Trainer(Options options);

	void train();
	void test();
	void demo();

private:
	Options opts;
	double compute_r_loss(torch::Tensor pred, torch::Tensor gt);

};