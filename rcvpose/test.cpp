// test.cpp : Defines the entry point for the application.
//

#include "rcvpose.h"
#include <iostream>
#include "torch/torch.h"
#include "opencv2/opencv.hpp"

using namespace std;

Options testing_options(int kpt = 3, bool rsm = true, int batchsize = 1, string output_directory = "kpt3", bool debug = false) {
    Options opts;
    opts.gpu_id = 0;
    opts.dname = "lm";
    opts.root_dataset = "C:/Users/User/.cw/work/datasets/test";
    //or ".../dataset/public/RCVLab/Bluewrist/16yw11"
    opts.model_dir = output_directory;
    opts.resume_train = rsm;
    opts.optim = "adam";
    opts.batch_size = batchsize;
    opts.class_name = "ape";
    opts.initial_lr = 0.0001;
    opts.reduce_on_plateau = false;
    opts.kpt_num = kpt;
    opts.demo_mode = true;
    opts.test_occ = false;
    opts.patience = 10;
    opts.debug = debug;
    return opts;
}

Options training_options(int kpt = 1, bool rsm = false, int batchsize = 2) {
    Options opts;
    opts.gpu_id = 0;
    opts.dname = "lm";
    opts.root_dataset = "C:/Users/User/.cw/work/datasets/test";
    //or ".../dataset/public/RCVLab/Bluewrist/16yw11"
    opts.model_dir = "kpt" + to_string(kpt);
    opts.resume_train = rsm;
    opts.optim = "adam";
    opts.batch_size = batchsize;
    opts.class_name = "ape";
    opts.initial_lr = 0.0001;
    opts.reduce_on_plateau = false;
    opts.kpt_num = kpt;
    opts.demo_mode = false;
    opts.test_occ = false;
    opts.patience = 10;
    opts.debug = false;
    return opts;
}


Options gpu_train_opts(int kpt, bool rsm, int batch_size, int gpuid) {
    Options opts;
	opts.gpu_id = gpuid;
	opts.dname = "lm";
	opts.root_dataset = "/ingenuity_NAS/dataset/public/RCVLab/Bluewrist/16yw113";
	opts.model_dir = "kpt" + to_string(kpt);
	opts.resume_train = rsm;
	opts.optim = "adam";
	opts.batch_size = batch_size;
	opts.class_name = "ape";
	opts.initial_lr = 0.0001;
	opts.reduce_on_plateau = false;
	opts.kpt_num = kpt;
	opts.demo_mode = false;
	opts.test_occ = false;
    opts.patience = 10;    
    opts.debug = false;
	return opts;
}

int main(int argc, char* args[])
{
    Options opts = testing_options(1, true, 1, "kpt1", false);
    RCVpose rcv(opts);
    rcv.save_tensor("tensors", 0, 1050);
}
