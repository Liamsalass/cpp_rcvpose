// test.cpp : Defines the entry point for the application.
//

#include "rcvpose.h"
#include <iostream>
#include "torch/torch.h"
#include "opencv2/opencv.hpp"

using namespace std;



Options testing_options(int kpt = 1, bool rsm = false, int batchsize = 2, string out_dir = "kpt1") {
    Options opts;
    opts.gpu_id = 0;
    opts.dname = "lm";
    opts.root_dataset = "C:/Users/User/.cw/work/datasets/test";
    //or ".../dataset/public/RCVLab/Bluewrist/16yw11"
    opts.model_dir = out_dir;
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
    return opts;
}

Options training_options(int kpt, bool rsm, int batch_size = 2) {
	Options opts;
	opts.gpu_id = 0;
	opts.dname = "lm";
	opts.root_dataset = "C:/Users/User/.cw/work/datasets/test";
    //or ".../dataset/public/RCVLab/Bluewrist/16yw11"
	opts.model_dir = "train_kpt" + to_string(kpt);
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
	return opts;
}

int main(int argc, char* args[])
{
	Options opts1 = gpu_train_opts(1, true, 24, 0);
	RCVpose rcv1(opts1);
	rcv1.save_tensor("kpt1_t", 0, 1050);

    Options opts2 = gpu_train_opts(2, true, 24, 0);
    RCVpose rcv2(opts2);
    rcv2.save_tensor("kpt2_t", 0, 1050);

    Options opts3 = gpu_train_opts(3, true, 24, 0);
    RCVpose rcv3(opts3);
    rcv2.save_tensor("kpt3_t", 0, 1050);
    
}
