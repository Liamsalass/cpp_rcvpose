// test.cpp : Defines the entry point for the application.
//

#include "rcvpose.h"
#include <iostream>


using namespace std;

Options testing_options() {
    Options opts;
    opts.gpu_id = 0;
    opts.dname = "lm";
    opts.root_dataset = "C:/Users/User/.cw/work/datasets/test";
    //or ".../dataset/public/RCVLab/Bluewrist/16yw11"
    opts.model_dir = "C:/Users/User/.cw/work/cpp_rcvpose/rcvpose/test_out/ape/";
    opts.resume_train = true;
    opts.optim = "adam";
    opts.batch_size = 2;
    opts.class_name = "ape";
    opts.initial_lr = 0.0001;
    opts.reduce_on_plateau = false;
    opts.patience = 10;
    opts.demo_mode = false;
    opts.verbose = true;
    opts.test_occ = false;
    return opts;
}

int main(int argc, char* args[])
{
    Options opts = testing_options();
  
    RCVpose rcv(opts);
    rcv.train();
    

    return 0;    
}
