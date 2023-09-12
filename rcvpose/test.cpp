// test.cpp : Defines the entry point for the application.
//

#include "rcvpose.h"
#include <iostream>


using namespace std;

Options testing_options() {
    Options opts;
    opts.gpu_id = 0;
    opts.dname = "lm";
    opts.root_dataset = "/ingenuity_NAS/dataset/public/RCVLab/Bluewrist/16yw113/";
    //or ".../dataset/public/RCVLab/Bluewrist/16yw11"
    opts.model_dir = "/home/19lhs4/cpp_rcvpose/models/ape_with_geo_constraint/";
    opts.resume_train = false;
    opts.optim = "adam";
    opts.batch_size = 1;
    opts.class_name = "ape";
    opts.initial_lr = 0.0001;
    opts.reduce_on_plateau = false;
    opts.patience = 2;
    opts.demo_mode = false;
    opts.verbose = false;
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
