// test.cpp : Defines the entry point for the application.
//

#include "rcvpose.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

Options testing_options() {
    Options opts;
    opts.dname = "lm";
    opts.root_dataset = "C:/Users/User/.cw/work/datasets/test";
    //or ".../dataset/public/RCVLab/Bluewrist/16yw11"
    opts.model_dir = "C:/Users/User/.cw/work/cpp_rcvpose/gpu_models/ape_test";
    opts.resume_train = true;
    opts.optim = "adam";
    opts.batch_size = 2;
    opts.class_name = "ape";
    opts.initial_lr = 0.0001;
    opts.reduce_on_plateau = true;
    opts.patience = 10;
    opts.demo_mode = true;
    opts.verbose = false;
    opts.test_occ = false;
    opts.mask_threshold = 0.8;
    return opts;
}


int main(int argc, char* args[])
{
    //Options opts = testing_options();
    Options opts;
    if ((argc > 1) &&(argc < 14)) {
        try {
            opts.dname = args[1];

            opts.root_dataset = args[2];

            opts.model_dir = args[3];

            if (args[4] == "true") {
                opts.resume_train = true;
            }
            else {
                opts.resume_train = false;
            }
            opts.optim = args[5];

            opts.batch_size = stoi(args[6]);

            opts.class_name = args[7];

            opts.initial_lr = stod(args[8]);
            if (args[9] == "true") {
                opts.reduce_on_plateau = true;
            }
            else {
                opts.reduce_on_plateau = false;
            }
            opts.patience = stoi(args[10]);
            if (args[11] == "true") {
                opts.demo_mode = true;
            }
            else {
                opts.demo_mode = false;
            }
            if (args[12] == "true") {
                opts.verbose = true;
            }
            else {
                opts.verbose = false;
            }
            if (args[13] == "true") {
                opts.test_occ = true;
            }
            else {
                opts.test_occ = false;
            }
        }
        catch (const exception& e) {
            cout << "Error: " << e.what() << endl;
            cout << "Usage: " << args[0] << " <dataset_name(lm)> <dataset_root> <model_directory> <resume_train(true/false)> <optim(adam/sgd)> <batch_size(int)> <class_name(string)> <initial_lr(double)> <reduce_on_plateau(true/false)> <patience(int)> <demo_mode(true/false)> <verbose(true/false)> <test_occ(true/false)(hasn't been implemented yet)>" << endl;
            cout << "Defaulting to testing options" << endl;
            opts = testing_options();
        }
    }
    else {
        cout << "Using Default Testing Options" <<  endl;
        opts = testing_options();
    }


    RCVpose rcv(opts);
    //Trains the model with the given parameters, if resume if true, will resume training from previous saved state
    rcv.train();

    // Runs through the entire test dataset and prints the ADD before and after ICP as well as time taken
    rcv.validate();

    // Estimates the pose of a single input RGBD image and prints the estimated pose as well as time taken 
    //rcv.estimate_pose();


 
    for (int i = 0; i < 100; i++) {
        string img_num_str = to_string(i);  
    
        string padded_img_num = string(6 - img_num_str.length(), '0') + img_num_str;
    
        string img_path = "C:/Users/User/.cw/work/datasets/test/LINEMOD/ape/JPEGImages/" + padded_img_num + ".jpg";
        string depth_path = "C:/Users/User/.cw/work/datasets/test/LINEMOD_ORIG/ape/data/depth" + img_num_str + ".dpt";
        rcv.estimate_pose(img_path, depth_path);
    }
  

    return 0;
}