// test.cpp : Defines the entry point for the application.
//

#include "rcvpose.h"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;

Options testing_options() {
    Options opts;
    opts.dname = "lm";
    opts.root_dataset = "C:/Users/User/.cw/work/datasets/test";
    //or ".../dataset/public/RCVLab/Bluewrist/16yw11"
    opts.model_dir = "C:/Users/User/.cw/work/cpp_rcvpose/gpu_models/ape2.0";
    opts.resume_train = true;
    opts.optim = "adam";
    opts.frontend = "ransac";
    opts.batch_size = 1;
    opts.class_name = "ape";
    opts.initial_lr = 0.0001;
    opts.reduce_on_plateau = true;
    opts.patience = 10;
    opts.demo_mode = false;
    opts.verbose = true;
    opts.test_occ = false;
    opts.mask_threshold = 0.8;
    opts.epsilon = 0.02;
    opts.use_gt = false;
    return opts;
}

//Note cam Failed at img 86 which is number img99 in dataset

int main(int argc, char* args[]) {
    bool train = false;
    bool validate = false;
    bool estimate = false;

    if (argc > 1) {
        if (strcmp(args[1], "train") == 0) {
            train = true;
        }
        else if (strcmp(args[1], "validate") == 0) {
            validate = true;
        }
        else if (strcmp(args[1], "estimate") == 0) {
            estimate = true;
        }
        else {
            cout << "Usage: " << args[0] << " <train/validate/estimate>" << endl;
            return 0;
        }
    }
    else {
        cout << "Usage: " << args[0] << " <train/validate/estimate>" << endl;
        cout << "Defaulting to validating" << endl;
        validate = true;
    }

    Options opts;

    if (argc > 2 && argc < 15) {
        try {
            opts.dname = args[2];
            opts.root_dataset = args[3];
            opts.model_dir = args[4];
            opts.resume_train = (strcmp(args[5], "true") == 0);
            opts.optim = args[6];
            opts.batch_size = stoi(args[7]);
            opts.class_name = args[8];
            opts.initial_lr = stod(args[9]);
            opts.reduce_on_plateau = (strcmp(args[10], "true") == 0);
            opts.patience = stoi(args[11]);
            opts.demo_mode = (strcmp(args[12], "true") == 0);
            opts.verbose = (strcmp(args[13], "true") == 0);
            opts.test_occ = (strcmp(args[14], "true") == 0);
        }
        catch (const exception& e) {
            cout << "Error: " << e.what() << endl;
            cout << "Usage: " << args[0] << " <dataset_name(lm)> <dataset_root> <model_directory> <resume_train(true/false)> <optim(adam/sgd)> <batch_size(int)> <class_name(string)> <initial_lr(double)> <reduce_on_plateau(true/false)> <patience(int)> <demo_mode(true/false)> <verbose(true/false)> <test_occ(true/false)(hasn't been implemented yet)>" << endl;
            cout << "Defaulting to testing options" << endl;
            opts = testing_options();
        }
    }
    else {
        cout << "Using Default Testing Options" << endl;
        opts = testing_options();
    }


    RCVpose rcv(opts);
    //Trains the model with the given parameters, if resume if true, will resume training from previous saved state
    if (train)
        rcv.train();

    // Runs through the entire test dataset and prints the ADD before and after ICP as well as time taken
    if (validate)
        rcv.validate();

    // Estimates the pose of a single input RGBD image and prints the estimated pose as well as time taken 
    if(estimate){
        for (int i = 0; i < 100; i++) {
            string img_num_str = to_string(i);  

            string padded_img_num = string(6 - img_num_str.length(), '0') + img_num_str;

            string img_path = "C:/Users/User/.cw/work/datasets/test/LINEMOD/ape/JPEGImages/" + padded_img_num + ".jpg";
            string depth_path = "C:/Users/User/.cw/work/datasets/test/LINEMOD_ORIG/ape/data/depth" + img_num_str + ".dpt";
            rcv.estimate_pose(img_path, depth_path);
        }
        return 0;
    }

    return 0;
}