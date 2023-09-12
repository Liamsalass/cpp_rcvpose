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
    bool train = false;


    if(argc > 1){
        if (strcmp(args[1], "train") == 0) {
            train = true;
        }
        else if (strcmp(args[1], "validate") == 0){
            cout << "Validation not functioning: defaulting to training" << endl;
            train = true;
        }
        else {
            cout << "Usage: " << args[0] << " <train/validate>" << endl;
            return 0;
        }
    } else {
        cout << "Usage: " << args[0] << " <train/validate>" << endl;
        cout << "Defaulting to train" << endl;
        train = true;
    }

    Options opts;
    if ((argc > 2) &&(argc <= 15)) {
        try {
            opts.dname = args[2];

            opts.root_dataset = args[3];

            opts.model_dir = args[4];

            if (args[5] == "true") {
                opts.resume_train = true;
            }
            else {
                opts.resume_train = false;
            }
            opts.optim = args[6];

            opts.batch_size = stoi(args[7]);

            opts.class_name = args[8];

            opts.initial_lr = stod(args[9]);
            if (args[10] == "true") {
                opts.reduce_on_plateau = true;
            }
            else {
                opts.reduce_on_plateau = false;
            }
            opts.patience = stoi(args[11]);

            if (args[12] == "true") {
                opts.demo_mode = true;
            }
            else {
                opts.demo_mode = false;
            }
            if (args[13] == "true") {
                opts.verbose = true;
            }
            else {
                opts.verbose = false;
            }
            if (args[14] == "true") {
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
    if (train){
        rcv.train();
    }

    return 0;
}