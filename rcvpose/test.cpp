// test.cpp : Defines the entry point for the application.
//

#include "rcvpose.h"
#include <iostream>
#include "torch/torch.h"
#include "opencv2/opencv.hpp"

using namespace std;

//Set true to load in a model and resume, doens't start any programs
#define rsm_check true

//Check library functionality
#define lib_check false

//Check devices available and GPU compatibility
#define dev_check false

//Check RCVpose class functionality
#define rcv_check false

//Check loader functionality
#define ldr_check false

// Check training functionality
#define trn_check true

// Check saving and loading model
#define stw_model false

// Train checkpoint
#define trn_ckpt false
#define trn_gpu false

Options testing_options() {
    Options opts;
    opts.gpu_id = 0;
    opts.dname = "lm";
    opts.root_dataset = "C:/Users/User/.cw/work/datasets/test";
    //or ".../dataset/public/RCVLab/Bluewrist/16yw11"
    opts.model_dir = "train_kpt1";
    opts.resume_train = rsm_check;
    opts.optim = "adam";
    opts.batch_size = 2;
    opts.class_name = "ape";
    opts.initial_lr = 0.0001;
    opts.reduce_on_plateau = false;
    opts.kpt_num = 1;
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

Options gpu_train_opts(int kpt, bool rsm, int batch_size) {
    Options opts;
	opts.gpu_id = 0;
	opts.dname = "lm";
	opts.root_dataset = "/dataset/public/RCVLab/Bluewrist/16yw11";
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
    if (lib_check) {
        std::cout << std::string(100, '=') << std::endl;
        std::cout << "Testing Torch Lib:" << std::endl;
        torch::Tensor tensor1 = torch::rand({ 500, 600 });
        torch::Tensor tensor2 = torch::rand({ 500, 600 });

        // Convert tensor to cv::Mat
        int rows = tensor1.size(0);
        int cols = tensor1.size(1);
        cv::Mat mat(rows, cols, CV_32FC1, tensor1.data_ptr<float>());
        // Display the matrix as an image
        cv::imshow("mat", mat);
        cv::waitKey(0);

        torch::Tensor stacked = torch::stack({ tensor1, tensor2 }, 0);
        cout << stacked.sizes() << endl;
        std::cout << std::string(50, '=') << std::endl;
    }
    if (dev_check) {
        cout << string(100, '=') << endl;
        cout << "Testing if torch can access cuda device" << endl;
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        cout << "Current device: " << device << endl;
        cout << "CUDA device count: " << torch::cuda::device_count() << endl;
    }
    if (rcv_check) {
        RCVpose rcv;
        rcv.summary();

        cout << string(100, '=') << endl;
        Options opts;
        opts.gpu_id = 0;
        opts.dname = "lm";
        opts.root_dataset = "LINEMODE";
        opts.resume_train = false;
        opts.optim = "adam";
        opts.batch_size = 1;
        opts.class_name = "ape";
        opts.initial_lr = 0.0001;
        opts.kpt_num = 8;
        opts.model_dir = "models";
        opts.demo_mode = false;
        opts.test_occ = false;
        RCVpose rcv2(opts);
    }
    if (ldr_check) {
        cout << string(100, '=') << endl;
        cout << "Testing Loaders" << endl;

        Options opts;
        opts.gpu_id = 0;
        opts.dname = "lm";
        opts.root_dataset = "C:/Users/User/.cw/work/datasets/test";
        opts.model_dir = "models";
        opts.batch_size = 4;
        RCVpose loader_check(opts);

        int test = loader_check.test_loaders();

        if (test != -1)
            cout << "Success in testing loader" << endl;
        else
            cout << "Failure in testing loader" << endl;

    }
    if (trn_check) {
        cout << string(100, '=') << endl;
        cout << string(40, ' ') << "Testing Training" << endl;

        RCVpose rcv(testing_options());

        rcv.train();
    }
    if (stw_model) {
        cout << string(100, '=') << endl;
        cout << string(40, ' ') << "Testing Saving and Loading Model" << endl;
        RCVpose rcv(testing_options());
        rcv.saveOutput(1 , "test_store");
    }

    if (trn_ckpt) {
        // Train checkpoint
        cout << string(100, '=') << endl;
        cout << string(40, ' ') << "Training Checkpoint" << endl;
        //Enter number of keypoints, if not int make them re-enter
        cout << "Enter number of keypoints: ";
        int kpts = cin.get();
        while (!(cin >> kpts)) {
            // user didn't input a number
			cout << "Enter number of keypoints: ";
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
        }
        //Enter batch size, if not int make them re-enter
        cout << "Enter batch size: ";
        int batch_size = cin.get();
        while (!(cin >> batch_size)) {
			// user didn't input a number
            cout << "Enter batch size: ";
			cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
		}
        //Enter y or n for resume training bool
        cout << "Resume training? (y/n): ";
		char resume = cin.get();
        while (resume != 'y' && resume != 'n') {
            // user didn't input a number
			cout << "Resume training? (y/n): ";
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
		}

        bool rsm = false;
        if (resume == 'y') 
            rsm = true;

        Options opts = training_options(kpts, rsm, batch_size);
        RCVpose rcv(opts);

        rcv.train();
    }
    if (trn_gpu) {
        Options opts = gpu_train_opts(1, false, 20);
        RCVpose rcv(opts);
        rcv.train();
    }
    return 0;
}
