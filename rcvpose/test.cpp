// test.cpp : Defines the entry point for the application.
//

#include "rcvpose.h"
#include <iostream>
#include "torch/torch.h"
#include "opencv2/opencv.hpp"

using namespace std;

#define lib_check false
#define dev_check false
#define rcv_check false
#define ldr_check false
#define trn_check true
#define rsm_check true

Options training_options() {
	Options opts;
	opts.gpu_id = 0;
	opts.dname = "lm";
	opts.root_dataset = "/ingenuity_NAS/dataset/public/RCVLab/Bluewrist/16yw113";
    //or ".../dataset/public/RCVLab/Bluewrist/16yw11"
	opts.model_dir = "t1";
	opts.resume_train = rsm_train;
	opts.optim = "adam";
    opts.batch_size = 2;
	opts.class_name = "ape";
	opts.initial_lr = 0.0001;
	opts.kpt_num = 3;
	opts.demo_mode = false;
	opts.test_occ = false;
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
    if (trn_check || rsm_check) {
        cout << string(100, '=') << endl;
        cout << string(40, ' ') << "Testing Training" << endl;
        RCVpose rcv(training_options());
        rcv.train();
    }

    return 0;
}
