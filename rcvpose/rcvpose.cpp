﻿#include "rcvpose.h"

using namespace std;

RCVpose::RCVpose(Options options)
{
    opts = options;
    init();
}

RCVpose::RCVpose(
    int gpu_id,
    std::string dname,
    std::string root_dataset,
    bool resume_train,
    std::string optim,
    int batch_size,
    std::string class_name,
    double initial_lr,
    int kpt_num,
    std::string model_dir,
    bool demo_mode,
    bool test_occ
) {
    opts.gpu_id = gpu_id;
    opts.dname = dname;
    opts.root_dataset = root_dataset;
    opts.resume_train = resume_train;
    opts.optim = optim;
    opts.batch_size = batch_size;
    opts.class_name = class_name;
    opts.initial_lr = initial_lr;
    opts.kpt_num = kpt_num;
    opts.model_dir = model_dir;
    opts.demo_mode = demo_mode;
    opts.test_occ = test_occ;
    init();
}


RCVpose::RCVpose() {
    cout << "Warning: Empty constructor called" << endl;
    init();
}

RCVpose::~RCVpose() {

}

void RCVpose::summary() {
    torch::Device device(device_type);
    // Print a line of '=' with "Summary" in the middle
    cout << endl;
    cout << string(100, '=') << endl;
    cout << string(45, ' ') << "Summary" << endl << endl;

    // Print the options
    cout << "Name: " << typeid(*this).name() << endl;
    cout << "Device: " << device << endl;  
    cout << "dname: " << opts.dname << endl;
    cout << "root_dataset: " << opts.root_dataset << endl;
    cout << "resume_train: " << opts.resume_train << endl;
    cout << "optim: " << opts.optim << endl;
    cout << "batch_size: " << opts.batch_size << endl;
    cout << "class_name: " << opts.class_name << endl;
    cout << "initial_lr: " << opts.initial_lr << endl;
    cout << "kpt_num: " << opts.kpt_num << endl;
    cout << "model_dir: " << opts.model_dir << endl;
    cout << "demo_mode: " << opts.demo_mode << endl;
    cout << "test_occ: " << opts.test_occ << endl;

    cout << "Configurations: " << endl;
    for (const auto& entry : opts.cfg) {
        const std::string& key = entry.first;
        const std::vector<float>& values = entry.second;

        cout << "\t" << key << ": [";
        for (const float& value : values) {
            cout << value << " ";
        }
        cout << "]" << endl;
    }
    cout << endl;
}

void RCVpose::train()
{
    try {
        Trainer trainer(opts);

        //trainer.test_compute_r_loss();

        trainer.train();
    }
    catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    
}


void RCVpose::validate() {
    // Implementation for evaluating the model
}

void RCVpose::compare_models(string model1, string model2) {

}

void RCVpose::save_tensor(const std::string& path, const int& idx) {
    try {
        Trainer trainer(opts);

        trainer.output_pred(idx,path);
    }
    catch (const std::exception& e) {
		std::cout << e.what() << std::endl;
	}
}

void RCVpose::save_tensor(const std::string& path, const int& start, const int& end) {
    Trainer trainer(opts);

    for (int i = start; i < end; i++) {
		trainer.output_pred(i, path);
	}  
}

void RCVpose::test_img(std::string img_path, std::string output_path)
{

}

void RCVpose::demo() {
    // Implementation for running the model in demo mode
}

void RCVpose::loadModel(std::string path) {
    // Implementation for loading a pretrained model
}

void RCVpose::test_loaders()
{
    auto val_dataset = RData(opts.root_dataset, opts.dname, "val", opts.class_name, opts.kpt_num);

    auto data_tensor = val_dataset.get(0).data();
    auto sem_target = val_dataset.get(0).sem_target();
    auto rad_target = val_dataset.get(0).target();

    auto test_tensor = Trainer(opts);

    test_tensor.tensorToFile(data_tensor, opts.model_dir + "/img0/input_data_tensor.txt");
    test_tensor.tensorToFile(sem_target, opts.model_dir + "/img0/input_sem_target.txt");
    test_tensor.tensorToFile(rad_target, opts.model_dir + "/img0/input_rad_target.txt");
}


void RCVpose::init() {
    // Initialization of the model
    // check if opts has all passed parameters, if not return error message with what needs to be initialized
    bool can_run = can_init();

    //Apply configs
    try {
        opts.cfg = get_config().at(1);
    }
    catch (const std::out_of_range& oor) {
        cout << "Error: Cannot find the configuration for the given dataset (Out of range)" << endl;
        can_run = false;
    }
    catch (const std::exception& e) {
        cout << "Error: Cannot find the configuration for the given dataset or invalid config" << endl;
        can_run = false;
    }

    //Set random seed
    resume = "";
    torch::manual_seed(0);

    //Set device type
    if (opts.gpu_id >= 0 && torch::cuda::is_available()) {
        device_type = torch::kCUDA;
    }
    else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    if (device_type == torch::kCUDA) {
        cudaError_t cuda_status = cudaSuccess;

        // Initialize CUDA
        cuda_status = cudaFree(0);

        if (cuda_status == cudaSuccess) {
            // GPU is properly initialized
            std::cout << "GPU is properly initialized." << std::endl;
        }
        else {
            // Failed to initialize GPU
            std::cout << "Failed to initialize GPU. Error: " << cudaGetErrorString(cuda_status) << std::endl;
            cout << "Switching device back to CPU" << endl;
            device_type = torch::kCPU;
        }
    }
    
    //If it cannot run, print error msg and exit
    if (!can_run) {
        cout << "Error: Cannot initialize the model with the given parameters" << endl;
        return;
    }
    else {
        //Print summary
        summary();
    }
}


bool RCVpose::can_init() {
    bool can_run = true;
    if (opts.dname.empty()) {
        cout << "Warning: dname not initialized" << endl;
        can_run = false;
    }
    if (opts.root_dataset.empty()) {
        cout << "Warning: root_dataset not initialized" << endl;
        can_run = false;
    }
    if (opts.class_name.empty()) {
        cout << "Warning: class_name not initialized" << endl;
        can_run = false;
    }
    if (opts.model_dir.empty()) {
        cout << "Warning: model directory not initialized" << endl;
        can_run = false;
    }
    // Instantiate the dataset
    try {
        auto train_dataset = RData(opts.root_dataset, opts.dname, "train", opts.class_name, opts.kpt_num);
        auto val_dataset = RData(opts.root_dataset, opts.dname, "test", opts.class_name, opts.kpt_num);

        // Instantiate the dataloaders 
        auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset),
            torch::data::DataLoaderOptions().batch_size(opts.batch_size).workers(1)
        );

        auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(val_dataset),
            torch::data::DataLoaderOptions().batch_size(opts.batch_size).workers(1)
        );
    }
    catch (const std::exception& e) {
        cout << "Error: Cannot instantiate the dataset\n" << e.what() << endl;
        can_run = false;
    }

    // Ensure opts.batch_size is an even number or equal to one, and is greater than 0
    if (opts.batch_size % 2 != 0 && opts.batch_size != 1) {
		cout << "Warning: batch_size is not an even number or equal to one" << endl;
		can_run = false;
	}

    return can_run;
}

