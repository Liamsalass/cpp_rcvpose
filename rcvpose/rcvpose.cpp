#include "rcvpose.h"

using namespace std;

RCVpose::RCVpose(Options& options)
{
   
    opts = options;
    
    cout << string(100, '=') << endl;
    cout << string(34, ' ') << "Initializing RCVpose" << endl << endl;
    

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
        auto train_dataset = RData(opts.root_dataset, opts.dname, "train", opts.class_name);
        auto val_dataset = RData(opts.root_dataset, opts.dname, "test", opts.class_name);

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
            if (opts.verbose) {
                std::cout << "GPU is properly initialized." << std::endl;
            }
        }
        else {
            // Failed to initialize GPU
            std::cout << "Failed to initialize GPU. Error: " << cudaGetErrorString(cuda_status) << std::endl;
            cout << "Switching device back to CPU" << endl;
            device_type = torch::kCPU;
        }
    }


    cout << "Setting up model" << endl;
    const string& out = opts.model_dir;

    if (!opts.resume_train) {
        // Instantiate the model
        try {
            filesystem::path outPath(out);
            if (!filesystem::exists(outPath)) {
                if (filesystem::create_directories(outPath)) {
                    std::cout << "Created directory: " << outPath << std::endl;
                }
                else {
                    std::cout << "Failed to create directory: " << outPath << std::endl;
                    can_run = false;
                }
            }

            model = DenseFCNResNet152(3, 4);
            model->to(device);

            if ((device == torch::kCUDA) && (opts.verbose)) {
                size_t free_memory, total_memory;
                cudaMemGetInfo(&free_memory, &total_memory);

                size_t used_memory = total_memory - free_memory;

                cout << "Total GPU memory: " << total_memory / (1024 * 1024 * 1024) << " GB, "
                    << (total_memory % (1024 * 1024 * 1024)) / (1024 * 1024) << " MB, "
                    << total_memory % (1024 * 1024) << " bytes" << endl;

                cout << "Memory Profile of Empty Model" << endl;
                cout << "\tUsed GPU memory: " << used_memory / (1024 * 1024 * 1024) << " GB, "
                    << (used_memory % (1024 * 1024 * 1024)) / (1024 * 1024) << " MB, "
                    << used_memory % (1024 * 1024) << " bytes" << endl;

                cout << "\tFree GPU memory: " << free_memory / (1024 * 1024 * 1024) << " GB, "
                    << (free_memory % (1024 * 1024 * 1024)) / (1024 * 1024) << " MB, "
                    << free_memory % (1024 * 1024) << " bytes" << endl;
            }


        }
        catch (const torch::Error& e) {
            cout << "Error: " << e.msg() << endl;
            can_run = false;
        }
    }
    else {
        try {
            //model = DenseFCNResNet152(3, 4);
            //DenseFCNResNet152 tmp_model = loader.getModel();

            //auto model_params = model->named_parameters();
            //auto tmp_model_params = tmp_model->named_parameters();
            //
            //for (auto& named_param : tmp_model_params) {
            //    if (model_params.contains(named_param.key())) {
            //        auto& model_param = model_params[named_param.key()];
            //        model_param.data().copy_(named_param.value().data());
            //    }
            //}

            CheckpointLoader loader(out, true);

            model = loader.getModel();

            model->to(device);


            torch::Tensor dummy_input = torch::randn({ 2, 3, 640, 480 }, device);

            if ((device == torch::kCUDA) && (opts.verbose)) {
                size_t free_memory, total_memory;
                cudaMemGetInfo(&free_memory, &total_memory);

                size_t used_memory = total_memory - free_memory;

                cout << "Total GPU memory: " << total_memory / (1024 * 1024 * 1024) << " GB, "
                    << (total_memory % (1024 * 1024 * 1024)) / (1024 * 1024) << " MB, "
                    << total_memory % (1024 * 1024) << " bytes" << endl;

                cout << "Memory Profile with dummy tensor and model on GPU" << endl;
                cout << "\tUsed GPU memory: " << used_memory / (1024 * 1024 * 1024) << " GB, "
                    << (used_memory % (1024 * 1024 * 1024)) / (1024 * 1024) << " MB, "
                    << used_memory % (1024 * 1024) << " bytes" << endl;

                cout << "\tFree GPU memory: " << free_memory / (1024 * 1024 * 1024) << " GB, "
                    << (free_memory % (1024 * 1024 * 1024)) / (1024 * 1024) << " MB, "
                    << free_memory % (1024 * 1024) << " bytes" << endl;
            }

            auto out = model->forward(dummy_input);

        }
        catch (const torch::Error& e) {
            cout << "Cannot Resume model from checkpoint" << endl;
            cout << "Error: " << e.msg() << endl;
            can_run = false;
        }
    }

    if (!can_run) exit(1);

    //Print params
    cout << endl;
    cout << string(100, '=') << endl;
    cout << string(45, ' ') << "Summary" << endl << endl;
    cout << "Device: " << device << endl;
    cout << "dname: " << opts.dname << endl;
    cout << "root_dataset: " << opts.root_dataset << endl;
    cout << "resume_train: " << opts.resume_train << endl;
    cout << "optim: " << opts.optim << endl;
    cout << "batch_size: " << opts.batch_size << endl;
    cout << "class_name: " << opts.class_name << endl;
    cout << "initial_lr: " << opts.initial_lr << endl;
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
    Trainer trainer(opts, model);
    trainer.train(opts, model);
}


void RCVpose::validate() {
    // Implementation for evaluating the model
}


void RCVpose::demo() {
    // Implementation for running the model in demo mode
}


