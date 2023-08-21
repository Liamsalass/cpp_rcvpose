#include "rcvpose.h"

using namespace std;

RCVpose::RCVpose(Options& options)
{
   
    opts = options;
    
    cout << string(75, '=') << endl;
    cout << string(26, ' ') << "Initializing RCVpose" << endl << endl;
    

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
    if (torch::cuda::is_available()) {
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

            CheckpointLoader model_loader(out, true, true, false);

            model = model_loader.getModel();

            model->to(device);

            model->eval();

            torch::NoGradGuard no_grad;

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
    cout << string(75, '=') << endl;
    cout << string(28, ' ') << "Summary" << endl << endl;
    cout << "Device: " << device << endl;
    cout << "dname: " << opts.dname << endl;
    cout << "root_dataset: " << opts.root_dataset << endl;
    cout << "resume_train: " << opts.resume_train << endl;
    cout << "optim: " << opts.optim << endl;
    cout << "batch_size: " << opts.batch_size << endl;
    cout << "class_name: " << opts.class_name << endl;
    cout << "Reduce Learning Rate on Plateau: ";
    if (opts.reduce_on_plateau) {
        cout << "True" << endl;
        cout << "\tPatience: " << opts.patience << endl;
    }
    else {
		cout << "False" << endl;
	}
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
    estimate_6d_pose_lm(opts, model);
}

void RCVpose::estimate_pose(cv::Mat img_with_depth_channel)
{
    //Img must have the shape (height, width, 4)
	cv::Mat img_(img_with_depth_channel.rows, img_with_depth_channel.cols, CV_8UC3);
	cv::cvtColor(img_with_depth_channel, img_, cv::COLOR_BGRA2BGR);
	cv::Mat depth_(img_with_depth_channel.rows, img_with_depth_channel.cols, CV_8UC1);
	cv::extractChannel(img_with_depth_channel, depth_, 3);
	estimate_pose(img_, depth_);
}

void RCVpose::estimate_pose(double* img, double* depth, const int height, const int width)
{
    //Img must have the shape (height, width, 3)
    //Depth must have the shape (height, width, 1)
    cv::Mat img_(height, width, CV_8UC3, img);
    cv::Mat depth_(height, width, CV_8UC1, depth);
    estimate_pose(img_, depth_);
}

void RCVpose::estimate_pose(cv::Mat img_, cv::Mat depth_)
{
    if (img.data)
        img.release();
    if (depth.data)
        depth.release();

    img = img_.clone();
    depth = depth_.clone();

    inference();

}

void RCVpose::estimate_pose(const std::string& img_path, const std::string& depth_path)
{
    if (img.data)
		img.release();
    if (depth.data)
        depth.release();

    try {
        img = cv::imread(img_path);
        depth = read_depth_to_cv(depth_path, false);
    }
    catch (const std::exception& e) {
		cout << "Error: " << e.what() << endl;
		exit(1);
	}

    inference();


}


void RCVpose::inference()
{
    string rootPath = opts.root_dataset + "/LINEMOD/" + opts.class_name + "/";

    string keypoints_path = rootPath + "Outside9.npy";

    vector<vector<double>> keypoints = read_double_npy(keypoints_path, false);

    string pcv_load_path = rootPath + "pc_" + opts.class_name + ".npy";

    vector<Vertex> orig_point_cloud = read_point_cloud(pcv_load_path);

    estimate_6d_pose(opts, model, img, depth, keypoints, orig_point_cloud);
}


void RCVpose::save_all_test_tensors() {
    cout << string(50, '-') << endl;
    
    // Forward pass all the test images through the model and save the ouput tensors to path
    const vector<double> mean = { 0.485, 0.456, 0.406 };
    const vector<double> standard = { 0.229, 0.224, 0.225 };


    ifstream test_file(opts.root_dataset + "/LINEMOD/" + opts.class_name + "/Split/val.txt");
    vector<string> test_list;
    string line;

    model->eval();

    // Read lines from test file
    if (test_file.is_open()) {
        while (getline(test_file, line)) {
            line.erase(line.find_last_not_of("\n\r\t") + 1);
            test_list.push_back(line);
        }
        test_file.close();
    }
    else {
        cerr << "Unable to open file containing test data" << endl;
        exit(EXIT_FAILURE);
    }

    const int total_num_img = test_list.size();
    const string path = opts.model_dir + "/test_tensors/";

    cout << "Saving all " << total_num_img << " test tensors to " << path << endl;

    int count = 0;
    auto start_save = chrono::high_resolution_clock::now();

    

    for (auto test_img : test_list) {
        string image_path = opts.root_dataset + "/LINEMOD/" + opts.class_name + "/JPEGImages/" + test_img + ".jpg";
        cv::Mat img = cv::imread(image_path);
        torch::Device device(device_type);

        img.convertTo(img, CV_32FC3);
        img /= 255.0;

        for (int i = 0; i < 3; i++) {
            cv::Mat channel(img.size(), CV_32FC1);
            cv::extractChannel(img, channel, i);
            channel = (channel - mean[i]) / standard[i];
            cv::insertChannel(channel, img, i);
        }
        if (img.rows % 2 != 0)
            img = img.rowRange(0, img.rows - 1);
        if (img.cols % 2 != 0)
            img = img.colRange(0, img.cols - 1);
        cv::Mat imgTransposed = img.t();

        torch::Tensor imgTensor = torch::from_blob(imgTransposed.data, { imgTransposed.rows, imgTransposed.cols, imgTransposed.channels() }, torch::kFloat32).clone();
        imgTensor = imgTensor.permute({ 2, 0, 1 });


        auto img_batch = torch::stack(imgTensor, 0).to(device);

        //torch::NoGradGuard no_grad;


        auto output = model->forward(img_batch);

        output = output.to(torch::kCPU);

        auto radial_output1 = output[0][0];
        auto radial_output2 = output[0][1];
        auto radial_output3 = output[0][2];
        auto semantic_output = output[0][3];

        saveTensorToFile(radial_output1, path + "/radial1/" + test_img + ".txt");
        saveTensorToFile(radial_output2, path + "/radial2/" + test_img + ".txt");
        saveTensorToFile(radial_output3, path + "/radial3/" + test_img + ".txt");
        saveTensorToFile(semantic_output, path + "/semantic/" + test_img + ".txt");

        printProgressBar(count, total_num_img, 50);
        count++;
    }
    cout << endl;

    auto end_save = chrono::high_resolution_clock::now();
    cout << "Saved all test tensors to " << path << " in " << chrono::duration_cast<chrono::seconds>(end_save - start_save).count() << " seconds" << endl << endl;
}


