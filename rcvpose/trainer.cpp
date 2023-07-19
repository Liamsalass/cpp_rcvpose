#include "trainer.h"

using namespace std;

cv::Mat torch_tensor_to_cv_mat(torch::Tensor tensor) {
    int rows = tensor.size(0);
    int cols = tensor.size(1);
    cv::Mat mat(rows, cols, CV_32FC1, tensor.data_ptr<float>());
    return mat.clone();
};

Trainer::Trainer(Options& options) : opts(options)
{
    cout << string(100, '=') << endl;
    cout << string (34, ' ') << "Initializing Trainer" << endl << endl;

    bool use_cuda = torch::cuda::is_available();
    device_type = use_cuda ? torch::kCUDA : torch::kCPU;
    cout << "Using " << (use_cuda ? "CUDA" : "CPU") << endl;
    cout << "Setting up model" << endl;

    if (!opts.resume_train) {
        // Instantiate the model
        try {
            //if(torch::cuda::device_count() > 1){
            //    cout << "Using " << torch::cuda::device_count() << " GPUs" << endl;
            //    model = DenseFCNResNet152(3, 2);
            //    model->to(device);
            //    model = torch::nn::parallel::data_parallel(model,torch::cude::device_count());
            //}else {
            model = DenseFCNResNet152(3, 2);
            // if opts.gpu_id is not -1, then set the device to that gpu
            torch::Device device(device_type);
            model->to(device);
            //model->to(torch::kCPU);
            cout << "Model initialized " << model->name() << endl;

            // Data parallelization not working
            //if (torch::cuda::device_count() > 1) {
            //    cout << "Using " << torch::cuda::device_count() << " GPUs" << endl;
            //    model = torch::nn::DataParallel(model, { 0, 1 });
            //
            //}
        }
        catch (const torch::Error& e) {
            cout << "Error: " << e.msg() << endl;
            return;
        }
        cout << "Setting up optimizer" << endl;
        // Instantiate the optimizer
        try {
            if (opts.optim == "adam") {
                optim  = new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(opts.initial_lr));

            }
            else if (opts.optim == "sgd") {
                optim = new torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(opts.initial_lr));
            }
            else {
                cout << "Error: Invalid optimizer" << endl;
                return;
            }
        }
        catch (const torch::Error& e) {
            cout << "Error: " << e.msg() << endl;
            return;
        }
        int count = 0;
        for (auto& params : optim->param_groups()) {
            cout << "Param Group " << count << " with LR value: " << params.options().get_lr() << endl;
            count++;
        }

        current_lr.clear();
        current_lr.push_back(opts.initial_lr);

        epoch = 0;
        starting_epoch = 0;
        
    } 
    else {
        try {
            // Load model from checkpoint
            cout << "Loading model from checkpoint" << endl;
            CheckpointLoader loader(opts.model_dir, true);
            epoch = loader.getEpoch();
            starting_epoch = epoch;
            cout << "Epoch: " << epoch << endl;
            //if(torch::cuda::available()){
            //    cout << "Using " << torch::cuda::device_count() << " GPUs" << endl;
            //    model = loader.loadModel();
            //    model->to(device);
            //    model = torch::nn::DataParallel(model,torch::cuda::device_count());
            //}
            model = loader.getModel();
            torch::Device device(device_type);
            model->to(device);
            cout << "Model loaded" << endl;
            optim = loader.getOptimizer();
            optim->parameters() = model->parameters();
            current_lr = loader.getLrList();
            cout << "Optimizer loaded" << endl;
            int count = 0;
            for (auto& params : optim->param_groups()) {
                params.options().set_lr(current_lr[0]);
                cout << "Param Group " << count << " with LR value: " << params.options().get_lr() << endl;
                count++;
            }

            current_lr.clear();
            current_lr.push_back(opts.initial_lr);

            best_acc_mean = loader.getBestAccuracy();
            cout << "Best Accuracy: " << best_acc_mean << endl;

            float prev_loss = loader.getLoss();
            cout << "Previous Loss: " << prev_loss << endl;

        } 
        catch (const torch::Error& e) {
            cout << "Cannot Resume Training" << endl;
			cout << "Error: " << e.msg() << endl;
            return;
		}
    }
    cout << "Setting up loss function" << endl;

    // Instantiate the loss function
    try {
        torch::Device device(device_type);
        loss_radial = torch::nn::L1Loss(torch::nn::L1LossOptions().reduction(torch::kSum));
        //loss_radial->to(torch::kCPU);

        loss_radial->to(device);
    }
    catch (const torch::Error& e) {
        cout << "Error: " << e.msg() << endl;
        return;
    }
    try {
        torch::Device device(device_type);        
        loss_sem = torch::nn::L1Loss();
        //loss_sem->to(torch::kCPU);
        loss_sem->to(device);
    }
    catch (const torch::Error& e) {
        cout << "Error: " << e.msg() << endl;
        return;
    }

    iteration = 0;
    iteration_val = 0;
    max_iteration = opts.cfg.at("max_iteration")[0];
    best_acc_mean = std::numeric_limits<double>::infinity();

    out = opts.model_dir;

    std::filesystem::path outPath(out);
    if (!std::filesystem::is_directory(outPath)) {
        if (std::filesystem::create_directories(outPath)) {
            std::cout << "Output directory created" << std::endl;
        }
        else {
            std::cout << "Failed to create output directory" << std::endl;
        }
    }
    else {
        std::cout << "Warning: Output directory already exists" << std::endl;
    }
    cout << "Model Path: " << std::filesystem::current_path() << out << endl;

    cout << "Trainer Initialized" << endl;

    
}

void Trainer::train()
{
    cout << string(100, '=') << endl; 
    cout << string(24, ' ') << "Begining Training Initialization" << endl << endl;
    

    //Select device base on opts.gpu_id

    torch::Device device(device_type);
   
    cout << "Setting up dataset loader" << endl;
    // Instantiate the training dataset
    // Can use .map(torch::data::transforms::Stack<>()) to stack batches into a single tensor
    auto train_dataset = RData(opts.root_dataset, opts.dname, "train", opts.class_name, opts.kpt_num);
    torch::optional<size_t> train_size = train_dataset.size();

    auto val_dataset = RData(opts.root_dataset, opts.dname, "val", opts.class_name, opts.kpt_num);
    torch::optional<size_t> val_size = val_dataset.size();

    // Check if dataset is empty
    if (train_size.value() == 0 || !train_size.has_value()) {
        cout << "Error: Could not get size of training dataset" << endl;
        return;
    }
    if (val_size.value() == 0 || !val_size.has_value()) {
        cout << "Error: Could not get size of validation dataset" << endl;
        return;
    }

    cout << "Train Data Set Size : " << train_size.value() << endl;
    cout << "Val Data Set Size : " << val_size.value() << endl;

    max_epoch = static_cast<int>(std::ceil(1.0 * max_iteration / train_size.value()));

    // Instantiate the dataloaders 

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(opts.batch_size).workers(1)
    );

    auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(val_dataset),
        torch::data::DataLoaderOptions().batch_size(opts.batch_size).workers(1)
    );

    cout << "Max Epochs : " << max_epoch << endl;

    auto total_train_start = std::chrono::steady_clock::now();
    while (epoch < max_epoch) {
        auto epoch_start_time = std::chrono::steady_clock::now();
        cout << string(100, '-') << endl;
        cout << string(43, ' ') << "Epoch " << epoch << endl;



        // ========================================================================================== \\
        // ====================================== Training ========================================== \\

        cout << "Training Epoch" << endl;
        int count = 0;
        model->train();
        
        auto train_start = std::chrono::steady_clock::now();
        for (const auto& batch : *train_loader) {
          

            count = batch.size() + count;
            iteration = batch.size() + iteration;

            printProgressBar(count, train_size.value(), 80);

            std::vector<torch::Tensor> batch_data;
            std::vector<torch::Tensor> batch_target;
            std::vector<torch::Tensor> batch_sem_target;

            for (const auto& example : batch) {
                batch_data.push_back(example.data());
                batch_target.push_back(example.target());
                batch_sem_target.push_back(example.sem_target());
            }

            optim->zero_grad();
         
            auto data = torch::stack(batch_data, 0).to(device); 
            auto target = torch::stack(batch_target, 0).to(device);
            auto sem_target = torch::stack(batch_sem_target, 0).to(device);

            std::tuple<torch::Tensor, torch::Tensor> scores = model->forward(data);

            auto& score = std::get<0>(scores);
            auto& score_rad = std::get<1>(scores);
            score_rad = score_rad.permute({ 1, 0, 2, 3 }) * sem_target.permute({ 1, 0, 2, 3 });
            score_rad = score_rad.permute({ 1, 0, 2, 3 });

            torch::Tensor loss_s = loss_sem(score, sem_target);
            torch::Tensor loss_r = compute_r_loss(score_rad, target);

            torch::Tensor loss = loss_r + loss_s;

            //cout << "Radial Loss: " << loss_r.item<float>() << " Semantic Loss: " << loss_s.item<float>() << " Total Loss: " << loss.item<float>() << "\r";

            loss.backward();

            optim->step();

            auto np_loss = loss.detach().cpu().numpy_T();

            if (np_loss.numel() == 0)
                std::runtime_error("Loss is empty");

        }

        auto train_end = std::chrono::steady_clock::now();
        auto train_duration = std::chrono::duration_cast<std::chrono::seconds>(train_end - train_start);
        cout << string(100, ' ');
        cout << "\rTraining Time: " << train_duration.count() << " s" << endl;

        // ========================================================================================== \\
        //                                  Validation Epoch 									       \\

        cout << "Validation Epoch" << endl;
        model->eval();
        float val_loss = 0;
        count = 0;
        torch::NoGradGuard no_grad;
        auto val_start = std::chrono::steady_clock::now();

        int val_count = 0;

        for (const auto& batch : *val_loader) {
            count = batch.size() + count;
            iteration_val = batch.size() + iteration_val;
            printProgressBar(count, val_size.value(), 80);


            std::vector<torch::Tensor> batch_data;
            std::vector<torch::Tensor> batch_target;
            std::vector<torch::Tensor> batch_sem_target;

            for (const auto& example : batch) {
                batch_data.push_back(example.data());
                batch_target.push_back(example.target());
                batch_sem_target.push_back(example.sem_target());
            }


            auto img = torch::stack(batch_data, 0);
            auto target = torch::stack(batch_target, 0);
            auto sem_target = torch::stack(batch_sem_target, 0);


            cout << "Image shape: " << img.sizes() << endl;
            img = img.to(device);
            target = target.to(device);
            sem_target = sem_target.to(device);

            auto output = model->forward(img);

            auto score = std::get<0>(output);
            auto score_rad = std::get<1>(output);

            if (opts.debug == true && (val_count % 50 == 0)) {
                auto score_cpu = score.to(torch::kCPU);
                auto rad_cpu = score_rad.to(torch::kCPU);

                auto score_unstack = torch::unbind(score_cpu, 0);
                auto score_rad_unstack = torch::unbind(rad_cpu, 0);

                auto out_score = score_unstack[0];
                auto out_score_rad = score_rad_unstack[0];

                out_score = out_score.permute({ 1,2,0 });
                out_score_rad = out_score_rad.permute({ 1,2,0 });

                cv::Mat sem = torch_tensor_to_cv_mat(out_score);
                cv::Mat rad = torch_tensor_to_cv_mat(out_score_rad);
                cv::imshow("Semantic", sem);
                cv::imshow("Radial", rad);
                cv::waitKey(0);
            }

            val_count++;

            auto loss_s = loss_sem(score, sem_target);
            auto loss_r = compute_r_loss(score_rad, target);

            auto loss = loss_r + loss_s;

            //cout << "Loss_r: " << loss_r.item<float>() << " Loss_s: " << loss_s.item<float>() << "\r";

            if (loss.numel() == 0)
                std::runtime_error("Loss is empty");

            val_loss += loss.item<float>();

        }
        auto val_end = std::chrono::steady_clock::now();
        auto val_duration = std::chrono::duration_cast<std::chrono::seconds>(val_end - val_start);
        cout << string(100, ' ');
        cout << "\rValidation Time: " << val_duration.count()<< " s" << endl;
        cout.flush();

        val_loss /= val_size.value();
        float mean_acc = val_loss;
        cout << "Mean Loss: " << mean_acc << endl;
        bool is_best = mean_acc < best_acc_mean;
        
        if (is_best) {
            best_acc_mean = mean_acc;
            epochs_without_improvement = 0;
        } else {
            epochs_without_improvement++;
        }
        cout << "Iterations: " << iteration << endl;
        cout << "Epochs without improvement: " << epochs_without_improvement << endl;



        //================================================================\\
        //                  Save Model and Optimizer                      \\
        
        if (!opts.debug) {
            try {
                std::string save_location;
                if (is_best) {
                    cout << "Saving New Best Model" << endl;
                    save_location = out + "/model_best";
                }
                else {
                    cout << "Saving Current Model" << endl;
                    save_location = out + "/current";
                }

                if (!std::filesystem::is_directory(save_location))
                    std::filesystem::create_directory(save_location);

                torch::serialize::OutputArchive output_model_info;
                output_model_info.write("epoch", epoch);
                output_model_info.write("iteration", iteration);
                output_model_info.write("arch", model->name());
                output_model_info.write("best_acc_mean", best_acc_mean);
                output_model_info.write("loss", val_loss);
                output_model_info.write("optimizer", opts.optim);
                output_model_info.write("lr", current_lr);

                output_model_info.save_to(save_location + "/info.pt");

                torch::serialize::OutputArchive output_model_archive;
                model->to(torch::kCPU);
                model->save(output_model_archive);
                model->to(device);
                output_model_archive.save_to(save_location + "/model.pt");


                torch::serialize::OutputArchive output_optim_archive;
                optim->save(output_optim_archive);
                output_optim_archive.save_to(save_location + "/optim.pt");
            }
            catch (std::exception& e) {
                std::cout << "Error saving model: " << e.what() << std::endl;
            }
        }
        else {
            cout << "Skipping Model saving: Debug == true" << endl;
        }
        

        // Reduce learning rate every 70 epoch
        if (opts.reduce_on_plateau = false){
            if (epoch % 70 == 0 && epoch != 0) {
                cout << "Learning rate reduction" << endl;
                current_lr.clear();
                for (auto& param_group : optim->param_groups()) {
                    if (param_group.has_options()) {
                        double lr = param_group.options().get_lr();
                        cout << "Current LR: " << lr << endl;
                        double new_lr = lr * 0.1;
                        if (opts.optim == "adam") 
                            static_cast<torch::optim::AdamOptions &>(param_group.options()).lr(new_lr);
                        if (opts.optim == "sgd")
                            static_cast<torch::optim::SGDOptions &>(param_group.options()).lr(new_lr);
                        cout << "New LR: " << new_lr << endl;
                        current_lr.push_back(new_lr);
                    }
                    else {
                        cout << "Error: param_group has no options" << endl;
                    }
                }
            }
        }
        if (opts.reduce_on_plateau && epochs_without_improvement >= opts.patience) {
            cout << "Reducing learning rate" << endl;
            current_lr.clear();
            for (auto& param_group : optim->param_groups()) {
                if (param_group.has_options()) {
                    double lr = param_group.options().get_lr();
                    cout << "Current LR: " << lr << endl;
                    double new_lr = lr * 0.1;
                    if (opts.optim == "adam")
                        static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(new_lr);
                    else if (opts.optim == "sgd")
                        static_cast<torch::optim::SGDOptions&>(param_group.options()).lr(new_lr);
                    else
                        cout << "Error: Invalid optimizer" << endl;
                    cout << "New LR: " << new_lr << endl;
                    current_lr.push_back(new_lr);
                } else {
                    cout << "Error: param_group has no options" << endl;
                }
            }
            epochs_without_improvement = 0;
        }
        if (iteration >= max_iteration) {
            break;
        }

        auto epoch_train_end = std::chrono::steady_clock::now(); 
        auto epoch_total_time = std::chrono::duration_cast<std::chrono::seconds>(epoch_train_end - epoch_start_time);
        cout << "Epoch Training Time: " << epoch_total_time.count() << " s" << endl;

        auto total_train_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_train_end - total_train_start);
        float average_epoch_time = total_train_duration.count() / (epoch + 1 - starting_epoch);
        cout << "Average Time per Epoch: " << average_epoch_time << " s" << endl;

        epoch++;
    }
}

torch::Tensor Trainer::compute_r_loss(torch::Tensor pred, torch::Tensor gt) {
    torch::Tensor gt_mask = gt != 0;
    torch::Tensor gt_masked = torch::masked_select(gt, gt_mask);
    torch::Tensor pred_masked = torch::masked_select(pred, gt_mask);
    // Compute the loss
    torch::Tensor loss = loss_radial(pred_masked, gt_masked);
    // Normalize the loss
    loss = loss / static_cast<float>(gt_masked.size(0));

    return loss;
}

//torch::Tensor Trainer::compute_r_loss(torch::Tensor pred, torch::Tensor gt) {
//    auto nonzero_indices = torch::nonzero(gt);
//    auto pred_nonzero = torch::index_select(pred, 0, nonzero_indices);
//    auto gt_nonzero = torch::index_select(gt, 0, nonzero_indices);
//
//    auto loss = loss_radial(pred_nonzero, gt_nonzero) / static_cast<float>(nonzero_indices.size(0));
//
//    return loss;
//}

void Trainer::printProgressBar(int current, int total, int width)
{
    float progress = float(current) / float(total);
	int barWidth = width - 7;
	cout << "[";
	int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
		if (i < pos) cout << "=";
		else if (i == pos) cout << ">";
		else cout << " ";
	}
	cout << "] " << int(progress * 100.0) << " %\r";
	cout.flush();
}

//void Trainer::test_compute_r_loss() {
//    // Test compute_r_loss function
//    // Pass two tensors, one with 0s and one without
//    torch::Tensor pred = torch::tensor({ 1.0, 2.0, 3.0, 4.0 });
//    torch::Tensor gt = torch::tensor({ 1, 0, 3, 4 });
//
//    cout << "Expected Loss: 0.5" << endl;
//    torch::Tensor loss = compute_r_loss(pred, gt);
//    cout << "Loss: " << loss << endl;
//}

// Cannot implement due to the way the dataloader works
// void Trainer::train_epoch() {
//     cout << string(50, '-') << endl;
//     cout << string(23, ' ') << "Training" << endl << endl;;
// }
// 
// 
// void Trainer::validate()
// {
//    cout << string(50, '-') << endl;
//    cout << string(23, ' ') << "Validating" << endl << endl;
// }

void Trainer::test() {
    cout << string(50, '=') << endl;
    cout << string(18, ' ') << "Begining Testing" << endl << endl;
    //Requires accumulator space for testing each metric
}

void Trainer::store_model(std::string path)
{
    torch::serialize::OutputArchive model_out;

    model->save(model_out);

    model_out.save_to(path + "/model.pt");
}

void Trainer::output_pred(const int& idx, const string& path)
{
    // Stores tensor to txt file, since archive, pickle, and jit doesn't work
    // - Look into implementation of archive, pickle, and jit
    cout << string(100, '=') << endl;
    string out_path = out + "/" + path;
    cout << "Storing Output:\n" << out_path << "/score_" << to_string(idx) << endl;
    cout << out_path << "/score_rad_" << to_string(idx) << endl;


    std::filesystem::path outPath(out_path);
    if (!std::filesystem::is_directory(outPath)) {
        if (std::filesystem::create_directories(outPath)) {
            cout << "Directory created" << endl;
        }
        else {
            cout << "Error creating directory" << endl;
        }
    }

    torch::Device device(device_type);

    auto val_dataset = RData(opts.root_dataset, opts.dname, "val", opts.class_name, opts.kpt_num);

    model->to(torch::kCPU);

    model->eval();

    auto data_tensor = val_dataset.get(idx).data();
    auto sem_target = val_dataset.get(idx).sem_target();
    // cout << "Data tensor size: " << data_tensor.sizes() << endl;

    auto batch = torch::stack(data_tensor, 0);

    // cout << "Batch Tensor Size: " << batch.sizes() << endl;

    auto output = model->forward(batch);

    auto score = std::get<0>(output).to(torch::kCPU);
    auto score_rad = std::get<1>(output).to(torch::kCPU);

    auto score_unstack = torch::unbind(score, 0);
    auto score_rad_unstack = torch::unbind(score_rad, 0);

 
    auto out_score = score_unstack[0];
    auto out_score_rad = score_rad_unstack[0];


    out_score = out_score.permute({ 1,2,0 });
    out_score_rad = out_score_rad.permute({ 1,2,0 });

 
    auto score_path = out_path + "/score_" + std::to_string(idx) + ".txt";
    auto score_rad_path = out_path + "/score_rad_" + std::to_string(idx) + ".txt";
    tensorToFile(out_score, score_path);
    tensorToFile(out_score_rad, score_rad_path);
    

    model->to(device);
}

void Trainer::tensorToFile(const torch::Tensor& tensor, const std::string& filename) {
    /*===============================================================================================
    Write a tensor to a file in binary format. The file format is as follows:
    1. The size of the tensor (number of dimensions and size of each dimension)
    2. The number of elements in the tensor
    3. The tensor data
    =================================================================================================
    How to read in data in python:
    import torch
    import struct

    def fileToTensor(filename):
        with open(filename, "rb") as file:
            # Read the number of dimensions from the file
            num_dimensions = struct.unpack("Q", file.read(8))[0]

            # Read the shape of the tensor from the file
            shape = struct.unpack(f"{num_dimensions}q", file.read(8 * num_dimensions))

            # Read the number of elements from the file
            num_elements = struct.unpack("Q", file.read(8))[0]

            # Read the tensor data from the file
            tensor_data = struct.unpack(f"{num_elements}f", file.read(4 * num_elements))

            # Create a torch.Tensor with the retrieved shape and data
            tensor = torch.tensor(tensor_data).reshape(shape)

        return tensor
    ================================================================================================*/

    // If file directory doesn't exist, create it.
    // Remove .txt and the file name from the path
    std::filesystem::path filePath(filename);
    std::filesystem::path fileDirectory = filePath.parent_path();
    if (!std::filesystem::is_directory(fileDirectory)) {
        if (std::filesystem::create_directories(fileDirectory)) {
			cout << "Directory created" << endl;
		}
        else {
			cout << "Error creating directory" << endl;
		}
	}

    std::ofstream outputFile(filename, std::ios::binary);
    if (!outputFile) {
        cout << "Error opening file: " << filename << endl;
        return;
    }

    torch::IntArrayRef size = tensor.sizes();
    const float* data = tensor.data_ptr<float>();

    size_t numDimensions = size.size();
    outputFile.write(reinterpret_cast<const char*>(&numDimensions), sizeof(size_t));
    outputFile.write(reinterpret_cast<const char*>(size.data()), numDimensions * sizeof(int64_t));

    size_t numElements = tensor.numel();
    outputFile.write(reinterpret_cast<const char*>(&numElements), sizeof(size_t));
    outputFile.write(reinterpret_cast<const char*>(data), numElements * sizeof(float));

    outputFile.close();

}
