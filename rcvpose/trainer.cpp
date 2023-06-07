#include "trainer.h"

using namespace std;

Trainer::Trainer(Options& options) : opts(options)
{
    cout << string(100, '=') << endl;
    cout << string (34, ' ') << "Initializing Trainer" << endl << endl;
 

    bool use_cuda = torch::cuda::is_available();
    device_type = use_cuda ? torch::kCUDA : torch::kCPU;
    torch::Device device(device_type);
    cout << "Using " << (use_cuda ? "CUDA" : "CPU") << endl;
    cout << "Setting up model" << endl;

    if (!opts.resume_train) {
        // Instantiate the model
        try {
            model = DenseFCNResNet152(3, 2);
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

        epoch = 0;
        
    } 
    else {
        // Load model from checkpoint
        cout << "Loading model from checkpoint" << endl;
        CheckpointLoader loader(opts.model_dir + "/current");
        epoch = loader.getEpoch();
        cout << "Epoch: " << epoch << endl;
        model = loader.getModel();
        model->to(device);
        cout << "Model loaded" << endl;
        optim = loader.getOptimizer(); 
        optim->parameters() = model->parameters();
        //std::vector<double> lr_list = loader.getLrList();
        //torch::autograd::variable_list lr_list_var;
        //for (int i = 0; i < lr_list.size(); i++) {
		//	lr_list_var.push_back(torch::autograd::make_variable(torch::tensor(lr_list[i])));
		//}

        //Set the lrs
        //optim->add_param_group(torch::optim::OptimizerParamGroup(lr_list_var));

        cout << "Optimizer loaded" << endl;

        //Print out optimizer state:
        //for (auto& group : optim->param_groups()) {
        //    cout << "Group: " << endl;
		//	for (auto& p : group.params()) {
		//		cout << "Param: " << p << endl;
		//	}
		//}
    }


    cout << "Setting up loss function" << endl;

    // Instantiate the loss function
    try {
        loss_radial = torch::nn::L1Loss(torch::nn::L1LossOptions().reduction(torch::kSum));
        //loss_radial->to(torch::kCPU);

        loss_radial->to(device);
    }
    catch (const torch::Error& e) {
        cout << "Error: " << e.msg() << endl;
        return;
    }
    try {
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
    // Begin training cycle
    while (epoch < max_epoch) {
        cout << string(100, '-') << endl;
        cout << string(43, ' ') << "Epoch " << epoch << endl;

        //Train epoch code
        // Can't use functions due to the way the dataloader works, need to figure out new method
        //train_epoch();
        //validate();
        
        //Training Epoch

        // ========================================================================================== \\
        //TODO, figure out how to move whole batches to the gpu
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

            // Create batch tensors by stacking individual tensors
            auto data = torch::stack(batch_data, 0).to(device); 
            auto target = torch::stack(batch_target, 0).to(device);
            auto sem_target = torch::stack(batch_sem_target, 0).to(device);

            // Print info on data's shape
            //std::cout << "Input Data Shape: " << data.sizes() << std::endl;

            optim->zero_grad();
            // std::tuple<torch::Tensor> scores;
            auto scores = model->forward(data);

            auto& score = std::get<0>(scores);
            auto& score_rad = std::get<1>(scores);
            score_rad = score_rad.permute({ 1, 0, 2, 3 }) * sem_target.permute({ 1, 0, 2, 3 });
            score_rad = score_rad.permute({ 1, 0, 2, 3 });

            torch::Tensor loss_s = loss_sem(score, sem_target);
            torch::Tensor loss_r = compute_r_loss(score_rad, target);

            torch::Tensor loss = loss_r + loss_s;

            loss.backward();

            optim->step();

            auto np_loss = loss.detach().cpu().numpy_T();
            //auto np_loss_r = loss_r.detach().cpu().numpy_T();
            //auto np_loss_s = loss_s.detach().cpu().numpy_T();

            if (np_loss.numel() == 0)
                std::runtime_error("Loss is empty");

        }

        cout << "\r" << string(100, ' ');
        auto train_end = std::chrono::steady_clock::now();
        auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);
        cout << "\rTraining Time: " << train_duration.count()/1000 << " s" << endl;

        // ========================================================================================== \\
        //Validation Epoch
        cout << "Validation Epoch" << endl;
        model->eval();
        float val_loss = 0;
        count = 0;
        torch::NoGradGuard no_grad;
        auto val_start = std::chrono::steady_clock::now();

        for (const auto& batch : *val_loader) {
            count = batch.size() + count;
            iteration_val = batch.size() + iteration_val;
            printProgressBar(count, val_size.value(), 80);
            // Extract data, target, and sem_target from batch
            std::vector<torch::Tensor> batch_data;
            std::vector<torch::Tensor> batch_target;
            std::vector<torch::Tensor> batch_sem_target;

            for (const auto& example : batch) {
                batch_data.push_back(example.data());
                batch_target.push_back(example.target());
                batch_sem_target.push_back(example.sem_target());
            }

            // Create batch tensors by stacking individual tensors
            auto img = torch::stack(batch_data, 0).to(device);
            auto target = torch::stack(batch_target, 0).to(device);
            auto sem_target = torch::stack(batch_sem_target, 0).to(device);

            // Pass the batch tensors to the GPU if needed
            // auto img = data.to(device);
            // auto target = target.to(device);
            // auto sem_target = sem_target.to(device);
            auto output = model->forward(img);

            auto score = std::get<0>(output);
            auto score_rad = std::get<1>(output);

            auto loss_s = loss_sem(score, sem_target);
            auto loss_r = compute_r_loss(score_rad, target);
            auto loss = loss_r + loss_s;

            if (loss.numel() == 0)
                std::runtime_error("Loss is empty");

            val_loss += loss.item<float>();

        }
        cout << "\r" << string(100, ' ');
        auto val_end = std::chrono::steady_clock::now();
        auto val_duration = std::chrono::duration_cast<std::chrono::milliseconds>(val_end - val_start);
        cout << "\rValidation Time: " << val_duration.count()/1000 << " s" << endl;
        cout.flush();

        val_loss /= val_size.value();
        float mean_acc = val_loss;
        cout << "Mean Loss: " << mean_acc << endl;
        bool is_best = mean_acc < best_acc_mean;
        if (is_best) 
            best_acc_mean = mean_acc;
        
        cout << "Iterations: " << iteration << endl;

        try {

            std::string save_location = out + "/current";
            if (!std::filesystem::is_directory(save_location))
                std::filesystem::create_directory(save_location);
            //Save epoch number, iteration number, model name, best accuracy, and loss
            torch::serialize::OutputArchive output_model_info;
            output_model_info.write("epoch", epoch);
            output_model_info.write("iteration", iteration);
            output_model_info.write("arch", model->name());
            output_model_info.write("best_acc_mean", best_acc_mean);
            output_model_info.write("loss", val_loss);
            output_model_info.write("optimizer", opts.optim);

            //std::vector<double> lr_list;
            //Save optimizer learning rates
            //for (auto& optim_pg : optim->param_groups()) {
            //    auto options = optim_pg.options();
            //    auto lr = options.get_lr();
            //    lr_list.push_back(lr);
            //}
            //output_model_info.write("lr_list", lr_list);
            output_model_info.save_to(save_location + "/info.pt");

            //Save model
            torch::serialize::OutputArchive output_model_archive;
            model->to(torch::kCPU);
            model->save(output_model_archive);
            model->to(device);
            output_model_archive.save_to(save_location + "/model.pt");
            //Save optimizer
            torch::serialize::OutputArchive output_optim_archive;
            optim->save(output_optim_archive);
            output_optim_archive.save_to(save_location + "/optim.pt");


            if (is_best) {
                std::filesystem::path best_path(out + "/model_best");
                if (!std::filesystem::is_directory(best_path))
                    std::filesystem::create_directory(best_path);

                std::string model_best_path = out + "/model_best";
                std::filesystem::copy(save_location, model_best_path, std::filesystem::copy_options::overwrite_existing);
            }
        }
        catch (std::exception& e) {
			std::cout << "Error saving model: " << e.what() << std::endl;
		}
        

        current_lr.clear();
        
        if (epoch % 70 == 0 && epoch != 0) {
            //optim->options.learning_rate(optim->options.learning_rate() * 0.1);
            for (auto& param_group : optim->param_groups()) {
                if (param_group.has_options()) {
                    int new_lr = param_group.options().get_lr() * 0.1;
                    if (opts.optim == "adam") 
                        static_cast<torch::optim::AdamOptions &>(param_group.options()).lr(new_lr);
                    if (opts.optim == "sgd")
                        static_cast<torch::optim::SGDOptions &>(param_group.options()).lr(new_lr);
                    cout << "Learning rate reduction, new LR: " << new_lr << endl;
                    current_lr.push_back(new_lr);
                }
            }
        }
        if (iteration >= max_iteration) {
            break;
        }

        epoch++;
    }
}

//Using L1Loss function, remove all 0s from dataset before calculating loss, ensure shape of both tensors is the same before computing loss
torch::Tensor Trainer::compute_r_loss(torch::Tensor pred, torch::Tensor gt) {
    try {
        // Compute the radial loss
        // Remove all 0s from the ground truth tensor
        torch::Tensor gt_mask = gt != 0;
        torch::Tensor gt_masked = torch::masked_select(gt, gt_mask);
        torch::Tensor pred_masked = torch::masked_select(pred, gt_mask);
        // Compute the loss
        torch::Tensor loss = loss_radial(pred_masked, gt_masked);
        return loss;
    }
    catch (const torch::Error& e) {
		cout << "Error: " << e.msg() << endl;
        return torch::Tensor(torch::empty({}));
	}
}

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

void Trainer::test_compute_r_loss() {
    // Test compute_r_loss function
    // Pass two tensors, one with 0s and one without
    torch::Tensor pred = torch::tensor({ 1.0, 2.0, 3.0, 4.0 });
    torch::Tensor gt = torch::tensor({ 1, 0, 3, 4 });

    cout << "Expected Loss: 0.5" << endl;
    torch::Tensor loss = compute_r_loss(pred, gt);
    cout << "Loss: " << loss << endl;
}

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