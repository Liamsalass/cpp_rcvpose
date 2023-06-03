#include "trainer.h"

using namespace std;

Trainer::Trainer(Options& options) : opts(options)
{
    cout << string(50, '=') << endl;
    cout << string (12, ' ') << "Initializing Trainer" << endl << endl;
 

    bool use_cuda = torch::cuda::is_available();
    device_type = use_cuda ? torch::kCUDA : torch::kCPU;
    torch::Device device(device_type);
    cout << "Using " << (use_cuda ? "CUDA" : "CPU") << endl;
    cout << "Setting up model" << endl;
    // Instantiate the model
    try {
        model = DenseFCNResNet152(3, 2);
        //model->to(device);
        model->to(torch::kCPU);
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
			optim = new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(opts.initial_lr));
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


	cout << "Setting up loss function" << endl;
	// Instantiate the loss function
    try {
		loss_radial = torch::nn::L1Loss(torch::nn::L1LossOptions().reduction(torch::kSum));
        loss_radial->to(torch::kCPU);

        //loss_radial->to(device);
	}
    catch (const torch::Error& e) {
		cout << "Error: " << e.msg() << endl;
		return;
	}
    try {
        loss_sem = torch::nn::L1Loss();
        loss_sem->to(torch::kCPU);
        //loss_sem->to(device);
    }
    catch (const torch::Error& e) {
        cout << "Error: " << e.msg() << endl;
        return;
    }

    cout << "Radial Loss Function: " << loss_radial << endl;
    cout << "Semantic Loss Function: " << loss_sem << endl;

	cout << "Setting up training parameters" << endl;
	// Set up training parameters
	epoch = 0;
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
    cout << string(50, '=') << endl; 
    cout << string(18, ' ') << "Begining Training Initialization" << endl << endl;
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
    for (int epoch = 0; epoch < max_epoch; epoch++) {
        cout << string(50, '-') << endl;
        cout << string(23, ' ') << "Epoch " << epoch << endl;

        //Train epoch code
        // Can't use functions due to the way the dataloader works, need to figure out new method
        //train_epoch();
        //validate();
        
        //Training Epoch


        // ========================================================================================== \\
        //TODO, figure out how to move whole batches to the gpu
        model->train();
        for (const auto& batch : *train_loader) {

            std::vector<torch::Tensor> batch_data;
            std::vector<torch::Tensor> batch_target;
            std::vector<torch::Tensor> batch_sem_target;

            for (const auto& example : batch) {
                batch_data.push_back(example.data());
                batch_target.push_back(example.target());
                batch_sem_target.push_back(example.sem_target());
            }
            //Print info about the tensors
            std::cout << "Batch Data Shape: " << batch_data[0].sizes() << std::endl;
            std::cout << "Batch Target Shape: " << batch_target[0].sizes() << std::endl;
            std::cout << "Batch Semantic Target Shape: " << batch_sem_target[0].sizes() << std::endl;

            //Print out the data
            //std::cout << "Batch Data: " << batch_data[0] << std::endl;
            //std::cout << "Batch Target: " << batch_target[0] << std::endl;
            //std::cout << "Batch Semantic Target: " << batch_sem_target[0] << std::endl;

            // Create batch tensors by stacking individual tensors
            auto data = torch::stack(batch_data, 0); 
            auto target = torch::stack(batch_target, 0);  
            auto sem_target = torch::stack(batch_sem_target, 0);  

            // Print info on data's shape
            std::cout << "Data Shape: " << data.sizes() << std::endl;

            optim->zero_grad();

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
            auto np_loss_r = loss_r.detach().cpu().numpy_T();
            auto np_loss_s = loss_s.detach().cpu().numpy_T();

            if (np_loss.numel() == 0)
                std::runtime_error("Loss is empty");

            cout << "Epoch: " << epoch << " Iteration: " << iteration << " Loss: " << np_loss << " Loss_r: " << np_loss_r << " Loss_s: " << np_loss_s << endl;

            if (iteration >= max_iteration)
                break;
        }




        // ========================================================================================== \\
        //Validation Epoch
        model->eval();
        float val_loss = 0;
        torch::NoGradGuard no_grad;
        for (const auto& batch : *val_loader) {
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
            auto img = torch::stack(batch_data, 0);  
            auto target = torch::stack(batch_target, 0);
            auto sem_target = torch::stack(batch_sem_target, 0);  

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

            iteration_val++;
        }


        val_loss /= val_size.value();
        float mean_acc = val_loss;
        bool is_best = mean_acc < best_acc_mean;
        if (is_best) {
            best_acc_mean = mean_acc;
        }

        std::string save_name = "ckpt.pth.tar";

        //Figuere out save functions
        //torch::serialize::OutputArchive output_archive;
        //output_archive.write("epoch", epoch);
        //output_archive.write("iteration", iteration);
        //output_archive.write("arch", model->name());
        //optim->save(output_archive);
        //model->save(output_archive);
        //output_archive.write("best_acc_mean", best_acc_mean);
        //output_archive.write("loss", val_loss);
        //
        //torch::save(output_archive, out + "/" + save_name);
        //
        //if (is_best) {
        //    std::string model_best_path = out + "/model_best.pth.tar";
        //    std::string save_path = out + "/" + save_name;
        //    std::filesystem::copy_file(save_path, model_best_path, std::filesystem::copy_options::overwrite_existing);
        //}

        if (epoch % 70 == 0 && epoch != 0) {
            //optim->options.learning_rate(optim->options.learning_rate() * 0.1);
            for (auto& g : optim->param_groups()) {
                if (g.has_options()) {
                    auto options = g.options();
                    options.set_lr(options.get_lr() * 0.1);
                    g.set_options(std::make_unique<torch::optim::OptimizerOptions>(options));
                    cout << "Learning Rate Reduction. New Learning Rate: " << options.get_lr() << endl;
                }
            }
        }
        if (iteration >= max_iteration) {
            break;
        }
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