#include "trainer.h"

using namespace std;


Trainer::Trainer(const Options& opts, DenseFCNResNet152& model) 
{
    cout << string(100, '=') << endl;
    cout << string (35, ' ') << "Initializing Trainer" << endl << endl;

    bool use_cuda = torch::cuda::is_available();
    device_type = use_cuda ? torch::kCUDA : torch::kCPU;
    torch::Device device(device_type);

    model->to(device);

    if (opts.verbose) {
        cout << "Using " << (use_cuda ? "CUDA" : "CPU") << endl;
        cout << "Setting up model" << endl;
    }
    out = opts.model_dir;

    if (!opts.resume_train) {
        if (opts.verbose) {
            cout << "Setting up optimizer" << endl;
        }
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
            if (opts.verbose) {
                cout << "Param Group " << count << " with LR value: " << params.options().get_lr() << endl;
            }
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
            CheckpointLoader loader(out, true);
            epoch = loader.getEpoch();
            starting_epoch = epoch;
            if (opts.verbose) {
                cout << "Epoch: " << epoch << endl;
            }
            optim = loader.getOptimizer();
            optim->parameters() = model->parameters();
            current_lr = loader.getLrList();
            if (opts.verbose) {
                cout << "Optimizer loaded" << endl;
            }
            int count = 0;
            for (auto& params : optim->param_groups()) {
                params.options().set_lr(current_lr[0]);
                if (opts.verbose) {
                    cout << "Param Group " << count << " with LR value: " << params.options().get_lr() << endl;
                }
                count++;
            }

            current_lr.clear();
            current_lr.push_back(opts.initial_lr);

            best_acc_mean = loader.getBestAccuracy();

            float prev_loss = loader.getLoss();
            if (opts.verbose) {
                cout << "Best Accuracy: " << best_acc_mean << endl;

                cout << "Previous Loss: " << prev_loss << endl;
            }

        } 
        catch (const torch::Error& e) {
            cout << "Cannot Resume Training" << endl;
			cout << "Error: " << e.msg() << endl;
            return;
		}
    }

    
    if (opts.verbose) {
        cout << "Setting up loss function" << endl;
    }

    // Instantiate the loss function
    try {
        loss_geo = torch::nn::SmoothL1Loss(torch::nn::SmoothL1LossOptions().reduction(torch::kMean));
        loss_radial = torch::nn::L1Loss(torch::nn::L1LossOptions().reduction(torch::kSum));
        

        loss_geo->to(device);
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
        std::cout << "Output directory already exists" << std::endl;
    }


    cout << "Model Path: " << out << endl;

    cout << "Trainer Initialized" << endl;

    
}

void Trainer::train(Options& opts, DenseFCNResNet152& model)
{
    cout << string(100, '=') << endl; 
    cout << string(24, ' ') << "Begining Training Initialization" << endl << endl;
    

    //Select device base on opts.gpu_id

    torch::Device device(device_type);
    if (opts.verbose) {
        cout << "Setting up dataset loader" << endl;
    }
    // Instantiate the training dataset
    // Can use .map(torch::data::transforms::Stack<>()) to stack batches into a single tensor
    auto train_dataset = RData(opts.root_dataset, opts.dname, "train", opts.class_name);
    torch::optional<size_t> train_size = train_dataset.size();

    auto val_dataset = RData(opts.root_dataset, opts.dname, "val", opts.class_name);
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
    epochs_without_improvement = 0;
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
        

        if(opts.verbose){
            cout << string(100, '-') << endl;
            cout << string(43, ' ') << "Epoch " << epoch << endl;
            cout << "Training Epoch" << endl;
        } else {
            cout << "Epoch : " << epoch << endl;
        }

        // ========================================================================================== \\
        // ====================================== Training ========================================== \\

        int count = 0;
        model->train();

        auto train_start = std::chrono::steady_clock::now();
        for (const auto& batch : *train_loader) {
            if (opts.verbose) {
                printProgressBar(count, train_size.value(), 75);
            }
            count = batch.size() + count;

            iteration = batch.size() + iteration;

            std::vector<torch::Tensor> batch_data;
            std::vector<torch::Tensor> batch_radial_1;
            std::vector<torch::Tensor> batch_radial_2;
            std::vector<torch::Tensor> batch_radial_3;
            std::vector<torch::Tensor> batch_sem_target;

            for (const auto& example : batch) {
                batch_data.push_back(example.data());
                batch_radial_1.push_back(example.rad_1());
                batch_radial_2.push_back(example.rad_2());
                batch_radial_3.push_back(example.rad_3());
                batch_sem_target.push_back(example.sem_target());
            }

            optim->zero_grad();

            auto data = torch::stack(batch_data, 0).to(device);
            auto rad_1 = torch::stack(batch_radial_1, 0).to(device);
            auto rad_2 = torch::stack(batch_radial_2, 0).to(device);
            auto rad_3 = torch::stack(batch_radial_3, 0).to(device);
            auto sem_target = torch::stack(batch_sem_target, 0).to(device);

            torch::Tensor scores = model->forward(data);

            auto score_rad_1 = scores.index({ torch::indexing::Slice(), 0}).unsqueeze(1);
            auto score_rad_2 = scores.index({ torch::indexing::Slice(), 1}).unsqueeze(1);
            auto score_rad_3 = scores.index({ torch::indexing::Slice(), 2}).unsqueeze(1);
            auto score_sem = scores.index({ torch::indexing::Slice(), 3 }).unsqueeze(1);

            score_rad_1 = (score_rad_1.permute({ 1, 0, 2, 3 }) * sem_target.permute({ 1, 0, 2, 3 }));
            score_rad_2 = (score_rad_2.permute({ 1, 0, 2, 3 }) * sem_target.permute({ 1, 0, 2, 3 }));
            score_rad_3 = (score_rad_3.permute({ 1, 0, 2, 3 }) * sem_target.permute({ 1, 0, 2, 3 }));


            score_rad_1 = score_rad_1.permute({ 1, 0, 2, 3 });
            score_rad_2 = score_rad_2.permute({ 1, 0, 2, 3 });
            score_rad_3 = score_rad_3.permute({ 1, 0, 2, 3 });


            torch::Tensor loss_s = loss_sem(score_sem, sem_target);
            torch::Tensor loss_r = compute_r_loss(score_rad_1, rad_1);
            loss_r += compute_r_loss(score_rad_2, rad_2);
            loss_r += compute_r_loss(score_rad_3, rad_3);
            torch::Tensor loss_g = compute_geo_constraint(score_rad_1, score_rad_2, score_rad_3, rad_1, rad_2, rad_3);
            auto loss_r_g = loss_r*0.8 + loss_g*0.2;

            torch::Tensor loss = loss_r_g + loss_s;

            //cout << "Radial Loss: " << loss_r.item<float>() << " Semantic Loss: " << loss_s.item<float>() << " Total Loss: " << loss.item<float>() << "\r";

            loss.backward();

            optim->step();

            auto np_loss = loss.detach().cpu().numpy_T();

            if (np_loss.numel() == 0)
                std::runtime_error("Loss is empty");
        }
    

        auto train_end = std::chrono::steady_clock::now();
        auto train_duration = std::chrono::duration_cast<std::chrono::seconds>(train_end - train_start);
        if (opts.verbose) {
            cout << "\r" << string(80, ' ') << "\r\r";
            cout << "Training Time: " << train_duration.count() << " s" << endl;
            cout << "Validation Epoch" << endl;
        }

        // ========================================================================================== \\
        //                                  Validation Epoch 									       \\
        
        if(((epoch % 10) == 0) && (epoch != 0)){
            model->eval();
            float val_loss = 0, sem_loss = 0, radial_loss = 0, geometric_loss = 0;
            count = 0;

            auto val_start = std::chrono::steady_clock::now();

            int val_count = 0;

            for (const auto& batch : *val_loader) {
                if (opts.verbose) {
                    printProgressBar(count, val_size.value(), 75);
                }
                count = batch.size() + count;
                iteration_val = batch.size() + iteration_val;
                torch::NoGradGuard no_grad;

                std::vector<torch::Tensor> batch_data;
                std::vector<torch::Tensor> batch_radial_1;
                std::vector<torch::Tensor> batch_radial_2;
                std::vector<torch::Tensor> batch_radial_3;
                std::vector<torch::Tensor> batch_sem_target;

                for (const auto& example : batch) {
                    batch_data.push_back(example.data());
                    batch_radial_1.push_back(example.rad_1());
                    batch_radial_2.push_back(example.rad_2());
                    batch_radial_3.push_back(example.rad_3());
                    batch_sem_target.push_back(example.sem_target());
                }


                auto img = torch::stack(batch_data, 0);
                auto rad_1 = torch::stack(batch_radial_1, 0);
                auto rad_2 = torch::stack(batch_radial_2, 0);
                auto rad_3 = torch::stack(batch_radial_3, 0);
                auto sem_target = torch::stack(batch_sem_target, 0);


                img = img.to(device);
                rad_1 = rad_1.to(device);
                rad_2 = rad_2.to(device);
                rad_3 = rad_3.to(device);
                sem_target = sem_target.to(device);

                torch::Tensor output = model->forward(img);

                auto score_rad_1 = output.index({ torch::indexing::Slice(), 0, torch::indexing::Slice(), torch::indexing::Slice() }).unsqueeze(1);
                auto score_rad_2 = output.index({ torch::indexing::Slice(), 1, torch::indexing::Slice(), torch::indexing::Slice() }).unsqueeze(1);
                auto score_rad_3 = output.index({ torch::indexing::Slice(), 2, torch::indexing::Slice(), torch::indexing::Slice() }).unsqueeze(1);
                auto score_sem = output.index({ torch::indexing::Slice(), 3, torch::indexing::Slice(), torch::indexing::Slice() }).unsqueeze(1);

                val_count++;

                auto loss_s = loss_sem(score_sem, sem_target);
                auto loss_r = compute_r_loss(score_rad_1, rad_1);
                loss_r += compute_r_loss(score_rad_2, rad_2);
                loss_r += compute_r_loss(score_rad_3, rad_3);
                auto loss_g = compute_geo_constraint(score_rad_1, score_rad_2, score_rad_3, rad_1, rad_2, rad_3);
                auto loss_r_g = loss_r * 0.8 + loss_g * 0.2;

                auto loss = loss_r_g + loss_s;


                //cout << "Loss_r: " << loss_r.item<float>() << " Loss_s: " << loss_s.item<float>() << "\r";

                if (loss.numel() == 0)
                    std::runtime_error("Loss is empty");

                val_loss += loss.item<float>();
                sem_loss += loss_s.item<float>();
                radial_loss += loss_r.item<float>();
                geometric_loss += loss_g.item<float>();
            }


            auto val_end = std::chrono::steady_clock::now();
            auto val_duration = std::chrono::duration_cast<std::chrono::seconds>(val_end - val_start);

            if (opts.verbose) {
                cout << "\r" << string(80, ' ') << "\r";
                cout << "Validation Time: " << val_duration.count() << " s" << endl;
            }

            val_loss /= val_size.value();
            float mean_acc = val_loss;

        

            cout << "Mean Loss: " << mean_acc << endl;
            cout << "\tSemantic Loss: " << sem_loss / val_size.value() << endl;
            cout << "\tRadial Loss: " << radial_loss / val_size.value() << endl;
            cout << "\tGeometric Loss: " << geometric_loss / val_size.value() << endl;

            bool is_best = mean_acc < best_acc_mean;

            if (is_best) {
                best_acc_mean = mean_acc;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement++;
            }
            if (opts.verbose) {
                cout << "Iterations: " << iteration << endl;
                cout << "Epochs without improvement: " << epochs_without_improvement << endl;
            }

            //================================================================\\
            //                  Save Model and Optimizer                      \\


            try {
                std::string save_location;
                if (is_best) {
                    if (opts.verbose) {
                        cout << "Saving New Best Model" << endl;
                    }
                    save_location = out + "/model_best";
                }
                else {
                    if (opts.verbose) {
                        cout << "Saving Current Model" << endl;
                    }
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


        // Reduce learning rate every 70 epoch
        if (!opts.reduce_on_plateau){
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

        auto total_train_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_train_end - total_train_start);
        float average_epoch_time = total_train_duration.count() / (epoch + 1 - starting_epoch);
  
        if (opts.verbose) {
            cout << "Epoch Training Time: " << epoch_total_time.count() << " s" << endl;
            cout << "Average Time pe6r Epoch: " << average_epoch_time << " s" << endl;
        }
        else {
            if (epoch % 10 == 0) {
                cout << "Average Time per Epoch: " << average_epoch_time << " s" << endl;
            }
        }
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

torch::Tensor Trainer::compute_geo_constraint(torch::Tensor score_rad_1, torch::Tensor score_rad_2, torch::Tensor score_rad_3, torch::Tensor rad_1, torch::Tensor rad_2, torch::Tensor rad_3) {
    torch::Tensor gt_mask = rad_1 != 0;
    torch::Tensor score_rad_1_masked = torch::masked_select(score_rad_1, gt_mask);
    torch::Tensor score_rad_2_masked = torch::masked_select(score_rad_2, gt_mask);
    torch::Tensor score_rad_3_masked = torch::masked_select(score_rad_3, gt_mask);
    torch::Tensor score_diff_1_2 = torch::abs(torch::sub(score_rad_1_masked, score_rad_2_masked));
    torch::Tensor score_diff_1_3 = torch::abs(torch::sub(score_rad_1_masked, score_rad_3_masked));
    torch::Tensor score_diff_2_3 = torch::abs(torch::sub(score_rad_3_masked, score_rad_2_masked));

    torch::Tensor rad_1_masked = torch::masked_select(rad_1, gt_mask);
    torch::Tensor rad_2_masked = torch::masked_select(rad_2, gt_mask);
    torch::Tensor rad_3_masked = torch::masked_select(rad_3, gt_mask);

    torch::Tensor diff_1_2 = torch::abs(torch::sub(rad_1_masked, rad_2_masked));
    torch::Tensor diff_1_3 = torch::abs(torch::sub(rad_1_masked, rad_3_masked));
    torch::Tensor diff_2_3 = torch::abs(torch::sub(rad_3_masked, rad_2_masked));
    // Compute the loss
    torch::Tensor loss_1_2 = loss_geo(score_diff_1_2, diff_1_2);
    torch::Tensor loss_1_3 = loss_geo(score_diff_1_3, diff_1_3);
    torch::Tensor loss_2_3 = loss_geo(score_diff_2_3, diff_2_3);
    // Normalize the loss
    torch::Tensor loss = (loss_1_2 / static_cast<float>(gt_mask.size(0)) +
        loss_1_3 / static_cast<float>(gt_mask.size(0)) +
        loss_2_3 / static_cast<float>(gt_mask.size(0)))/3;

    return loss;
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



