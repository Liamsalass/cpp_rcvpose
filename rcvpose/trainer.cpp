#include "trainer.h"

using namespace std;

Trainer::Trainer(Options& options) : opts(options)
{
    cout << string(50, '=') << endl;
    cout << string (12, ' ') << "Initializing Trainer" << endl << endl;
 

    bool use_cuda = torch::cuda::is_available();
    device_type = use_cuda ? torch::kCUDA : torch::kCPU;
    cout << "Using " << (use_cuda ? "CUDA" : "CPU") << endl;
    cout << "Setting up model" << endl;
    // Instantiate the model
    try {
        model = DenseFCNResNet152(3, 2);
        model->to(device_type);
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
	}
    catch (const torch::Error& e) {
		cout << "Error: " << e.msg() << endl;
		return;
	}
    try {
        loss_sem = torch::nn::L1Loss();
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
    cout << string (10, ' ') << "Starting Training Cycle" << endl << endl;
    cout << "Setting up dataset loader" << endl;
    // Instantiate the dataset
    auto train_dataset = RData(opts.root_dataset, opts.dname, "trainall", opts.class_name, opts.kpt_num);

    // Instantiate the dataloaders 
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(opts.batch_size).workers(1)
    );

    torch::optional<size_t> train_size = train_dataset.size();

    if (train_size.value() == 0 || !train_size.has_value()) {
		cout << "Error: Could not get size of training dataset" << endl;
		return;
	}
    cout << "Data Set Size : " << train_size.value() << endl;

    int max_epoch = static_cast<int>(std::ceil(1.0 * max_iteration/ train_size.value()));

    cout << "Max Epochs : " << max_epoch << endl;

    void train_epoch(); {
        cout << "Starting Epoch " << epoch << endl;
    };

}

torch::Tensor Trainer::compute_r_loss(torch::Tensor pred, torch::Tensor gt) {
    torch::Tensor loss = loss_radial(pred.masked_select(gt.ne(0)), gt.masked_select(gt.ne(0))) / static_cast<double>(torch::nonzero(gt).size(0));
    return loss;
}

void Trainer::test_compute_r_loss() {
 
    torch::Tensor pred = torch::tensor({ 1.0, 2.0, 3.0, 4.0 });
    torch::Tensor gt = torch::tensor({ 1, 0, 3, 4 });
    torch::Tensor expected_loss = torch::tensor(3.5);

    torch::Tensor loss = compute_r_loss(pred, gt);

    std::cout << "Loss: " << loss << std::endl;
    std::cout << "Expected Loss: " << expected_loss << std::endl;

    if (torch::allclose(loss, expected_loss)) {
        std::cout << "Test passed!" << std::endl;
    }
    else {
        std::cout << "Test failed!" << std::endl;
    }
}

void Trainer::validate()
{
    cout << "Starting Testing Cycle" << endl;
    cout << "Setting up dataset loader" << endl;
    //Set up dataloaders
    try {
        auto val_dataset = RData(opts.root_dataset, opts.dname, "test", opts.class_name, opts.kpt_num);
        auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(val_dataset),
            torch::data::DataLoaderOptions().batch_size(opts.batch_size).workers(1)
        );
    }
    catch (const torch::Error& e) {
        cout << "Error: " << e.msg() << endl;
        return;
    }
}

