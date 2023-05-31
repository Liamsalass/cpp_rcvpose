#include "rcvpose.h"

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
    cout << string(50, '=') << endl;
    cout << string(17, ' ') << "Summary" << endl << endl;

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

        trainer.test_compute_r_loss();

        trainer.train();
    }
    catch (const std::exception& e) {
		std::cerr << e.what() << '\n';
	}
}


void RCVpose::validate() {
    // Implementation for evaluating the model
}

void RCVpose::compare_models(string model1, string model2) {

}

void RCVpose::saveModel(std::string path) {
    // Implementation for saving the model
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

int RCVpose::test_loaders()
{
    int ret = 0;
    try {
        //Print image data
        RData test(opts.root_dataset, opts.dname, "trainall", opts.class_name, opts.kpt_num);
        torch::optional dataset_size = test.size();
        cout << "Dataset size: " << dataset_size.value() << endl;
        cv::Mat img = test.get_img(0);

        cout << string(50, '=') << endl;
        cout << "Image Data" << endl;
        cout << "Image size: " << img.size() << endl;
        cout << "Image type: " << img.type() << endl;
        cout << "Image channels: " << img.channels() << endl;
        cout << "Image depth: " << img.depth() << endl;
        cout << "Image dims: " << img.dims << endl;
        cout << "Image total: " << img.total() << endl;

        // Add additional print statements for image information
        cout << "Image width: " << img.cols << endl;
        cout << "Image height: " << img.rows << endl;
        cout << "Image element size in bytes: " << img.elemSize() << endl;
        cout << "Image total number of elements: " << img.total() << endl;
        cout << "Image step size (in bytes): " << img.step << endl;
        cout << "Image step size (in pixels): " << img.step1() << endl;

        cv::Mat target = test.get_target(0);
        cout << string(50, '=') << endl;
        cout << "Target Data" << endl;
        cout << "Target size: " << target.size() << endl;
        cout << "Target type: " << target.type() << endl;
        cout << "Target channels: " << target.channels() << endl;
        cout << "Target depth: " << target.depth() << endl;
        cout << "Target dims: " << target.dims << endl;
        cout << "Target total: " << target.total() << endl;

        // Add additional print statements for target information
        cout << "Target width: " << target.cols << endl;
        cout << "Target height: " << target.rows << endl;
        cout << "Target element size in bytes: " << target.elemSize() << endl;
        cout << "Target total number of elements: " << target.total() << endl;
        cout << "Target step size (in bytes): " << target.step << endl;
        cout << "Target step size (in pixels): " << target.step1() << endl;



        // Apply Transform
        cout << string(50, '=') << endl;
        cout << "Testing Transform Function" << endl;
        std::tuple < torch::Tensor, torch::Tensor, torch::Tensor> transform = test.transform(img, target);
        torch::Tensor img_tensor = std::get<0>(transform);
        torch::Tensor lbl_tensor = std::get<1>(transform);
        torch::Tensor sem_lbl_tensor = std::get<2>(transform);

        //Print info about tensors:
        cout << "Image Tensor" << endl;
        cout << "Image size: " << img_tensor.sizes() << endl;
        cout << "Image type: " << img_tensor.dtype() << endl;
        cout << "Image device: " << img_tensor.device() << endl;
        cout << "Image layout: " << img_tensor.layout() << endl;
        cout << "Image requires grad: " << img_tensor.requires_grad() << endl;
        cout << "Image grad: " << img_tensor.grad() << endl;
        cout << "Image grad_fn: " << img_tensor.grad_fn() << endl;
        cout << "Image is_leaf: " << img_tensor.is_leaf() << endl;
        cout << "Image is_cuda: " << img_tensor.is_cuda() << endl;
        cout << "Image is_sparse: " << img_tensor.is_sparse() << endl;
        cout << "Image is_contiguous: " << img_tensor.is_contiguous() << endl;
        cout << "Image numel: " << img_tensor.numel() << endl;
        cout << "Image ndimension: " << img_tensor.ndimension() << endl;
        cout << "Image element_size: " << img_tensor.element_size() << endl << endl;

        cout << "lbl tensor" << endl;
        cout << "lbl size: " << lbl_tensor.sizes() << endl;
        cout << "lbl type: " << lbl_tensor.dtype() << endl;
        cout << "lbl device: " << lbl_tensor.device() << endl;
        cout << "lbl layout: " << lbl_tensor.layout() << endl;
        cout << "lbl requires grad: " << lbl_tensor.requires_grad() << endl;
        cout << "lbl grad: " << lbl_tensor.grad() << endl;
        cout << "lbl grad_fn: " << lbl_tensor.grad_fn() << endl;
        cout << "lbl is_leaf: " << lbl_tensor.is_leaf() << endl;
        cout << "lbl is_cuda: " << lbl_tensor.is_cuda() << endl;
        cout << "lbl is_sparse: " << lbl_tensor.is_sparse() << endl;
        cout << "lbl is_contiguous: " << lbl_tensor.is_contiguous() << endl;
        cout << "lbl numel: " << lbl_tensor.numel() << endl;
        cout << "lbl ndimension: " << lbl_tensor.ndimension() << endl;
        cout << "lbl element_size: " << lbl_tensor.element_size() << endl << endl;

        cout << "sem lbl tensor" << endl;
        cout << "sem lbl size: " << sem_lbl_tensor.sizes() << endl;
        cout << "sem lbl type: " << sem_lbl_tensor.dtype() << endl;
        cout << "sem lbl device: " << sem_lbl_tensor.device() << endl;
        cout << "sem lbl layout: " << sem_lbl_tensor.layout() << endl;
        cout << "sem lbl requires grad: " << sem_lbl_tensor.requires_grad() << endl;
        cout << "sem lbl grad: " << sem_lbl_tensor.grad() << endl;
        cout << "sem lbl grad_fn: " << sem_lbl_tensor.grad_fn() << endl;
        cout << "sem lbl is_leaf: " << sem_lbl_tensor.is_leaf() << endl;
        cout << "sem lbl is_cuda: " << sem_lbl_tensor.is_cuda() << endl;
        cout << "sem lbl is_sparse: " << sem_lbl_tensor.is_sparse() << endl;
        cout << "sem lbl is_contiguous: " << sem_lbl_tensor.is_contiguous() << endl;
        cout << "sem lbl numel: " << sem_lbl_tensor.numel() << endl;
        cout << "sem lbl ndimension: " << sem_lbl_tensor.ndimension() << endl;
        cout << "sem lbl element_size: " << sem_lbl_tensor.element_size() << endl << endl;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        ret = -1;
    }
    return ret;
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

    return can_run;
}

