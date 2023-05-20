
#include "rcvpose.h"

using namespace std;



RCVpose::RCVpose(Options options) : opts(options) {
    init();
}

RCVpose::RCVpose(
    std::string mode,
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
    opts.mode = mode;
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

void RCVpose::setGpuId(const int gpu_id) {
    opts.gpu_id = gpu_id;
}

void RCVpose::setDname(const std::string& dname) {
    opts.dname = dname;
}

void RCVpose::setRootDataset(const std::string& root_dataset) {
    opts.root_dataset = root_dataset;
}

void RCVpose::setResumeTrain(bool resume_train) {
    opts.resume_train = resume_train;
}

void RCVpose::setOptim(const std::string& optim) {
    opts.optim = optim;
}

void RCVpose::setBatchSize(const int batch_size) {
    opts.batch_size = batch_size;
}

void RCVpose::setClassName(const std::string& class_name) {
    opts.class_name = class_name;
}

void RCVpose::setInitialLR(const double initial_lr) {
    opts.initial_lr = initial_lr;
}

void RCVpose::setKptNum(const int kpt_num) {
    opts.kpt_num = kpt_num;
}

void RCVpose::setModelDir(const std::string& model_dir) {
    opts.model_dir = model_dir;
}

void RCVpose::setDemoMode(bool demo_mode) {
    opts.demo_mode = demo_mode;
}

void RCVpose::setTestOcc(bool test_occ) {
    opts.test_occ = test_occ;
}


void RCVpose::summary() {
    // Print a line of '=' with "Summary" in the middle
    cout << endl;
    cout << string(50, '=') << endl;
    cout << string(17, ' ') << "Summary" << endl;
    cout << string(50, '=') << endl;

    // Print the options
    cout << "mode: " << opts.mode << endl;
    cout << "gpu_id: " << opts.gpu_id << endl;
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
    cout << string(50, '=') << endl;
    cout << endl;
}

void RCVpose::start()
{
}


void RCVpose::evaluate() {
    // Implementation for evaluating the model
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

void RCVpose::test_loaders(const std::string& path)
{
}

void RCVpose::init() {
    // Initialization of the model
    // check if opts has all passed parameters, if not return error message with what needs to be initialized
    bool can_run = can_init();

    opts.cfg = get_config().at(1);

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

    if (opts.mode == "train") {
        cout << "Train mode" << endl;
    }
    else
        cout << "Test mode" << endl;

    //Set the loaders
    std::pair<std::unique_ptr<torch::data::StatelessDataLoader<RData, torch::data::samplers::RandomSampler>>,
        std::unique_ptr<torch::data::StatelessDataLoader<RData, torch::data::samplers::SequentialSampler>>
    > data_loaders =
        get_data_loaders(opts);

    TrainLoader train_loader = std::move(data_loaders.first);
    TestLoader test_loader = std::move(data_loaders.second);




}

bool RCVpose::can_init() {
    if (opts.mode.empty()) {
        cout << "Error: mode not initialized" << endl;
        return false;
    }
    if (opts.gpu_id) {
        cout << "Error: gpu_id not initialized" << endl;
        return false;
    }
    if (opts.dname.empty()) {
        cout << "Error: dname not initialized" << endl;
        return false;
    }
    if (opts.root_dataset.empty()) {
        cout << "Error: root_dataset not initialized" << endl;
        return false;
    }
    if (opts.optim.empty()) {
        cout << "Error: optim not initialized" << endl;
        return false;
    }
    if (opts.batch_size) {
        cout << "Error: batch_size not initialized" << endl;
        return false;
    }
    if (opts.class_name.empty()) {
        cout << "Error: class_name not initialized" << endl;
        return false;
    }
    if (opts.initial_lr) {
        cout << "Error: initial_lr not initialized" << endl;
        return false;
    }
    if (opts.kpt_num) {
        cout << "Error: kpt_num not initialized" << endl;
        return false;
    }
    if (opts.model_dir.empty()) {
        cout << "Error: model_dir not initialized" << endl;
        return false;
    }
    return true;
}