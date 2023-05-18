#include "pch.h"
#include "rcvpose.h"

using namespace std;

RCVpose::RCVpose(Options options) : opts(options) {
    cout << "Constructor called" << endl;
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
    float initial_lr,
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
    cout << "Constructor called" << endl;
}

RCVpose::RCVpose() {
    cout << "Empty constructor called" << endl;
}

RCVpose::~RCVpose() {}

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

void RCVpose::setInitialLR(const float initial_lr) {
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
    // Implementation for printing model summary
}

void RCVpose::train() {
    // Implementation for training the model
}

void RCVpose::evaluate() {
    // Implementation for evaluating the model
}

void RCVpose::saveModel(std::string path) {
    // Implementation for saving the model
}

void RCVpose::test(std::string img_path, std::string output_path) {
    // Implementation for testing the model on a single image and saving the output
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