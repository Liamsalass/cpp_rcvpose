#include "pch.h"
#include "rcvpose.h"

#include "rcvpose.h"

rcvpose::rcvpose(Options options) : opts(options) {}

rcvpose::rcvpose(
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
}

rcvpose::rcvpose() {}

rcvpose::~rcvpose() {}

void rcvpose::setGpuId(const int gpu_id) {
    opts.gpu_id = gpu_id;
}

void rcvpose::setDname(const std::string& dname) {
    opts.dname = dname;
}

void rcvpose::setRootDataset(const std::string& root_dataset) {
    opts.root_dataset = root_dataset;
}

void rcvpose::setResumeTrain(bool resume_train) {
    opts.resume_train = resume_train;
}

void rcvpose::setOptim(const std::string& optim) {
    opts.optim = optim;
}

void rcvpose::setBatchSize(const int batch_size) {
    opts.batch_size = batch_size;
}

void rcvpose::setClassName(const std::string& class_name) {
    opts.class_name = class_name;
}

void rcvpose::setInitialLR(const float initial_lr) {
    opts.initial_lr = initial_lr;
}

void rcvpose::setKptNum(const int kpt_num) {
    opts.kpt_num = kpt_num;
}

void rcvpose::setModelDir(const std::string& model_dir) {
    opts.model_dir = model_dir;
}

void rcvpose::setDemoMode(bool demo_mode) {
    opts.demo_mode = demo_mode;
}

void rcvpose::setTestOcc(bool test_occ) {
    opts.test_occ = test_occ;
}



void rcvpose::summary() {
    // Implementation for printing model summary
}

void rcvpose::train() {
    // Implementation for training the model
}

void rcvpose::evaluate() {
    // Implementation for evaluating the model
}

void rcvpose::saveModel(std::string path) {
    // Implementation for saving the model
}

void rcvpose::test(std::string img_path, std::string output_path) {
    // Implementation for testing the model on a single image and saving the output
}

void rcvpose::demo() {
    // Implementation for running the model in demo mode
}

void rcvpose::loadModel(std::string path) {
    // Implementation for loading a pretrained model
}

void rcvpose::test_loaders(const std::string& path)
{
}