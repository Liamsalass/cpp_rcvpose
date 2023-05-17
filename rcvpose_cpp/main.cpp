#include <iostream>
#include <string>
#include "train.h"
#include "options.hpp"
#include "data_loader.h"
#include <torch/torch.h>
#include "utils.hpp"
#include <warning.h>
#include <boost/program_options.hpp>


using namespace std;
namespace po = boost::program_options;

typedef std::unique_ptr<torch::data::StatelessDataLoader<RData, torch::data::samplers::RandomSampler>> TrainLoader;
typedef std::unique_ptr<torch::data::StatelessDataLoader<RData, torch::data::samplers::SequentialSampler>> TestLoader;

// Main code
int main(int argc, char* args[])
{
    //Set random seed
	string resume = "";
	torch::manual_seed(0);

    Options opts;

    //Parse command line arguments
	po::options_description p_desc("Program Descritopn");
    p_desc.add_options()
        ("mode,m", po::value<std::string>()->default_value("train"), "Mode of operation")
        ("gpu_id,g", po::value<int>()->default_value(-1), "ID of the GPU to use")
        ("dname,d", po::value<std::string>()->default_value("lm"), "Dataset name")
        ("root_dataset,r", po::value<std::string>()->default_value("./datasets/LINEMOD"), "Root directory of the dataset")
        ("resume_train,rt", po::value<bool>()->default_value(false), "Resume training from a checkpoint")
        ("optim,o", po::value<std::string>()->default_value("Adam"), "Optimizer to use")
        ("batch_size,bs", po::value<int>()->default_value(4), "Batch size")
        ("class_name,cn", po::value<std::string>()->default_value("ape"), "Name of the class")
        ("initial_lr,lr", po::value<float>()->default_value(1e-4), "Initial learning rate")
        ("kpt_num,kn", po::value<int>()->default_value(1), "Number of keypoints")
        ("model_dir,md", po::value<std::string>()->default_value("ckpts/"), "Directory for saving model checkpoints")
        ("demo_mode,dm", po::value<bool>()->default_value(false), "Enable demo mode")
        ("test_occ,to", po::value<bool>()->default_value(false), "Perform occlusion testing");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, args, p_desc), vm);
    po::notify(vm);


    // Set up options struct
    opts.mode = vm["mode"].as<std::string>();
    opts.gpu_id = vm["gpu_id"].as<int>();
    opts.dname = vm["dname"].as<std::string>();
    opts.root_dataset = vm["root_dataset"].as<std::string>();
    opts.resume_train = vm["resume_train"].as<bool>();
    opts.optim = vm["optim"].as<std::string>();
    opts.batch_size = vm["batch_size"].as<int>();
    opts.class_name = vm["class_name"].as<std::string>();
    opts.initial_lr = vm["initial_lr"].as<float>();
    opts.kpt_num = vm["kpt_num"].as<int>();
    opts.model_dir = vm["model_dir"].as<std::string>();
    opts.demo_mode = vm["demo_mode"].as<bool>();
    opts.test_occ = vm["test_occ"].as<bool>();

    // Set up config, cfg is a map of shape std::string, std::vector<float>
    opts.cfg = get_config().at(1);


    //Set device type
    torch::DeviceType device_type;

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

    std::pair<
        std::unique_ptr<torch::data::StatelessDataLoader<RData, torch::data::samplers::RandomSampler>>,
        std::unique_ptr<torch::data::StatelessDataLoader<RData, torch::data::samplers::SequentialSampler>>
    > data_loaders = 
        get_data_loaders(opts);
    
    TrainLoader train_loader = std::move(data_loaders.first);
    TestLoader test_loader = std::move(data_loaders.second);

    Trainer trainer(train_loader, test_loader, opts);
	
    if (opts.mode == "test") {
		trainer.test();
	}
    else {
		trainer.train();
	}
	
}
