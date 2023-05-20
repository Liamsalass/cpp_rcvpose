
// rcvpose.h - Used to create and train rcvpose models. Allows for testing and demoing of rcvpose models.
#pragma once

#ifdef RCVPOSE_EXPORTS
#define RCVPOSE_API __declspec(dllexport)
#else
#define RCVPOSE_API __declspec(dllimport)
#endif

#include <string>
#include <map>
#include <iostream>
#include <vector>
#include "utils.hpp"
#include <torch/torch.h>
#include "data_loader.h"
#include <warning.h>


//typedef std::unique_ptr<torch::data::StatelessDataLoader<RData, torch::data::samplers::RandomSampler>> TrainLoader;
//typedef std::unique_ptr<torch::data::StatelessDataLoader<RData, torch::data::samplers::SequentialSampler>> TestLoader;

#ifdef __cplusplus
extern "C" {
#endif

    // Parameters for rcvpose model
    // Includes fields:
    // mode: Mode of operation either "train" or "test"
    // gpu_id: GPU ID to use
    // dname: Dataset name ("lm" = LINEMOD, "ycbv", "tless")
    // root_dataset: Root dataset directory
    // resume_train: Resume training from a checkpoint
    // optim: Optimizer to use
    // batch_size: Batch size
    // class_name: Class name
    // initial_lr: Initial learning rate
    // kpt_num: Number of keypoints
    // model_dir: Directory to save model
    // demo_mode: run in Demo mode
    // test_occ: run in Test Occ mode
    struct RCVPOSE_API Options {
        //Mode of operation either "train" or "test"
        std::string mode = "train";
        // GPU ID to use
        int gpu_id = -1;
        // Dataset name ("lm" = LINEMOD, "ycbv", "tless")
        std::string dname = "lm";
        // Root dataset directory
        std::string root_dataset = "./datasets/LINEMOD";
        // Resume training from a checkpoint
        bool resume_train = false;
        // Optimizer to use
        std::string optim = "Adam";
        // Batch size
        int batch_size = 4;
        // Class name
        std::string class_name = "ape";
        // Initial learning rate
        double initial_lr = 1e-4;
        // Number of keypoints
        int kpt_num = 1;
        // Directory to save model
        std::string model_dir = "ckpts/";
        // run in Demo mode
        bool demo_mode = false;
        // run in Test Occ mode
        bool test_occ = false;
        // Configs
        std::map<std::string, std::vector<float>> cfg;
    };

    // rcvpose class
    // Used to create and train rcvpose models. Allows for testing and demoing of rcvpose models.
    // Example usage:
    // rcvpose rcvpose_model;
    class RCVPOSE_API RCVpose
    {
    public:
        // Constructor with passing options
        // Useful for storing a set of options and params
        RCVpose(Options options);

        // Constructor
        RCVpose(
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
        );

        // Default constructor
        RCVpose();

        // Default destructor
        ~RCVpose();


        void setGpuId(const int gpu_id);
        void setDname(const std::string& dname);
        void setRootDataset(const std::string& root_dataset);
        void setResumeTrain(bool resume_train);
        void setOptim(const std::string& optim);
        void setBatchSize(const int batch_size);
        void setClassName(const std::string& class_name);
        void setInitialLR(const double initial_lr);
        void setKptNum(const int kpt_num);
        void setModelDir(const std::string& model_dir);
        void setDemoMode(bool demo_mode);
        void setTestOcc(bool test_occ);

        // Prints a summary of the model
        void summary();

        //Begins the training or testing process depending on opts.mode value
        void start();

        // Evaluates the model on the test set
        void evaluate();

        // Saves the model to specified directory
        void saveModel(std::string path);

        // Tests on a single image and saves the output
        void test_img(std::string img_path, std::string output_path);

        //Tests if specified data is loadable
        void test_loaders(const std::string& path);

        void demo();

        // Loads a pretrained model
        void loadModel(std::string path);

    private:
        Options opts;
        void init();
        bool can_init();
        torch::DeviceType device_type;
        std::string resume;
        //TrainLoader train_loader;
        //TestLoader test_loader;
    };

#ifdef __cplusplus
}
#endif
#pragma once
