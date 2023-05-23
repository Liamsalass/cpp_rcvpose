
// rcvpose.h - Used to create and train rcvpose models. Allows for testing and demoing of rcvpose models.
#pragma once

#ifdef _WIN32
#ifdef RCVPOSE_EXPORTS
#define RCVPOSE_API __declspec(dllexport)
#else
#define RCVPOSE_API __declspec(dllimport)
#endif
#else
#define RCVPOSE_API
#endif


#include <string>
#include <map>
#include <iostream>
#include <vector>
#include "utils.hpp"
#include <torch/torch.h>
#include "data_loader.h"
#include <warning.h>
#include "options.hpp"
#include "trainer.h"


#ifdef __cplusplus
extern "C" {
#endif

    // rcvpose class
    // Used to create and train rcvpose models. Allows for testing and demoing of rcvpose models.
    // Example usage:
    // rcvpose rcvpose_model(string mode = "train", int gpu_id = -1, string dname = "lm", string root_dataset = "./datasets/LINEMOD", bool resume_train = false, string optim = "Adam", int batch_size = 4, string class_name = "ape", double initial_lr = 1e-4, int kpt_num = 1, string model_dir = "ckpts/", bool demo_mode = false, bool test_occ = false);
    class RCVPOSE_API RCVpose
    {
    public:
        RCVpose(Options options);

        // Constructor
        RCVpose(
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

        void train();

        //Begins the training or testing process depending on opts.mode value
        void test();

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
