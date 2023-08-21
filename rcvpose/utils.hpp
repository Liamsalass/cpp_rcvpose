#pragma once

#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cstdint>
#include <torch/serialize.h>
#include <torch/torch.h>
#include "models/denseFCNResNet152.h"

struct Vertex {
    double x, y, z;
};

// Shape of config is {<int, <string, vector<float>>>} in a map
// Betas has two values, so it is a vector, while the rest are single values
inline std::map<int, std::map<std::string, std::vector<float>>> get_config() {
    return {
        {1,
         {
             {"max_iteration", {200000}},
             {"lr", {1e-4}},
             {"momentum", {0.99}},
             {"betas", {0.9, 0.999}},
             {"weight_decay", {0}},
             {"interval_validate", {1000}},
         }}
    };
}




class CheckpointLoader {
public:
    CheckpointLoader()
        : epoch_(c10::IValue(0)), iteration_(c10::IValue(0)), modelName_(c10::IValue("")),
        bestAccuracy_(c10::IValue(0.0)), loss_(c10::IValue(0.0)), optimizerName_(c10::IValue("")),
        lr_(c10::IValue(std::vector<double>{})),
        model_(), optim_(nullptr) {
        // Default initialization
    }

    CheckpointLoader(const std::string& checkpointPath, bool get_best_ckpt, bool load_model, bool load_optimizer) {
        if (get_best_ckpt) {
            path = checkpointPath + "/model_best";
        }
        else {
            path = checkpointPath + "/current";
        }
        // Load model info
        torch::serialize::InputArchive modelInfoArchive;
        modelInfoArchive.load_from(path + "/info.pt");
        modelInfoArchive.read("epoch", epoch_);
        modelInfoArchive.read("iteration", iteration_);
        modelInfoArchive.read("arch", modelName_);
        modelInfoArchive.read("best_acc_mean", bestAccuracy_);
        modelInfoArchive.read("loss", loss_);
        modelInfoArchive.read("optimizer", optimizerName_);
        modelInfoArchive.read("lr", lr_);
        // Load model
        if (load_model) {
            torch::serialize::InputArchive modelArchive;
            modelArchive.load_from(path + "/model.pt");
            model_->load(modelArchive);
        }
		else {
			model_ = nullptr;
		}
        optim_ = nullptr;

  
    }


    int getEpoch() const {
        return epoch_.toInt();
    }

    int getIteration() const {
        return iteration_.toInt();
    }

    std::string getModelName() const {
        return modelName_.toStringRef();
    }

    float getBestAccuracy() const {
        return bestAccuracy_.toDouble();
    }

    float getLoss() const {
        return loss_.toDouble();
    }

    DenseFCNResNet152& getModel() {
        if (!model_) {
			throw std::runtime_error("Model was not loaded");
		}
        return model_;
    }

    torch::optim::Optimizer* getOptimizer() {
        if (!model_) {
            throw std::runtime_error("Model was not loaded");
        }
        torch::serialize::InputArchive optimArchive;
        optimArchive.load_from(path + "/optim.pt");
        //TODO, load current LR values and store them
        optim_ = new torch::optim::Adam(model_->parameters(), torch::optim::AdamOptions(0.0001));
        optim_->load(optimArchive);

        if (!optim_) {
            throw std::runtime_error("Optimizer was not loaded");
        }
        return optim_;
    }
    
    std::string getOptimizerName() const {
        return optimizerName_.toStringRef();
    }

    std::vector<double> getLrList() const {
        return lr_.toDoubleVector();
    }


private:
    c10::IValue epoch_, iteration_, modelName_, bestAccuracy_, loss_, optimizerName_, lr_;
    DenseFCNResNet152 model_;
    torch::optim::Optimizer* optim_;
    std::string path;
};


inline void printProgressBar(int current, int total, int width)
{
    float progress = float(current) / float(total);
    int barWidth = width - 7;
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}




inline void saveTensorToFile(const torch::Tensor& tensor, const std::string& filename) {
    // Extract the directory path from the full filename
    std::size_t found = filename.find_last_of("/\\");
    std::string directory = filename.substr(0, found);

    // Create the directory if it doesn't exist
    if (!std::filesystem::is_directory(directory)) {
        if (!std::filesystem::create_directories(directory)) {
            std::cerr << "Error creating directory." << std::endl;
            return;
        }
    }

    std::ofstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }

    torch::IntArrayRef sizes = tensor.sizes();
    int64_t num_dimensions = sizes.size();
    file.write(reinterpret_cast<const char*>(&num_dimensions), sizeof(int64_t));
    file.write(reinterpret_cast<const char*>(sizes.data()), num_dimensions * sizeof(int64_t));

    int64_t num_elements = tensor.numel();
    file.write(reinterpret_cast<const char*>(&num_elements), sizeof(int64_t));

    std::vector<float> tensor_data(num_elements);
    auto tensor_data_ptr = tensor.data_ptr<float>();
    for (int64_t i = 0; i < num_elements; ++i) {
        tensor_data[i] = tensor_data_ptr[i];
    }
    file.write(reinterpret_cast<const char*>(tensor_data.data()), num_elements * sizeof(float));

    file.close();
}

