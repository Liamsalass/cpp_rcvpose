#pragma once

#include <map>
#include <string>
#include <iostream>
#include <vector>
#include <torch/serialize.h>
#include <torch/torch.h>
#include "models/denseFCNResNet152.h"


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
    CheckpointLoader(const std::string& checkpointPath, bool get_best_ckpt = false) {
        std::string path;
        if (get_best_ckpt) {
            path = checkpointPath + "/model_best";
            std::cout << "Resuming from best checkpoint" << std::endl;
        }
        else {
            path = checkpointPath + "/current";
            std::cout << "Resuming from current checkpoint" << std::endl;
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
        torch::serialize::InputArchive modelArchive;
        modelArchive.load_from(path + "/model.pt");
        model_->load(modelArchive);

        // Load optimizer
        torch::serialize::InputArchive optimArchive;
        optimArchive.load_from(path + "/optim.pt");
        //TODO, load current LR values and store them
        optim_ = new torch::optim::Adam(model_->parameters(), torch::optim::AdamOptions(0.0001));
        optim_->load(optimArchive);
    }

    // Getter methods to retrieve loaded information
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

    DenseFCNResNet152 getModel() const {
        return model_;
    }

    torch::optim::Optimizer* getOptimizer() const {
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
};

