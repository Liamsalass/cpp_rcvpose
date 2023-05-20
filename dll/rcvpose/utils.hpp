#pragma once

#include <map>
// Shape of config is {<int, <string, vector<float>>>} in a map
// Betas has two values, so it is a vector, while the rest are single values
std::map<int, std::map<std::string, std::vector<float>>> get_config() {
    return {
        {1,
         {
             {"max_iteration", {700000}},
             {"lr", {1e-4}},
             {"momentum", {0.99}},
             {"betas", {0.9, 0.999}},
             {"weight_decay", {0}},
             {"interval_validate", {1000}},
         }}
    };
}
