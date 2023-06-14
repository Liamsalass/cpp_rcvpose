// test_storage.cpp : Defines the entry point for the application.
//

#include "test_storage.h"


#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include <torch/torch.h>

using namespace std;


void tensorToFile(const torch::Tensor& tensor, const std::string& filename) {
    std::ofstream outputFile(filename, std::ios::binary);
    if (!outputFile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Get the size and data pointer of the tensor
    torch::IntArrayRef size = tensor.sizes();
    const float* data = tensor.data_ptr<float>();

    // Write the size of the tensor to the file
    size_t numDimensions = size.size();
    outputFile.write(reinterpret_cast<const char*>(&numDimensions), sizeof(size_t));
    outputFile.write(reinterpret_cast<const char*>(size.data()), numDimensions * sizeof(int64_t));

    // Write the tensor data to the file
    size_t numElements = tensor.numel();
    outputFile.write(reinterpret_cast<const char*>(&numElements), sizeof(size_t));
    outputFile.write(reinterpret_cast<const char*>(data), numElements * sizeof(float));

    outputFile.close();

    // Print location of file
    std::cout << "Tensor data written to file: " << filename << std::endl;
}

int main() {
    // Create a sample tensor
    torch::Tensor tensor = torch::rand({ 3, 4, 2 });

    // Convert tensor to C datatype and write to file
    tensorToFile(tensor, "../../../data.bin");

    return 0;
}
