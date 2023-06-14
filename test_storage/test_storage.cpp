// test_storage.cpp : Defines the entry point for the application.
//

#include "test_storage.h"
#include <torch/torch.h>
using namespace std;

int main()
{
	// Create random torch tensor
	torch::Tensor tensor = torch::rand({ 2, 3 });
	cout << tensor << endl;
	//store the tensor data to specified path
	torch::save(tensor, "../../../../tensor.pt");

}
