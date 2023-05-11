#include <iostream>

#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <open3d/geometry/PointCloud.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;

int main(int argc, char* argv[])
{
	std::cout << "[cpp] Hello World!" << std::endl;
	py::scoped_interpreter guard{};
	py::exec(R"(
		print("[py] Hello World!")
	)");

}