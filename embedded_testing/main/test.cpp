#include <iostream>

#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <open3d/geometry/PointCloud.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;
using namespace std;

int main(int argc, char* argv[])
{
	try {
		py::scoped_interpreter guard{};
		auto radius3d_lm = py::module::import("scripts.3DRadius_lm");
		auto exe_radiuslm = radius3d_lm.attr("exe_radiuslm");
		exe_radiuslm();
		cout << "Test completed" << endl;
	}
	catch (const std::exception& e) {
		std::cout << e.what() << std::endl;
	}
}