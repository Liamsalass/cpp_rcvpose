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

		//Testing example file
		auto example = py::module::import("scripts.example");
		auto func = example.attr("add");
		int sum = func(1, 2).cast<int>();
		cout << "sum: " << sum << endl;

		//Testing radius file
		auto radius3d_lm = py::module::import("scripts.3DRadius_lm");
		//auto exe_radiuslm = radius3d_lm.attr("exe_radiuslm");
		//exe_radiuslm();
		cout << "Test end" << endl;
	}
	catch (py::error_already_set& e) {
		std::cout << e.what() << std::endl;
	}
}