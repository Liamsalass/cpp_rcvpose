#include "accumulator_space.h"


using namespace std;
using namespace open3d;
namespace e = Eigen;

typedef e::MatrixXd matrix;
typedef shared_ptr<geometry::PointCloud> pc_ptr;
typedef geometry::PointCloud pc;

Options testing_options() {
    Options opts;
    opts.gpu_id = 0;
    opts.dname = "lm";
    opts.root_dataset = "C:/Users/User/.cw/work/datasets/test";
    //or ".../dataset/public/RCVLab/Bluewrist/16yw11"
    opts.model_dir = "train_kpt2";
    opts.resume_train = false;
    opts.optim = "adam";
    opts.batch_size = 2;
    opts.class_name = "ape";
    opts.initial_lr = 0.0001;
    opts.reduce_on_plateau = false;
    opts.kpt_num = 2;
    opts.demo_mode = false;
    opts.test_occ = false;
    return opts;
}


//vector<string> lm_cls_names = {"ape", "cam", "can", "cat", "driller", "duck", "eggbox", "glue", "holepuncher", "iron", "lamp"};
vector<string> lm_cls_names = { "ape" };

unordered_map<std::string, double> add_threshold = {
    {"eggbox", 0.019735770122546523},
    {"ape", 0.01421240983190395},
    {"cat", 0.018594838977253875},
    {"cam", 0.02222763033276377},
    {"duck", 0.015569664208967385},
    {"glue", 0.01930723067998101},
    {"can", 0.028415044264086586},
    {"driller", 0.031877906042},
    {"holepuncher", 0.019606109985}
};

array<array<double, 3>, 3> linemod_K {{
    {572.4114, 0.0, 325.2611},
    { 0.0, 573.57043, 242.04899 },
    { 0.0, 0.0, 1.0 }
    }};


matrix cvmat_to_eigen(const cv::Mat& mat)
{
    int width = mat.cols;
    int height = mat.rows;
    int channels = mat.channels();

    int type = mat.type();
    int dataType = type & CV_MAT_DEPTH_MASK;

    matrix eigenMat;
    if (dataType == CV_8U)
    {
        cout << "Data type: CV_8U" << endl;
        eigenMat.resize(height, width * channels);
        #pragma omp parallel for collapse(3)
        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                for (int c = 0; c < channels; ++c)
                {
                    eigenMat(row, col * channels + c) = static_cast<double>(mat.at<cv::Vec3b>(row, col)[c]);
                }
            }
        }
    }
    else if (dataType == CV_32F)
    {
        cout << "Data type: CV_32F" << endl;
        eigenMat.resize(height, width * channels);
        #pragma omp parallel for collapse(3)
        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                for (int c = 0; c < channels; ++c)
                {
                    eigenMat(row, col * channels + c) = static_cast<double>(mat.at<cv::Vec3f>(row, col)[c]);
                }
            }
        }
    }
    else if (dataType == CV_64F)
    {
        cout << "Data type: CV_64F" << endl;
        eigenMat.resize(height, width * channels);
        #pragma omp parallel for collapse(3)
        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                for (int c = 0; c < channels; ++c)
                {
                    eigenMat(row, col * channels + c) = mat.at<cv::Vec3d>(row, col)[c];
                }
            }
        }
    }
    else
    {
        cout << "Error: Datatype unknown.\n Data type: " << dataType << endl;
        return eigenMat;
    }

    return eigenMat;
}



void project(const matrix& xyz, const matrix& K, const matrix& RT, matrix& xy, matrix& actual_xyz) {
    // xyz: [N, 3]
    // K: [3, 3]
    // RT: [3, 4]
    matrix xyz_rt = (xyz * RT.block(0, 0, 3, 3).transpose()) + RT.block(0, 3, 3, 1).transpose().replicate(4, 1);
    actual_xyz = xyz_rt;
    matrix xyz_k = xyz_rt * K.transpose();
    xy = xyz_k.leftCols(2).array() / xyz_k.col(2).array().replicate(1, 2);
}


pc_ptr rgbd_to_point_cloud(const matrix& K, const matrix& depth) {
    pc point_cloud;
    
    e::Index heigh = depth.rows();
    e::Index width = depth.cols();

    pc_ptr cloud(new geometry::PointCloud);

    for (e::Index i = 0; i < heigh; i++) {
        for (e::Index j = 0; j < width; j++) {
			double z = depth(i, j);
            if (z > 0) {
				double x = (j - K(0, 2)) * z / K(0, 0);
				double y = (i - K(1, 2)) * z / K(1, 1);
				point_cloud.points_.push_back(e::Vector3d(x, y, z));
			}
		}
	}

    cloud->points_ = std::move(point_cloud.points_);
    return cloud;
}


void estimate_6d_pose_lm(const Options opts)
{
    for (auto class_name : lm_cls_names) {
        string rootPath = opts.root_dataset + "/LINEMOD_ORIG/" + class_name + "/";
        string rootpvPath = opts.root_dataset + "/LINEMOD/" + class_name + "/";

        ifstream test_file(opts.root_dataset + "/LINEMOD/" + class_name + "/Split/val.txt");
        vector<string> test_list;
        string line;

        if (test_file.is_open()) {
            while (getline(test_file, line)) {
                line.erase(line.find_last_not_of("\n\r\t") + 1);
                test_list.push_back(line);
            }
            test_file.close();
        }

        string pcv_load_path = opts.root_dataset + "/LINEMOD/" + class_name + "/" + class_name + ".ply";

        cout << pcv_load_path << endl;

        pc_ptr pcv(new geometry::PointCloud);

        //try {
        //    cout << "File format " << open3d::utility::filesystem::GetFileExtensionInLowerCase(pcv_load_path) << endl;
        //}
        //catch (const std::exception& e) {
		//	cout << "Error: " << e.what() << endl;
		//}


        if (!io::ReadPointCloud(pcv_load_path, *pcv, io::ReadPointCloudOption("auto", false, false, false))) {
            if (!io::ReadPointCloud(pcv_load_path, *pcv, io::ReadPointCloudOption("auto", false, false, false))) {
                cout << "Error: cannot load point cloud" << endl;
            }
        }
        // "C:\Users\User\.cw\work\datasets\test\LINEMOD\ape\ape.ply"
        pcv = io::CreatePointCloudFromFile("C:/Users/User/.cw/work/datasets/test/LINEMOD/ape/ape.ply", "auto");
        pcv = io::CreatePointCloudFromFile("C:\Users\User\.cw\work\datasets\test\LINEMOD\ape\ape.ply", "auto");


        cout << "Number of points " << pcv->points_.size() << endl;

        for (int i = 0; i < 5 && i < pcv->points_.size(); ++i) {
            const Eigen::Vector3d& point = pcv->points_[i];
            cout << "Point " << i + 1 << ": " << point.transpose() << endl;
        }

        return; 
    }
}


#define proj_func false
#define rgbd_to_pc false

int main() {
    cout << "Testing acc space" << endl;

    if (proj_func) {
        e::MatrixXd xyz(4, 3); 
        xyz << 1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0;

        cout << "xyz\n" << xyz << endl;

        e::MatrixXd K(3, 3);  
        K << 1000.0, 0.0, 500.0,
            0.0, 1000.0, 300.0,
            0.0, 0.0, 1.0;

        cout << "K\n" << K << endl;

        e::MatrixXd RT(3, 4);
        RT << 2.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 5.0,
            0.0, 0.0, 2.0, 0.0;

        cout << "RT\n" << RT << endl << endl;

        matrix xy, actual_xyz;

        project(xyz, K, RT, xy, actual_xyz);

        cout << "Projected xy coordinates:\n" << xy << endl;
        cout << "Actual XYZ:\n" << actual_xyz << endl;
    }
    if (rgbd_to_pc) {

        matrix K(3, 3);
        K << 572.4114, 0.0, 325.2611,
			0.0, 573.57043, 242.04899,
			0.0, 0.0, 1.0;

        cv::Mat image = cv::imread("C:/Users/User/.cw/work/cpp_rcvpose/acc_space/images/000000.jpg", cv::IMREAD_UNCHANGED);
        if (image.empty()){
			cout << "Could not open or find the image" << endl;
			return -1;
        }

        cout << "Image matrix size: " << image.size() << endl;

        matrix image_matrix = cvmat_to_eigen(image);

        cout << "Eigen matrix size: " << image_matrix.rows() << "x" << image_matrix.cols() << endl;

        pc_ptr point_cloud = rgbd_to_point_cloud(K, image_matrix);
        
        cout << "Number of points: " << point_cloud->points_.size() << endl;
        
       

    }

    Options opts = testing_options();

    estimate_6d_pose_lm(opts);

    return 0;

}





