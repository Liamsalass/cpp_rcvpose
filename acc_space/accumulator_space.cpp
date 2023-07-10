#include "accumulator_space.h"


using namespace std;
using namespace open3d;
namespace e = Eigen;

typedef e::MatrixXd matrix;
typedef shared_ptr<geometry::PointCloud> pc_ptr;
typedef geometry::PointCloud pc;

const vector<double> mean = { 0.485, 0.456, 0.406 };
const vector<double> standard = { 0.229, 0.224, 0.225 };


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

void FCNResNetBackbone(DenseFCNResNet152 model, string input_path, torch::Tensor& radial_output, torch::Tensor& semantic_output, const torch::DeviceType device_type, const bool& debug) {

    torch::Device device(device_type);

    //if (debug) {
    //    cout << "Loading image from path: \n\t" << input_path << endl << endl;
    //}

    //Same transform func as dataloader
    cv::Mat img = cv::imread(input_path);
    img.convertTo(img, CV_32FC3);
    img /= 255.0;
    for (int i = 0; i < 3; i++) {
        cv::Mat channel(img.size(), CV_32FC1);
        cv::extractChannel(img, channel, i);
        channel = (channel - mean[i]) / standard[i];
        cv::insertChannel(channel, img, i);
    }
    if (img.rows % 2 != 0)
        img = img.rowRange(0, img.rows - 1);
    if (img.cols % 2 != 0)
        img = img.colRange(0, img.cols - 1);
    cv::Mat imgTransposed = img.t();
    torch::Tensor imgTensor = torch::from_blob(imgTransposed.data, { imgTransposed.rows, imgTransposed.cols, imgTransposed.channels() }, torch::kFloat32).clone();
    imgTensor = imgTensor.permute({ 2, 0, 1 });

    auto img_batch = torch::stack(imgTensor, 0).to(device);


    if (debug) {
        cout << "Image Tensor Shape: " << imgTensor.sizes() << endl;
    }

    auto output = model->forward(imgTensor);

    semantic_output = get<0>(output).to(torch::kCPU);
    radial_output = get<1>(output).to(torch::kCPU);


    if (debug) {
        cout << "Semantic Output Shape: " << semantic_output.sizes() << endl;
        cout << "Radial Output Shape: " << radial_output.sizes() << endl;
    }

}

void estimate_6d_pose_lm(const Options opts = testing_options(), bool debug = "true")
{
    cout << string(100, '=') << endl;
    cout << string(17, ' ') << "Estimating 6D Pose for LINEMOD" << endl;

    bool use_cuda = torch::cuda::is_available();
    if (use_cuda) {
        cout << "Using GPU" << endl;
    }
    else {
        cout << "Using CPU" << endl;
    }

    torch::DeviceType device_type = use_cuda ? torch::kCUDA : torch::kCPU;

    for (auto class_name : lm_cls_names) {
        cout << string(50, '-') << endl << string(17, ' ') << "Estimating for " << class_name << " object" << endl << endl;

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

        string pcv_load_path = "C:/Users/User/.cw/work/cpp_rcvpose/acc_space/python/pc_ape.npy";

        if (debug) {
			cout << "Loading point cloud from path: \n\t" << pcv_load_path << endl << endl;
		}

        pc_ptr pcv = read_point_cloud(pcv_load_path, debug);


        long long net_time = 0;
        long long acc_time = 0;
        int general_counter = 0;

        int bf_icp = 0;
        int af_icp = 0;
        
        vector<DenseFCNResNet152> model_list;

        cout << "Loading Models" << endl;
        for (int i = 1; i < 4; i++) {
            string model_dir = "kpt" + to_string(i);
            if (debug) {
                cout << "\t" << model_dir << endl;
            }
            DenseFCNResNet152 model(3, 2);
            CheckpointLoader loader(model_dir, true);
            model = loader.getModel();
            model->eval();
            model_list.push_back(model);
            if (debug) {
                break;
            }
        }

        vector<string> file_list;

        vector<Vertex> xyz_load;
        for (auto point : pcv->points_) {
            Vertex v;
            v.x = point.x();
            v.y = point.y();
            v.z = point.z();
            xyz_load.push_back(v);
        }

        if (debug) {
            cout << "XYZ data" << endl;
            for (int i = 0; i < 10; i++) {
                cout << "x: " << xyz_load[i].x << " y: " << xyz_load[i].y << " z: " << xyz_load[i].z << endl;
            }
            cout << endl;
        }

        string keypoints_path = opts.root_dataset + "/LINEMOD/" + class_name + "/Outside9.npy";

        if (debug) {
			cout << "Loading keypoints from path: \n\t" << keypoints_path << endl << endl;
		}

        vector<vector<double>> keypoints = read_key_points(keypoints_path, debug);
        
        string data_path = rootpvPath + "JPEGImages/";

        int img_num = 0;

        //Loop through all images in the img_path
        for (auto test_img : test_list){
            cout << string(100, '-') << endl;
            cout << string(15, ' ') << "Estimating for image " << test_img << ".jpg" << endl;

            string image_path = data_path + test_img + ".jpg";

            string RTGT_path = opts.root_dataset + "/LINEMOD/" + class_name + "/pose_txt/pose" + to_string(img_num) + ".txt";

            if (debug) {
                cout << "RTGT Path: \n\t" << RTGT_path << endl << endl;
            }

            img_num++;

            vector<vector<double>> RTGT = read_ground_truth(RTGT_path, true);

            if (debug) {
                cout << "RTGT data: " << endl;
                for (auto row : RTGT) {
                    for (auto item : row) {
                        cout << item << " ";
                    }
                    cout << endl;
                }
                cout << endl;

                               
            }

            int keypoint_count = 1;

            for (const auto& keypoint : keypoints) {
                cout << "Keypoint Count: " << keypoint_count << endl;

                //Ask Greenspan about this
                auto keypoint = keypoints[keypoint_count];

                if (debug) {
                    cout << "Keypoint data: \n" << keypoint << endl << endl;
                }
            
                int iteration_count = 0;
            
                vector<vector<double>> centers_list;

                //string GTRadiusPath = rootPath + "Out_pt" + to_string(keypoint_count) + "_dm

                auto keypoint_matrix = vectorToEigenMatrix(keypoint);
                auto RTGT_matrix = vectorOfVectorToEigenMatrix(RTGT);

                //transformed_gt_center_mm = (np.dot(keypoint, RTGT[:, :3].T) + RTGT[:, 3:].T)*1000

                auto transformed_gt_center_mm = transformKeyPoint(keypoint_matrix, RTGT_matrix, false);

                if (debug) {
                    cout << "Transformed Ground Truth Center: " << endl;
                    cout << transformed_gt_center_mm << endl;
                }

                torch::Tensor semantic_output;
                torch::Tensor radial_output;

                auto model = model_list.at(keypoint_count - 1);
                FCNResNetBackbone(model, image_path, semantic_output, radial_output, device_type, debug);
                
               
                if (debug) {
                    break;
                }
              
            }

            if (debug) {
                break;
            }
        }

        return; 
    }
}


#define proj_func false
#define rgbd_to_pc false

int main() {
    Options opts = testing_options();
    
    estimate_6d_pose_lm();

    return 0;

}





