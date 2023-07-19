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




void FCResBackbone(DenseFCNResNet152& model, const string& input_path, torch::Tensor& radial_output, torch::Tensor& semantic_output, const torch::DeviceType device_type, const bool& debug) {

    torch::Device device(device_type);

    if (debug) {
        cout << "Loading image from path: \n\t" << input_path << endl << endl;
    }

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

    auto output = model->forward(img_batch);

    auto sem_out = get<0>(output).to(torch::kCPU);
    auto rad_out = get<1>(output).to(torch::kCPU);

    auto sem_out_vec = torch::unbind(sem_out, 0);
    auto rad_out_vec = torch::unbind(rad_out, 0);

    semantic_output = sem_out_vec[0];
    radial_output = rad_out_vec[0];

    semantic_output = semantic_output.permute({ 1,2,0 });
    radial_output = radial_output.permute({ 1,2,0 });

    //semantic_output = semantic_output.squeeze(2);
    //radial_output = radial_output.squeeze(2);
}

void estimate_6d_pose_lm(const Options opts = testing_options(), bool debug = "true")
{
    cout << string(75, '=') << endl;
    cout << string(17, ' ') << "Estimating 6D Pose for LINEMOD" << endl;

    bool use_cuda = torch::cuda::is_available();
    if (use_cuda) {
        cout << "Using GPU" << endl;
    }
    else {
        cout << "Using CPU" << endl;
    }

    if (debug) {
        cout << endl;
        cout << "\t\t\tDebug Mode" << endl;
        cout << "\t\t\t\b------------" << endl << endl;
    }

    torch::DeviceType device_type = use_cuda ? torch::kCUDA : torch::kCPU;
    torch::Device device(device_type);

    for (auto class_name : lm_cls_names) {
        cout << string(75, '-') << endl << string(17, ' ') << "Estimating for " << class_name << " object" << endl << endl;

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
			cout << "Loading point cloud from path: \n\t" << pcv_load_path << endl;
		}

        pc_ptr pcv = read_point_cloud(pcv_load_path, debug);


        long long net_time = 0;
        long long acc_time = 0;
        int general_counter = 0;

        int bf_icp = 0;
        int af_icp = 0;
        
        vector<DenseFCNResNet152> model_list;

        cout << endl << "Loading Models" << endl;
        for (int i = 1; i < 4; i++) {
         
            string model_dir = "kpt" + to_string(i);
            if (debug) {
                cout << "\t" << model_dir << endl;
            }

            DenseFCNResNet152 model;
            CheckpointLoader loader(model_dir, true);

            model = loader.getModel();
            model->eval();
            model->to(device);

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
            cout << "XYZ data (top 5)" << endl;
            for (int i = 0; i < 5; i++) {
                cout << "\tx: " << xyz_load[i].x << " y: " << xyz_load[i].y << " z: " << xyz_load[i].z << endl;
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
            cout << string(75, '-') << endl;
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

                matrix keypoint_matrix = vectorToEigenMatrix(keypoint);
                matrix RTGT_matrix = vectorOfVectorToEigenMatrix(RTGT);

                //transformed_gt_center_mm = (np.dot(keypoint, RTGT[:, :3].T) + RTGT[:, 3:].T)*1000

                auto transformed_gt_center_mm = transformKeyPoint(keypoint_matrix, RTGT_matrix, false);

                if (debug) {
                    cout << "Transformed Ground Truth Center: " << endl;
                    cout << transformed_gt_center_mm << endl << endl;
                }

                torch::Tensor semantic_output;
                torch::Tensor radial_output;

                auto model = model_list.at(keypoint_count - 1);

                auto start = chrono::high_resolution_clock::now();

                FCResBackbone(model, image_path, radial_output, semantic_output, device_type, debug);

                auto end = chrono::high_resolution_clock::now();
                
                if (debug) {
                    cout << "FCResBackbone Speed: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
                    cout << "Semantic Output Shape: " << semantic_output.sizes() << endl;
                    cout << "Radial Output Shape: " << radial_output.sizes() << endl << endl;;
                }

                net_time += chrono::duration_cast<chrono::milliseconds>(end - start).count();

                cv::Mat sem_cv = torch_tensor_to_cv_mat(semantic_output);
                cv::Mat rad_cv = torch_tensor_to_cv_mat(radial_output);

                string depth_path = rootPath + "data/depth" + to_string(img_num) + ".dpt";

                cv::Mat depth_cv = read_depth_to_cv(depth_path, debug);
      
                if (debug) {
                    cout << endl << "Data Shapes: " << endl;
                    cout << "\tSemantic Shape: " << sem_cv.size() << endl;
                    cout << "\tSemantic Datatype: " << sem_cv.type() << endl;
                    cout << "\tRadial Shape: " << rad_cv.size() << endl;
                    cout << "\tRadial Datatype: " << rad_cv.type() << endl;
                    cout << "\tDepth Shape: " << depth_cv.size() << endl;
                    cout << "\tDepth Datatype: " << depth_cv.type() << endl << endl;
                }

                cv::transpose(sem_cv, sem_cv);
                cv::transpose(rad_cv, rad_cv);

                cv::Mat thresholded;
                cv::threshold(sem_cv, thresholded, 0.8, 1, cv::THRESH_BINARY);
                thresholded.convertTo(sem_cv, sem_cv.type());
                thresholded.release();


                //TODO: Change implementation to ensure that the depth values don't go to all 1
                //cv::Mat result;
                //cv::multiply(depth_cv, sem_cv, result, 1.0, depth_cv.type());
                //cv::divide(result, 1000, depth_cv);
                //result.release();
                cv::Mat sem_tmp, depth_tmp, rad_tmp;

                sem_cv.convertTo(sem_tmp, CV_32F);
                depth_cv.convertTo(depth_tmp, CV_32F);
                rad_cv.convertTo(rad_tmp, CV_32F);

                rad_tmp = rad_tmp.mul(sem_cv);
                depth_tmp = depth_tmp.mul(sem_tmp);
                depth_tmp = depth_tmp / 1000.0;
                depth_cv = depth_tmp.clone();
                rad_cv = rad_tmp.clone();

                rad_tmp.release();
                depth_tmp.release();
                sem_tmp.release();


                vector<Vertex> pixel_coor;
                if (debug) {
                    cout << "Gathering Pixel Coordinates from semantic output" << endl;
                }

                #pragma omp parallel for collapse(2)
                for (int i = 0; i < sem_cv.rows; i++) {
                    for (int j = 0; j < sem_cv.cols; j++) {
                        if (sem_cv.at<float>(i, j) == 1) {
                            Vertex v;
                            v.x = i;
                            v.y = j;
                            v.z = 1;
                            #pragma omp critical
							pixel_coor.push_back(v);
						}
					}
				}

                if (debug) {
                    cout << "\tPixel Coord Size: " << pixel_coor.size() << endl;
                }

                vector<double> radial_list;

                #pragma omp parallel for 
                for (auto cord : pixel_coor) {
                    Vertex v;
                    v.x = cord.x;
                    v.y = cord.y;
                    v.z = rad_cv.at<float>(cord.x, cord.y);
                    #pragma omp critical
                    radial_list.push_back(v.z);
                }

                if (debug) {
                    cout << "\tRadial List Size: " << radial_list.size() << endl << endl;
                    cout << "Converting depth image to pointcloud" << endl;
                }

                //Check imkplementation since values are slightly off from python (Most likely depth that is different causing different output)
                auto xyz = rgbd_to_point_cloud(linemod_K, depth_cv);


                if (debug) {
					cout << "\tPointcloud Size: " << xyz.points_.size() << endl << endl;
				}

                Eigen::MatrixXd dump, xyz_load_transformed;

                if (debug) {
                    cout << "Converting XYZ_Load and Linemod_K to matrix" << endl << endl;
                }

                matrix xyz_load_matrix = convertToEigenMatrix(xyz_load);
                matrix linemod_K_matrix = convertToEigenMatrix(linemod_K);


                if (debug) {
                    cout << "Projecting pointcloud to image plane" << endl;
                }
      

                project(xyz_load_matrix, linemod_K_matrix, RTGT_matrix, dump, xyz_load_transformed);

                if (debug) {
                    cout << "\tTransformed Pointcloud Size: " << xyz_load_transformed.rows() << endl;
                    cout << "\tTransformed Pointcloud 5 points: " << endl;
                    for (int i = 0; i < 5; i++) {
						cout << "\t\t" << xyz_load_transformed.row(i) << endl;

					}
                    cout << endl;
                }


                if (debug) {
                    cout << "Calculating 3D vector center (Accumulator_3D)" << endl;
                    cout << "Function inputs: " << endl;
                    cout << "\tXYZ pointcloud size: " << xyz.points_.size() << endl;
                    cout << "\tXYZ pointcloud data: " << endl;
                    for (int i = 0; i < 5; i++) {
                        cout << "\t\t" << xyz.points_[i] << endl;
                    }
                    cout << "\tRadial List Size: " << radial_list.size() << endl;
                    cout << "\tRadial List Data: " << endl;
                    for (int i = 0; i < 5; i++) {
						cout << "\t\t" << radial_list[i] << endl;
					}
				    cout << endl;

                    start = chrono::high_resolution_clock::now();
                }


                Eigen::Vector3d center_mm_s = Accumulator_3D(xyz, radial_list, debug);

                if (debug) {
                    end = chrono::high_resolution_clock::now();
                    cout << "Acc Space Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl << endl;
                }


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


int main() {
    Options opts = testing_options();
    
    estimate_6d_pose_lm();

    return 0;

}





