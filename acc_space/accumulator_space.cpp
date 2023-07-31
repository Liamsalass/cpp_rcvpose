#include "accumulator_space.h"

using namespace std;
using namespace open3d;
namespace e = Eigen;

typedef e::MatrixXd matrix;
typedef shared_ptr<geometry::PointCloud> pc_ptr;
typedef geometry::PointCloud pc;

const vector<double> mean = { 0.485, 0.456, 0.406 };
const vector<double> standard = { 0.229, 0.224, 0.225 };




void FCResBackbone(DenseFCNResNet152& model, const string& input_path, torch::Tensor& radial_output, torch::Tensor& semantic_output, const torch::DeviceType device_type, const bool& debug) {

    torch::Device device(torch::kCPU);

    if (debug) {
        cout << "Loading image from path: \n\t" << input_path << endl << endl;
    }
    model->to(device);

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

    model->to(torch::kCPU);

    //semantic_output = semantic_output.squeeze(2);
    //radial_output = radial_output.squeeze(2);
}

void estimate_6d_pose_lm(const Options opts = testing_options())
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
    

    if (opts.verbose) {
        cout << endl;
        cout << "\t\t\Debug Mode" << endl;
        cout << "\t\t\b------------" << endl << endl;
    }
    
    if (opts.demo_mode) {
        cout << "\t\tDemo Mode" << endl;
        cout << "\t\t\b-----------" << endl << endl;
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

        if (opts.verbose) {
			cout << "Loading point cloud from path: \n\t" << pcv_load_path << endl;
		}

        pc_ptr pcv = read_point_cloud(pcv_load_path, opts.verbose);

        long long net_time = 0;
        long long acc_time = 0;
        int general_counter = 0;

        int bf_icp = 0;
        int af_icp = 0;

        vector<string> filename_list;
        
        vector<DenseFCNResNet152> model_list;

        cout << endl << "Loading Models" << endl;
        for (int i = 1; i < 4; i++) {
        
            string model_dir = "kpt" + to_string(i);

            if (opts.verbose) {
                cout << "\t" << model_dir << endl;
            }

            DenseFCNResNet152 model;

            CheckpointLoader loader(model_dir, true);

            model = loader.getModel();
            model->eval();
            
            model_list.push_back(model);

            //if (opts.verbose) {
            //    break;
            //}

        }
        if (opts.verbose) {
            cout << endl;
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

        if (opts.verbose) {
            cout << "XYZ data (top 5)" << endl;
            for (int i = 0; i < 5; i++) {
                cout << "\tx: " << xyz_load[i].x << " y: " << xyz_load[i].y << " z: " << xyz_load[i].z << endl;
            }
            cout << endl;
        }

        string keypoints_path = opts.root_dataset + "/LINEMOD/" + class_name + "/Outside9.npy";

        if (opts.verbose) {
			cout << "Loading keypoints from path: \n\t" << keypoints_path << endl << endl;
		}

        vector<vector<double>> keypoints = read_key_points(keypoints_path, opts.verbose);
        
        string data_path = rootpvPath + "JPEGImages/";

        int img_num = 0;

        //ms clock
        auto acc_start = chrono::high_resolution_clock::now();

        //Loop through all images in the img_path
        for (auto test_img : test_list){
            auto img_start = chrono::high_resolution_clock::now();

            cout << string(75, '-') << endl;
            cout << string(15, ' ') << "Estimating for image " << test_img << ".jpg" << endl;

            string image_path = data_path + test_img + ".jpg";
            
            
            double estimated_kpts[3][3];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    estimated_kpts[i][j] = 0;
                }
            }

            string RTGT_path = opts.root_dataset + "/LINEMOD/" + class_name + "/pose_txt/pose" + to_string(img_num) + ".txt";

            if (opts.verbose) {
                cout << "RTGT Path: \n\t" << RTGT_path << endl << endl;
            }

            img_num++;

            vector<vector<double>> RTGT = read_ground_truth(RTGT_path, true);

            if (opts.verbose) {
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
            
            matrix xyz_load_matrix = convertToEigenMatrix(xyz_load);
            matrix linemod_K_matrix = convertToEigenMatrix(linemod_K);

            MatrixXd dump, xyz_load_transformed;


            if (opts.verbose) {
                cout << "Projecting pointcloud to image plane" << endl;
            }

            matrix RTGT_matrix = vectorOfVectorToEigenMatrix(RTGT);

            project(xyz_load_matrix, linemod_K_matrix, RTGT_matrix, dump, xyz_load_transformed);

            for (const auto& keypoint : keypoints) {
        
                cout << string(50, '-') << endl;
                cout << string(15, ' ')  << "Keypoint Count: " << keypoint_count << endl;

                //Ask Greenspan about this
                auto keypoint = keypoints[keypoint_count];

                if (opts.verbose) {
                    cout << "Keypoint data: \n" << keypoint << endl << endl;
                }
            
                int iteration_count = 0;
            
                vector<MatrixXd> centers_list;

                //string GTRadiusPath = rootPath + "Out_pt" + to_string(keypoint_count) + "_dm

                matrix keypoint_matrix = vectorToEigenMatrix(keypoint);
                
                //transformed_gt_center_mm = (np.dot(keypoint, RTGT[:, :3].T) + RTGT[:, 3:].T)*1000

                auto transformed_gt_center_mm = transformKeyPoint(keypoint_matrix, RTGT_matrix, false);

                if (opts.verbose) {
                    cout << "Transformed Ground Truth Center: " << endl;
                    cout << transformed_gt_center_mm << endl << endl;
                }

                torch::Tensor semantic_output;
                torch::Tensor radial_output;

                auto start = chrono::high_resolution_clock::now();
               
                FCResBackbone(model_list.at(keypoint_count - 1), image_path, radial_output, semantic_output, device_type, opts.verbose);

                auto end = chrono::high_resolution_clock::now();
                
                if (opts.verbose) {
                    cout << "FCResBackbone Speed: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
                    cout << "Semantic Output Shape: " << semantic_output.sizes() << endl;
                    cout << "Radial Output Shape: " << radial_output.sizes() << endl << endl;;
                }

                net_time += chrono::duration_cast<chrono::milliseconds>(end - start).count();

                cv::Mat sem_cv = torch_tensor_to_cv_mat(semantic_output);
                cv::Mat rad_cv = torch_tensor_to_cv_mat(radial_output);

                string depth_path = rootPath + "data/depth" + to_string(img_num) + ".dpt";

                cv::Mat depth_cv = read_depth_to_cv(depth_path, opts.verbose);
      
                if (opts.verbose) {
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
                if (opts.verbose) {
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

                if (opts.verbose) {
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

                if (opts.verbose) {
                    cout << "\tRadial List Size: " << radial_list.size() << endl << endl;
                    cout << "Converting depth image to pointcloud" << endl;
                }

                //Check imkplementation since values are slightly off from python (Most likely depth that is different causing different output)
                auto xyz = rgbd_to_point_cloud(linemod_K, depth_cv);


                if (opts.verbose) {
					cout << "\tPointcloud Size: " << xyz.points_.size() << endl << endl;
				}


                if (opts.verbose) {
                    cout << "\tTransformed Pointcloud Size: " << xyz_load_transformed.rows() << endl;
                    cout << "\tTransformed Pointcloud 5 points: " << endl;
                    for (int i = 0; i < 5; i++) {
						cout << "\t\t" << xyz_load_transformed.row(i) << endl;

					}
                    cout << endl;
                }


                if (opts.verbose) {
                    cout << "Calculating 3D vector center (Accumulator_3D)" << endl;
                    start = chrono::high_resolution_clock::now();
                }

                Vector3d center_mm_s = Accumulator_3D(xyz, radial_list, false, true, false);

                if (opts.verbose) {
                    end = chrono::high_resolution_clock::now();
                    cout << "\tAcc Space Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl << endl;
                    cout << "\tNumber of Centers Returned: " << center_mm_s.size() << endl;
                    cout << "\tEstimate: " << center_mm_s << endl << endl;
                    cout << "\tCalculating offset" << endl;
                }

                auto estimated_center_mm = center_mm_s;

                double pre_center_off_mm = numeric_limits<double>::infinity();

                Vector3d transformed_gt_center_mm_vector(transformed_gt_center_mm(0, 0), transformed_gt_center_mm(0, 1),transformed_gt_center_mm(0, 2));

                Vector3d diff = transformed_gt_center_mm_vector - estimated_center_mm;

                double center_off_mm = diff.norm();

                cout << "Estimated offset: " << center_off_mm << endl << endl;

                if (opts.verbose) {
                    cout << "Saving estimation to centers data" << endl;
                }

                MatrixXd centers = MatrixXd::Zero(1, 9); // 1 row and 9 columns since Eigen doesn't support 3D structures directly.

                centers(0, 0) = keypoint[0];
                centers(0, 1) = keypoint[1];
                centers(0, 2) = keypoint[2];
                
                transformed_gt_center_mm_vector *= 0.001;

                centers(0, 3) = transformed_gt_center_mm_vector[0];
                centers(0, 4) = transformed_gt_center_mm_vector[1];
                centers(0, 5) = transformed_gt_center_mm_vector[2];

                estimated_center_mm *= 0.001;
                centers(0, 6) = estimated_center_mm[0];
                centers(0, 7) = estimated_center_mm[1];
                centers(0, 8) = estimated_center_mm[2];

                for (int i = 0; i < 3; i++) {
                    estimated_kpts[keypoint_count - 1][i] = estimated_center_mm[i] * 1000;
                }
                
  
                centers_list.push_back(centers);
    
                filename_list.push_back(test_img);

                iteration_count++;

                keypoint_count++;

                if (keypoint_count == 4) {
                    break;
                }
            }

            const int num_keypoints = 3;  
            double kpts[num_keypoints][3]; 

            double RT[4][4];

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
					RT[i][j] = 0;
				}
			}


            for (int i = 0; i < num_keypoints; i++) {
                for (int j = 0; j < 3; j++) {
                    kpts[i][j] = keypoints[i][j];
                }
            }

            if (opts.verbose) {
                cout << "Calculating RT" << endl;
                cout << "Input data: " << endl;
                cout << "\tkpts: " << endl;
                for (int i = 0; i < num_keypoints; i++) {
                    cout << "\t\t";
                    for (int j = 0; j < 3; j++) {
						cout << kpts[i][j] << " ";
					}
                    cout << endl;
                }
                cout << "\testimated_kpts: " << endl;
                for (int i = 0; i < num_keypoints; i++) {
					cout << "\t\t";
                    for (int j = 0; j < 3; j++) {
                        cout << estimated_kpts[i][j] << " ";
                    }
                    cout << endl;
                }
                cout << endl;
            }   


            lmshorn(kpts, estimated_kpts, num_keypoints, RT);

            if (opts.verbose) {
                cout << "RT Data: " << endl;
                for (int i = 0; i < 4; i++) {
                    cout << "\t";
                    for (int j = 0; j < 4; j++) {
                        cout << RT[i][j] << " ";
                    }
                    cout << endl;
                }
                cout << endl;
                cout << "Preforming Projection on estimated RT" << endl;
            }

            //Convert RT to MatrixXd
            MatrixXd RT_matrix = MatrixXd::Zero(3, 4);

     
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 4; j++) {
                    RT_matrix(i, j) = RT[i][j];
                }
            }

            matrix xy, xyz_load_est_transformed;

            if (opts.verbose) {
                cout << "Projecting Estimated position" << endl;
            }

            //Check outputs of function, xy out of range of data
            project(xyz_load_matrix * 1000, linemod_K_matrix, RT_matrix, xy, xyz_load_est_transformed);

            if (opts.verbose) {
                cout << "XYZ_load_est_transformed: " << endl;
                cout << "\tXYZ_load_est_transformed size : " << xyz_load_est_transformed.rows() << " x " << xyz_load_est_transformed.cols() << endl;
                for (int i = 0; i < 5; i++) {
                    cout << "\t" << xyz_load_est_transformed.row(i) << endl;
                }
                cout << endl;
            }

            if (opts.demo_mode) {
                cout << "XY:" << endl;
                cout << "\tXY Size: " << xy.rows() << " x " << xy.cols() << endl;
                cout << "\tXY Data: " << endl;
                for (int i = 0; i < 5; i++) {
                    cout << "\t\t" << xy.row(i) << endl;
                }
                cout << endl << endl;
                cout << "Displaying " << test_img << " with estimated position" << endl;
                int out_of_range = 0;
                cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
                cv::Mat img_only_points;
                img_only_points = cv::Mat::zeros(img.size(), CV_8UC3);

                for (int i = 0; i < xy.rows(); i++) {
                    int y = static_cast<int>(xy(i, 1) / 1000);
                    int x = static_cast<int>(xy(i, 0) / 1000);
                    if (0 <= y && y < xy.rows() && 0 <= x && x < xy.cols()) {
                        img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
                        img_only_points.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
                    }
                    else {
                        out_of_range++;
                    }
                }
                cout << "Number of points out of image range: " << out_of_range << endl;
                cv::imshow("Image with point overlay", img);
                cv::imshow("Image with points", img_only_points);
            }

            vector<Vector3d> pointsGT, pointsEst;

            for (int i = 0; i < xyz_load_transformed.rows(); ++i) {
                pointsGT.push_back(1000 * Vector3d(xyz_load_transformed(i, 0), xyz_load_transformed(i, 1), xyz_load_transformed(i, 2)));
            }

            for (int i = 0; i < xyz_load_est_transformed.rows(); ++i) {
                pointsEst.push_back(Vector3d(xyz_load_est_transformed(i, 0), xyz_load_est_transformed(i, 1), xyz_load_est_transformed(i, 2)));
            }

            geometry::PointCloud sceneGT;
            geometry::PointCloud sceneEst;

            sceneGT.points_ = pointsGT;
            sceneEst.points_ = pointsEst;

            sceneGT.PaintUniformColor({ 0,0,1 });
            sceneEst.PaintUniformColor({ 1,0,0 });

            auto distances = sceneGT.ComputePointCloudDistance(sceneEst);
            double distance = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
            double min_distance = *std::min_element(distances.begin(), distances.end());

            if (opts.verbose) {
				cout << "Distance: " << distance << endl;
				cout << "Min Distance: " << min_distance << endl;
			}


            if (distance <= add_threshold[class_name] * 1000) {
                bf_icp += 1;
            }
         
            Eigen::Matrix4d trans_init = Eigen::Matrix4d::Identity();

            double threshold = distance;

            open3d::pipelines::registration::ICPConvergenceCriteria criteria(2000000);

            auto reg_p2p = open3d::pipelines::registration::RegistrationICP(
                sceneGT, sceneEst, threshold, trans_init,
                open3d::pipelines::registration::TransformationEstimationPointToPoint(),
                criteria);

            sceneGT.Transform(reg_p2p.transformation_);

            distances = sceneGT.ComputePointCloudDistance(sceneEst);

            distance = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();

            min_distance = *std::min_element(distances.begin(), distances.end());

            if (distance <= add_threshold[class_name] * 1000) {
                af_icp += 1;
            }
            
            general_counter += 1;

            auto img_end = chrono::high_resolution_clock::now();

            if (opts.verbose) {
                auto duration = chrono::duration_cast<chrono::seconds>(img_end - img_start);
                auto min = duration.count() / 60;
                auto sec = duration.count() % 60;
                cout << test_img << " took " << min << " minutes and " << sec << " seconds to calculate offset." << endl;
            }
            if (opts.demo_mode) {
                cout << "Press any key to continue..." << endl;
                cin.get();
            }
        }

        auto acc_end = chrono::high_resolution_clock::now();

        cout << "ADD(s) of " << class_name << " before ICP: " << bf_icp / general_counter << endl;
        cout << "ADD(s) of " << class_name << " after ICP: " << af_icp / general_counter << endl;
        cout << "Accumulator time: " << acc_time / general_counter << endl;
        auto duration = chrono::duration_cast<chrono::seconds>(acc_end - acc_start);
        auto hours = duration.count() / 3600;
        auto min = (duration.count() % 3600) / 60;
        auto sec = (duration.count() % 3600) % 60;
        cout << "Total Accumulator time: " << hours << " hours, " << min << " minutes, and " << sec << " seconds." << endl;

        return; 
    }
}


int main() {
    Options opts = testing_options();
    
    estimate_6d_pose_lm();

    return 0;

}





