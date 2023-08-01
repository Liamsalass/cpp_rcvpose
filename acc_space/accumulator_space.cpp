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

// Function for estimating 6D pose for LINEMOD
void estimate_6d_pose_lm(const Options opts = testing_options())
{
    // Print header for console output
    cout << string(75, '=') << endl;
    cout << string(17, ' ') << "Estimating 6D Pose for LINEMOD" << endl;

    // Check whether CUDA is available
    bool use_cuda = torch::cuda::is_available();
    if (use_cuda) {
        cout << "Using GPU" << endl;
    }
    else {
        cout << "Using CPU" << endl;
    }

    // Check if verbose mode is enabled
    if (opts.verbose) {
        cout << endl;
        cout << "\t\t\Debug Mode" << endl;
        cout << "\t\t\b------------" << endl << endl;
    }

    // Check if demo mode is enabled
    if (opts.demo_mode) {
        cout << "\t\tDemo Mode" << endl;
        cout << "\t\t\b-----------" << endl << endl;
    }

    // Define the device to be used
    torch::DeviceType device_type = use_cuda ? torch::kCUDA : torch::kCPU;
    torch::Device device(device_type);

    // Loop over all classes in LINEMOD dataset
    for (auto class_name : lm_cls_names) {

        cout << string(75, '-') << endl << string(17, ' ') << "Estimating for " << class_name << " object" << endl << endl;

        // Define paths to dataset
        string rootPath = opts.root_dataset + "/LINEMOD_ORIG/" + class_name + "/";
        string rootpvPath = opts.root_dataset + "/LINEMOD/" + class_name + "/";

        // Open file containing test data
        ifstream test_file(opts.root_dataset + "/LINEMOD/" + class_name + "/Split/val.txt");
        vector<string> test_list;
        string line;

        // Read lines from test file
        if (test_file.is_open()) {
            while (getline(test_file, line)) {
                line.erase(line.find_last_not_of("\n\r\t") + 1);
                test_list.push_back(line);
            }
            test_file.close();
        }
        else {
            cerr << "Unable to open file containing test data" << endl;
            exit(EXIT_FAILURE);
        }
        
        if (opts.verbose) {
            cout << "Number of test images: " << test_list.size() << endl;
        }

        // Define path to load point cloud
        string pcv_load_path = "C:/Users/User/.cw/work/cpp_rcvpose/acc_space/python/pc_ape.npy";

        // Log path if verbose
        if (opts.verbose) {
            cout << "Loading point cloud from path: \n\t" << pcv_load_path << endl;
        }

        // Read the point cloud from the file
        pc_ptr pcv = read_point_cloud(pcv_load_path, opts.verbose);

        // Variables for timing
        long long net_time = 0;
        long long acc_time = 0;
        int general_counter = 0;

        // Variables for counting before and after ICP
        int bf_icp = 0;
        int af_icp = 0;

        // List to hold filenames
        vector<string> filename_list;

        // List to hold models
        vector<DenseFCNResNet152> model_list;

        // Print loading message
        cout << endl << "Loading Models" << endl;

        // Load all models
        for (int i = 1; i < 4; i++) {

            // Define directory of model
            string model_dir = "kpt" + to_string(i);

            // Log model directory if verbose
            if (opts.verbose) {
                cout << "\t" << model_dir << endl;
            }

            // Instantiate model
            DenseFCNResNet152 model;

            // Instantiate loader
            CheckpointLoader loader(model_dir, true);

            // Load the model
            model = loader.getModel();

            // Put the model in evaluation mode
            model->eval();

            // Add the model to the model list
            model_list.push_back(model);
        }

        // Print an extra line if verbose
        if (opts.verbose) {
            cout << endl;
        }

        // Vector to hold file list
        vector<string> file_list;
        vector<string> file_list_icp;
        vector<string> incorrect_list;

        // Vector to hold xyz data
        vector<Vertex> xyz_load;

        // Convert point cloud points to Vertex and add to xyz_load
        for (auto point : pcv->points_) {
            Vertex v;
            v.x = point.x();
            v.y = point.y();
            v.z = point.z();
            xyz_load.push_back(v);
        }

        // Print top 5 XYZ data if verbose
        if (opts.verbose) {
            cout << "XYZ data (top 5)" << endl;
            for (int i = 0; i < 5; i++) {
                cout << "\tx: " << xyz_load[i].x << " y: " << xyz_load[i].y << " z: " << xyz_load[i].z << endl;
            }
            cout << endl;
        }

        // Define path to keypoints
        string keypoints_path = opts.root_dataset + "/LINEMOD/" + class_name + "/Outside9.npy";

        // Log path if verbose
        if (opts.verbose) {
            cout << "Loading keypoints from path: \n\t" << keypoints_path << endl << endl;
        }

        // Load keypoints from the file
        vector<vector<double>> keypoints = read_key_points(keypoints_path, opts.verbose);

        // Define path to data
        string data_path = rootpvPath + "JPEGImages/";


        // Start high resolution timer for accumulation
        auto acc_start = chrono::high_resolution_clock::now();

        // Loop through all images in the img_path
        for (auto test_img : test_list) {
         
            int img_num = stoi(test_img);
    
            // Record the current time using a high-resolution clock
            auto img_start = chrono::high_resolution_clock::now();

            // Print a separator and a message for the current image being estimated
            cout << string(75, '-') << endl;
            cout << string(15, ' ') << "Estimating for image " << test_img << ".jpg" << endl;

            // Generate the image path by appending the image name to the base data path
            string image_path = data_path + test_img + ".jpg";

            // Initialize a 3x3 matrix of estimated keypoints with zeros
            double estimated_kpts[3][3];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    estimated_kpts[i][j] = 0;
                }
            }

            // Define the path of the ground truth rotation and translation (RTGT)
            string RTGT_path = opts.root_dataset + "/LINEMOD/" + class_name + "/pose_txt/pose" + to_string(img_num) + ".txt";

            // Print the RTGT path if verbose option is enabled
            if (opts.verbose) {
                cout << "RTGT Path: \n\t" << RTGT_path << endl << endl;
            }  

            // Load the ground truth rotation and translation (RTGT) data from the path
            vector<vector<double>> RTGT = read_ground_truth(RTGT_path, true);

            // Print the RTGT data if verbose option is enabled
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

            // Initialize keypoint count
            int keypoint_count = 1;

            // Convert the xyz_load and linemod_K to Eigen Matrices for further computations
            matrix xyz_load_matrix = convertToEigenMatrix(xyz_load);
            matrix linemod_K_matrix = convertToEigenMatrix(linemod_K);

            // Define empty matrices for storing dump and transformed load data
            MatrixXd dump, xyz_load_transformed;

            // Message to indicate projection of pointcloud to image plane if verbose option is enabled
            if (opts.verbose) {
                cout << "Projecting pointcloud to image plane" << endl;
            }

            // Convert the RTGT data to an Eigen matrix
            matrix RTGT_matrix = vectorOfVectorToEigenMatrix(RTGT);

            // Project the point cloud data onto the image plane
            project(xyz_load_matrix, linemod_K_matrix, RTGT_matrix, dump, xyz_load_transformed);

            // Loop over each keypoint for estimation
            for (const auto& keypoint : keypoints) {

                // Print a separator and the current keypoint count
                cout << string(50, '-') << endl;
                cout << string(15, ' ') << "Keypoint Count: " << keypoint_count << endl;

                // Get the current keypoint
                auto keypoint = keypoints[keypoint_count];

                // Print the keypoint data if verbose option is enabled
                if (opts.verbose) {
                    cout << "Keypoint data: \n" << keypoint << endl << endl;
                }

                // Initialize iteration count
                int iteration_count = 0;

                // Initialize list for storing centers
                vector<MatrixXd> centers_list;

                // Convert the current keypoint to an Eigen matrix for further computations
                matrix keypoint_matrix = vectorToEigenMatrix(keypoint);

                // Compute the transformed ground truth center in millimeters
                auto transformed_gt_center_mm = transformKeyPoint(keypoint_matrix, RTGT_matrix, false);

                // Print the transformed ground truth center if verbose option is enabled
                if (opts.verbose) {
                    cout << "Transformed Ground Truth Center: " << endl;
                    cout << transformed_gt_center_mm << endl << endl;
                }

                // Declare tensors for storing the semantic and radial output
                torch::Tensor semantic_output;
                torch::Tensor radial_output;

                // Record the current time before executing the FCResBackbone
                auto start = chrono::high_resolution_clock::now();

                // Execute the FCResBackbone model to get semantic and radial output
                FCResBackbone(model_list.at(keypoint_count - 1), image_path, radial_output, semantic_output, device_type, opts.verbose);

                // Record the current time after executing the FCResBackbone
                auto end = chrono::high_resolution_clock::now();

                // Print the FCResBackbone speed, semantic output shape, and radial output shape if verbose option is enabled
                if (opts.verbose) {
                    cout << "FCResBackbone Speed: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
                    cout << "Semantic Output Shape: " << semantic_output.sizes() << endl;
                    cout << "Radial Output Shape: " << radial_output.sizes() << endl << endl;;
                }

                // Add the time taken to execute the FCResBackbone to the total network time
                net_time += chrono::duration_cast<chrono::milliseconds>(end - start).count();

                // Convert the semantic and radial output tensors to OpenCV matrices
                cv::Mat sem_cv = torch_tensor_to_cv_mat(semantic_output);
                cv::Mat rad_cv = torch_tensor_to_cv_mat(radial_output);

                // Define the depth image path
                string depth_path = rootPath + "data/depth" + to_string(img_num) + ".dpt";

                // Load the depth image
                cv::Mat depth_cv = read_depth_to_cv(depth_path, opts.verbose);

                // Print the shapes and datatypes of semantic, radial, and depth data if verbose option is enabled
                if (opts.verbose) {
                    cout << endl << "Data Shapes: " << endl;
                    cout << "\tSemantic Shape: " << sem_cv.size() << endl;
                    cout << "\tSemantic Datatype: " << sem_cv.type() << endl;
                    cout << "\tRadial Shape: " << rad_cv.size() << endl;
                    cout << "\tRadial Datatype: " << rad_cv.type() << endl;
                    cout << "\tDepth Shape: " << depth_cv.size() << endl;
                    cout << "\tDepth Datatype: " << depth_cv.type() << endl << endl;
                }

                // Transpose the semantic and radial matrices for correct orientation
                cv::transpose(sem_cv, sem_cv);
                cv::transpose(rad_cv, rad_cv);

                // Threshold the semantic matrix to binary
                cv::Mat thresholded;
                cv::threshold(sem_cv, thresholded, 0.8, 1, cv::THRESH_BINARY);
                thresholded.convertTo(sem_cv, sem_cv.type());
                thresholded.release();

                // Define temporary matrices for semantic, depth, and radial data
                cv::Mat sem_tmp, depth_tmp, rad_tmp;

                // Convert the datatypes of semantic, depth, and radial matrices
                sem_cv.convertTo(sem_tmp, CV_32F);
                depth_cv.convertTo(depth_tmp, CV_32F);
                rad_cv.convertTo(rad_tmp, CV_32F);
                

                // Multiply the radial matrix by the semantic matrix
                rad_tmp = rad_tmp.mul(sem_cv);
                depth_tmp = depth_tmp.mul(sem_tmp);
                depth_tmp = depth_tmp / 1000.0;
                depth_cv = depth_tmp.clone();
                rad_cv = rad_tmp.clone();
                
                // Release the temporary matrices
                rad_tmp.release();
                depth_tmp.release();
                sem_tmp.release();

                // Print the shapes and datatypes of semantic, radial, and depth data if verbose option is enabled
                vector<Vertex> pixel_coor;
                if (opts.verbose) {
                    cout << "Gathering Pixel Coordinates from semantic output" << endl;
                }

                // Gather the pixel coordinates from the semantic output
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

                // Print the number of pixel coordinates gathered if verbose option is enabled
                if (opts.verbose) {
                    cout << "\tPixel Coord Size: " << pixel_coor.size() << endl;
                }

                // Define a vector for storing the radial values
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

                // Print the number of radial values gathered if verbose option is enabled
                if (opts.verbose) {
                    cout << "\tRadial List Size: " << radial_list.size() << endl << endl;
                    cout << "Converting depth image to pointcloud" << endl;
                }

                //Check imkplementation since values are slightly off from python (Most likely depth that is different causing different output)
                // Convert the depth image to a pointcloud
                auto xyz = rgbd_to_point_cloud(linemod_K, depth_cv);

                // Print the number of points in the pointcloud if verbose option is enabled
                if (opts.verbose) {
					cout << "\tPointcloud Size: " << xyz.points_.size() << endl << endl;
				}

                // Define a vector for storing the transformed pointcloud
                if (opts.verbose) {
                    cout << "\tTransformed Pointcloud Size: " << xyz_load_transformed.rows() << endl;
                    cout << "\tTransformed Pointcloud 5 points: " << endl;
                    for (int i = 0; i < 5; i++) {
						cout << "\t\t" << xyz_load_transformed.row(i) << endl;

					}
                    cout << endl;
                }

                // Define a vector for storing the transformed pointcloud
                if (opts.verbose) {
                    cout << "Calculating 3D vector center (Accumulator_3D)" << endl;
                    start = chrono::high_resolution_clock::now();
                }

                // Calculate the 3D vector center
                Vector3d estimated_center_mm = Accumulator_3D(xyz, radial_list, false, true, false);

                // Print the number of centers returned and the estimate if verbose option is enabled
                if (opts.verbose) {
                    end = chrono::high_resolution_clock::now();
                    cout << "\tAcc Space Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl << endl;
                    cout << "\tNumber of Centers Returned: " << estimated_center_mm.size() << endl;
                    cout << "\tEstimate: " << estimated_center_mm << endl << endl;
                    cout << "\tCalculating offset" << endl;
                }


                // Define a vector for storing the transformed pointcloud
                double pre_center_off_mm = numeric_limits<double>::infinity();

                // Define a vector for storing the transformed pointcloud
                Vector3d transformed_gt_center_mm_vector(transformed_gt_center_mm(0, 0), transformed_gt_center_mm(0, 1),transformed_gt_center_mm(0, 2));

                // Calculate the offset
                Vector3d diff = transformed_gt_center_mm_vector - estimated_center_mm;
                double center_off_mm = diff.norm();

                if (opts.verbose) {
                    cout << "Estimated offset: " << center_off_mm << endl << endl;
                    cout << "Saving estimation to centers data" << endl;
                }

                // Save the estimation to the centers data
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

                // Save the estimation to the centers data
                for (int i = 0; i < 3; i++) {
                    estimated_kpts[keypoint_count - 1][i] = estimated_center_mm[i] * 1000;
                }
                
                // Save the estimation to the centers data
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

            // Calculate the RT matrix
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

            // Project the estimated position
            project(xyz_load_matrix * 1000, linemod_K_matrix, RT_matrix, xy, xyz_load_est_transformed);

            if (opts.verbose) {
                cout << "XYZ_load_est_transformed: " << endl;
                cout << "\tXYZ_load_est_transformed size : " << xyz_load_est_transformed.rows() << " x " << xyz_load_est_transformed.cols() << endl;
                for (int i = 0; i < 5; i++) {
                    cout << "\t" << xyz_load_est_transformed.row(i) << endl;
                }
                cout << endl;
            }

            // If in demo mode, display the estimated position
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

            // Load the ground truth points
            for (int i = 0; i < xyz_load_transformed.rows(); ++i) {
                pointsGT.push_back(1000 * Vector3d(xyz_load_transformed(i, 0), xyz_load_transformed(i, 1), xyz_load_transformed(i, 2)));
            }
            // Load the estimated points
            for (int i = 0; i < xyz_load_est_transformed.rows(); ++i) {
                pointsEst.push_back(Vector3d(xyz_load_est_transformed(i, 0), xyz_load_est_transformed(i, 1), xyz_load_est_transformed(i, 2)));
            }

            // Create point clouds for the ground truth and estimated points
            geometry::PointCloud sceneGT;
            geometry::PointCloud sceneEst;

            sceneGT.points_ = pointsGT;
            sceneEst.points_ = pointsEst;

            // Paint the point clouds
            sceneGT.PaintUniformColor({ 0,0,1 });
            sceneEst.PaintUniformColor({ 1,0,0 });

            if (opts.verbose) {
                cout << "Computing distance between ground truth and estimated point clouds" << endl;
            }

            // Compute the distance between the ground truth and estimated point clouds
            auto distances = sceneGT.ComputePointCloudDistance(sceneEst);
            double distance = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
            double min_distance = *std::min_element(distances.begin(), distances.end());

            if (opts.verbose) {
				cout << "\tBefore ICP distance: " << distance << endl;
				cout << "\tBefore ICP min distance: " << min_distance << endl << endl;
                cout << "Calculating ICP" << endl;
			}

            // If the distance is less than the threshold, increment the number of correct matches
            if (distance <= add_threshold[class_name] * 1000) {
                bf_icp += 1;
                if (opts.verbose) {
                    cout << "\tICP not needed" << endl;
                    cout << "\tSaving file" << endl;
                }
                ofstream bf_icp_file;
                bf_icp_file.open("bf_icp_" + class_name + ".txt");
                bf_icp_file << bf_icp;
                bf_icp_file.close();
            }
            else {
                if (opts.verbose) {
					cout << "\tICP needed" << endl;
				}
			}
            // Perform ICP
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

            if (opts.verbose) {
                cout << "\tAfter ICP distance: " << distance << endl;
                cout << "\tAfter ICP min distance: " << min_distance << endl << endl;
            }

            // If the distance is less than the threshold, increment the number of correct matches
            if (distance <= add_threshold[class_name] * 1000) {
                af_icp += 1;
                if (opts.verbose) {
                    cout << "\tCorrect match!" << endl;
                    cout << "\tSaving file" << endl;
                }
                ofstream af_icp_file;
                af_icp_file.open("af_icp_" + class_name + ".txt");
                af_icp_file << af_icp;
                af_icp_file.close();
            }
            else {
                if (opts.verbose) {
                    cout << "\tIncorrect match!" << endl;
                    cout << "\tSaving file" << endl;
                }
                ofstream incorrect_file;
                incorrect_file.open("incorrect_" + class_name + ".txt");
                incorrect_file << test_img << endl;
                incorrect_file.close();

            

            }
            
            general_counter += 1;

            auto img_end = chrono::high_resolution_clock::now();

            if (opts.verbose) {
                auto duration = chrono::duration_cast<chrono::seconds>(img_end - img_start);
                auto min = duration.count() / 60;
                auto sec = duration.count() % 60;
                cout << "Image number " << img_num << " took " << min << " minutes and " << sec << " seconds to calculate offset." << endl;
                cout << "Before ICP Count " << bf_icp << endl;
                cout << "After ICP Count " << af_icp << endl;
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





