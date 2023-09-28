#include "accumulator_space.h"

namespace fs = std::filesystem;
using namespace std;
using namespace open3d;
namespace e = Eigen;


typedef e::MatrixXd matrix;
typedef shared_ptr<geometry::PointCloud> pc_ptr;
typedef geometry::PointCloud pc;

const vector<double> mean = { 0.485, 0.456, 0.406 };
const vector<double> standard = { 0.229, 0.224, 0.225 };

unordered_map<string, double> add_threshold = {
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



void FCResBackbone(DenseFCNResNet152& model, cv::Mat& img, torch::Tensor& semantic_output, torch::Tensor& radial_output1, torch::Tensor& radial_output2, torch::Tensor& radial_output3, const torch::DeviceType device_type, const bool& debug = false) {
    torch::Device device(device_type);

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

    torch::NoGradGuard no_grad;
    
    auto output = model->forward(img_batch);

    output = output.to(torch::kCPU);

    radial_output1 = output[0][0];
    radial_output2 = output[0][1];
    radial_output3 = output[0][2];
    semantic_output = output[0][3];
}



// Function for estimating 6D pose for LINEMOD
void estimate_6d_pose_lm(const Options& opts, DenseFCNResNet152& model)
{
    // Print header for console output
    cout << endl << string(75, '=') << endl;
    cout << string(17, ' ') << "Estimating 6D Pose for LINEMOD" << endl;

    // Check whether CUDA is available
    bool use_cuda = torch::cuda::is_available();



    // Check if verbose mode is enabled
    if (opts.verbose) {
        if (use_cuda) {
            cout << "Using GPU" << endl;
        }
        else {
            cout << "Using CPU" << endl;
        }
        cout << endl;
        cout << "\t\tDebug Mode" << endl;
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
    string class_name = opts.class_name;

    if (opts.verbose) {
        cout << string(75, '-') << endl << string(17, ' ') << "Estimating for " << class_name << " object" << endl << endl;
    }
    else {
        cout << "Estimating for " << class_name << endl;
    }

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

    const int total_num_img = test_list.size();

    cout << "Number of test images: " << total_num_img << endl;
    
    const float mask_threshold = opts.mask_threshold;

    cout << "Masking Threshold: " << mask_threshold << endl;


    // Define path to load point cloud
    string pcv_load_path = rootpvPath + "pc_" + class_name + ".npy";

    // Log path if verbose
    if (opts.verbose) {
        cout << "Loading point cloud from path: \n\t" << pcv_load_path << endl;
    }

    // Read the point cloud from the file
    pc_ptr pcv = read_point_cloud(pcv_load_path, false);

    // Variables for timing
    long long backend_net_time = 0;
    long long icp_time = 0;
    long long acc_time = 0;
    long long avg_acc_length = 0;
    int general_counter = 0;

    // Variables for counting before and after ICP
    int bf_icp = 0;
    int af_icp = 0;


    vector<double> bf_icp_distances, bf_icp_min_distances, bf_icp_max_distances, bf_icp_median_distances, bf_icp_std_deviations;
    vector<double> af_icp_distances, af_icp_min_distances, af_icp_max_distances, af_icp_median_distances, af_icp_std_deviations;



    // List to hold filenames
    vector<string> filename_list;



    // Put the model in evaluation mode
    model->eval();

    model->to(device);

    //Pass the model a dummy tensor
    try {
        if (opts.verbose) {
            cout << "Testing model with dummy tensor" << endl;
        }
        torch::Tensor dummy_tensor = torch::randn({ 1, 3, 640, 480 });
        dummy_tensor = dummy_tensor.to(device);

        auto dummy_output = model->forward(dummy_tensor);
        if (opts.verbose) {
            cout << "Model test successful" << endl;
        }
    }
    catch (const c10::Error& e) {
        cerr << "Error loading model" << endl;
        exit(EXIT_FAILURE);
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
    vector<vector<double>> keypoints = read_double_npy(keypoints_path, false);



    // Define path to data
    string data_path = rootpvPath + "JPEGImages/";


    // Start high resolution timer for accumulation
    auto acc_start = chrono::high_resolution_clock::now();
    // Loop through all images in the img_path
    for (auto test_img : test_list) {
        int img_num = stoi(test_img);

        //if (img_num != 10) {
        //    general_counter++;
        //    continue;
        //}

        // Record the current time using a high-resolution clock
        auto img_start = chrono::high_resolution_clock::now();

        // Print a separator and a message for the current image being estimated
        if (opts.verbose) {
            cout << string(75, '-') << endl;
            cout << string(15, ' ') << "Estimating for image " << test_img << ".jpg" << endl;
        }

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

        // Load the ground truth rotation and translation (RTGT) data from the path
        vector<vector<double>> RTGT = read_ground_truth(RTGT_path, false);


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
        Eigen::MatrixXd dump, xyz_load_transformed;


        // Convert the RTGT data to an Eigen matrix
        matrix RTGT_matrix = vectorOfVectorToEigenMatrix(RTGT);

        // Project the point cloud data onto the image plane
        project(xyz_load_matrix, linemod_K_matrix, RTGT_matrix, dump, xyz_load_transformed);

        torch::Tensor semantic_output;
        torch::Tensor radial_output1;
        torch::Tensor radial_output2;
        torch::Tensor radial_output3;

     
        cv::Mat img = cv::imread(image_path);


        // Record the current time before executing the FCResBackbone
        auto start = chrono::high_resolution_clock::now();


        //FCResBackbone(model, img, semantic_output, radial_output1, radial_output2, radial_output3, device_type, false);

        // Use GT for testing
        string gt1_path = opts.root_dataset + "/LINEMOD/" + class_name + "/Out_pt1_dm/" + test_img + ".npy";
        string gt2_path = opts.root_dataset + "/LINEMOD/" + class_name + "/Out_pt2_dm/" + test_img + ".npy";
        string gt3_path = opts.root_dataset + "/LINEMOD/" + class_name + "/Out_pt3_dm/" + test_img + ".npy";
        radial_output1 = npy_to_tensor(gt1_path);
        radial_output2 = npy_to_tensor(gt2_path);
        radial_output3 = npy_to_tensor(gt3_path);

        // Use python estimated radii maps (paper backend)
        //string r1_path = opts.root_dataset + "/LINEMOD/" + class_name + "/Estimated_out_pt1_dm/" + test_img + ".npy";
        //string r2_path = opts.root_dataset + "/LINEMOD/" + class_name + "/Estimated_out_pt2_dm/" + test_img + ".npy";
        //string r3_path = opts.root_dataset + "/LINEMOD/" + class_name + "/Estimated_out_pt3_dm/" + test_img + ".npy";
        //torch::Tensor radial_output1 = npy_to_tensor(r1_path);
        //torch::Tensor radial_output2 = npy_to_tensor(r2_path);
        //torch::Tensor radial_output3 = npy_to_tensor(r3_path);


        semantic_output = torch::where(radial_output1 > 0, torch::ones_like(radial_output1), -torch::ones_like(radial_output1));
        
        auto end = chrono::high_resolution_clock::now();


        backend_net_time += chrono::duration_cast<chrono::milliseconds>(end - start).count();

        vector<torch::Tensor> radial_outputs = { radial_output1, radial_output2, radial_output3 };

        // Print the FCResBackbone speed, semantic output shape, and radial output shape if verbose option is enabled
        if (opts.verbose) {
            cout << "FCResBackbone Speed: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl << endl;
        }

    

        // Loop over each keypoint for estimation
        for (const auto& keypoint : keypoints) {
            if (opts.verbose) {
                cout << string(50, '-') << endl;
                cout << string(15, ' ') << "Keypoint Count: " << keypoint_count << endl;
            }

            // Get the current keypoint
            auto keypoint = keypoints[keypoint_count];

            // Print the keypoint data if verbose option is enabled
            if (opts.verbose) {
                cout << "Keypoint data: \n" << keypoint << endl << endl;
            }

            const string mesh_path = opts.root_dataset + "/LINEMOD/" + class_name + "/mesh_npy.npy";
            vector<Vertex> cad = read_point_cloud(mesh_path);

            for (auto point : cad) {
                point.x /= 1000;
                point.y /= 1000;
                point.z /= 1000;
            }

            double max_radius_dm = 0.0;
            for (int i = 0; i < cad.size(); i++) {
                double dx = cad[i].x - keypoint[0];
                double dy = cad[i].y - keypoint[1];
                double dz = cad[i].z - keypoint[2];
                double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (max_radius_dm < distance) {
                    max_radius_dm = distance;
                }

            }

            // Initialize iteration count
            int iteration_count = 0;

            // Initialize list for storing centers
            vector<Eigen::MatrixXd> centers_list;

            // Convert the current keypoint to an Eigen matrix for further computations
            matrix keypoint_matrix = vectorToEigenMatrix(keypoint);

            // Compute the transformed ground truth center in mm
            auto transformed_gt_center_mm = transformKeyPoint(keypoint_matrix, RTGT_matrix, false);


            // Print the transformed ground truth center if verbose option is enabled
            if (opts.verbose) {
                cout << "Transformed Ground Truth Center: " << endl;
                cout << transformed_gt_center_mm << endl << endl;
            }


            // Convert the semantic and radial output tensors to OpenCV matrices
            //cv::Mat sem_cv = torch_tensor_to_cv_mat(semantic_output);
            cv::Mat rad_cv = torch_tensor_to_cv_mat(radial_outputs[keypoint_count - 1]);
            
            cv::Mat sem_cv = torch_tensor_to_cv_mat(semantic_output);

      
            // Define the depth image path
            //string depth_path = rootpvPath + "data/depth_npy/depth" + to_string(img_num) + ".npy";
            string depth_path = rootpvPath + "data/depth" + to_string(img_num);

            // Check if ".dpt" file exists, if not, check for ".png"
            if (fs::exists(depth_path + ".dpt")) {
                depth_path += ".dpt";
            }
            else if (fs::exists(depth_path + ".png")) {
                depth_path += ".png";
            }
            else {
                cout << "No suitable depth image file found for: " << depth_path << "..." << endl;
                return;
            }

            // Load the depth image
            cv::Mat depth_cv = read_depth_to_cv(depth_path, false);

            //vector<vector<double>> depth;
            //vector<UINT16> data;
            //vector<unsigned long> shape;
            //bool fortran_order;
            //
            //npy::LoadArrayFromNumpy(depth_path, shape, fortran_order, data);
            //
            //for (int i = 0; i < shape[0]; i++) {
			//	vector<double> row;
            //    for (int j = 0; j < shape[1]; j++) {
			//		row.push_back(data[i * shape[1] + j]);
			//	}
			//	depth.push_back(row);
			//}
            //cout << "Depth at max: " << depth[216][272] << endl;
            //
            //cv::Mat depth_cv = cv::Mat::zeros(480, 640, CV_32FC1);
            ////Convert depth to depth_cv
            //for (int i = 0; i < depth_cv.rows; i++) {
            //    for (int j = 0; j < depth_cv.cols; j++) {
            //        depth_cv.at<float>(i, j) = static_cast<float>(depth[i][j]);
            //    }
            //}


            // Convert the datatypes of semantic, depth, and radial matrices
            //sem_cv.convertTo(sem_cv, CV_32F);
            //depth_cv.convertTo(depth_cv, CV_32F);
            //rad_cv.convertTo(rad_cv, CV_32F);

            // Transpose the semantic and radial matrices for correct orientation

            //cv::transpose(sem_cv, sem_cv);
            //cv::transpose(rad_cv, rad_cv);

            //cv::normalize(sem_cv, sem_cv, 0, 1, cv::NORM_MINMAX);

            // Check if the images have the same dimensions
            if (sem_cv.size() != rad_cv.size() || sem_cv.type() != rad_cv.type()) {
                std::cerr << "Error: The dimensions or types of the two matrices are not the same." << std::endl;
            }

            
            //double min;
            //double max;
            //
            //cv::Point min_loc;
            //cv::Point max_loc;
            //
            //minMaxLoc(rad_cv, &min, &max, &min_loc, &max_loc);

            //cout << "Radial Max: " << max << " Radial Min: " << min << endl;
            //cout << "Max Index: " << max_loc.x << " " << max_loc.y << endl;
            //cout << "Min Index: " << min_loc.x << " " << min_loc.y << endl;

            // Initialize a 3-channel matrix with zeros
            //std::vector<cv::Mat> channels(3, cv::Mat::zeros(sem_cv.size(), sem_cv.type()));

            // Set blue channel to semantic and red channel to radial
            //channels[0] = sem_cv;  // Blue
            //channels[2] = rad_cv;  // Red

            // Merge single-channel mats in 'channels' into a 3-channel mat
            //cv::Mat combined;
            //cv::merge(channels, combined);

            // Display the result
            //cv::imshow("Combined Image", combined);

            cv::Mat thresholded;
            cv::threshold(sem_cv, thresholded, opts.mask_threshold, 1, cv::THRESH_BINARY);
            thresholded.convertTo(sem_cv, sem_cv.type());
            thresholded.release();

            cv::Mat sem_tmp, depth_tmp, rad_tmp;

            // Convert the datatypes of semantic, depth, and radial matrices
            sem_cv.convertTo(sem_tmp, CV_32F);
            depth_cv.convertTo(depth_tmp, CV_32F);
            rad_cv.convertTo(rad_tmp, CV_32F);


            // Multiply the radial matrix by the semantic matrix
            rad_tmp = rad_tmp.mul(sem_cv);
            depth_tmp = depth_tmp.mul(sem_tmp);
            depth_cv = depth_tmp.clone();
            rad_cv = rad_tmp.clone();

            rad_tmp.release();
            depth_tmp.release();
            sem_tmp.release();


            //cv::imshow("Semantic", sem_cv);
            //cv::imshow("Radial", rad_cv);

            // Gather the pixel coordinates from the semantic output
            vector<Vertex> pixel_coor;
            for (int i = 0; i < sem_cv.rows; i++) {
                for (int j = 0; j < sem_cv.cols; j++) {
                    if (sem_cv.at<float>(i, j) == 1) {
                        Vertex v;
                        v.x = i;
                        v.y = j;
                        v.z = depth_cv.at<double>(i, j);

                        pixel_coor.push_back(v);
                    }
                }
            }

   
            
            unordered_map<double, vector<Vertex>> point_list_per_radius;

     
            for (auto cord : pixel_coor) {
				double radius = static_cast<double>(rad_cv.at<float>(cord.x, cord.y));
				point_list_per_radius[radius].push_back(cord);
			}

            // Print out the number of points per radius if verbose option is enabled
            if (opts.verbose) {
                cout << "Number of points per radius: " << endl;
                for (auto item : point_list_per_radius) {
					cout << "\t" << item.first << ": " << item.second.size() << endl;
				}
				cout << endl;

            }

            




            // Print the number of pixel coordinates gathered if verbose option is enabled
            if (opts.verbose) {
                cout << "Number of pixels gatherd: " << pixel_coor.size() << endl << endl;
            }

            // Define a vector for storing the radial values
            vector<double> radial_list;

            for (auto cord : pixel_coor) {
                radial_list.push_back(static_cast<double>(rad_cv.at<float>(cord.x, cord.y)));
            }

            //minMaxLoc(depth_cv, &min, &max, &min_loc, &max_loc);
            //
            //cout << "Depth Max: " << max << endl;
            //cout << "Max Index: " << max_loc.x << " " << max_loc.y << endl;
            //
            //
            //
            //// Convert the depth image to a pointcloud
            //cv::imshow("Depth with mask", depth_cv);
            //cv::waitKey(0);



            vector<Vertex> xyz = rgbd_to_point_cloud(linemod_K, depth_cv);

            //for(int i = 0; i < 20; i++) {
			//	cout << xyz[i].x << " " << xyz[i].y << " " << xyz[i].z << endl;
			//}


            depth_cv.release();

            for (int i = 0; i < xyz.size(); i++) {
                xyz[i].x = xyz[i].x / 1000;
                xyz[i].y = xyz[i].y / 1000;
                xyz[i].z = xyz[i].z / 1000;
            }

            avg_acc_length += xyz.size();

            if (xyz.size() == 0 || radial_list.size() == 0) {
                cout << "Error: xyz or radial list is empty" << endl;
            }

            int num_above_threshold = 0;
            for (int i = 0; i < xyz.size(); i++) {
                if (radial_list[i] > max_radius_dm) {
                    radial_list[i] = 0;
                    num_above_threshold++;

                }
            }

            if (num_above_threshold != 0) {
                cout << "Max radial distance: " << max_radius_dm << endl;
                cout << "Removed " << num_above_threshold << " points from radial list.\n";
                cout << "New size: " << xyz.size() << endl;
            }

            // Define a vector for storing the transformed pointcloud
            if (opts.verbose) {
                cout << "Calculating 3D vector center (Accumulator_3D)" << endl;

            }
            //Calculate the estimated center in mm
            auto acc_start = chrono::high_resolution_clock::now();
            Eigen::Vector3d estimated_center_mm;
            try {
            

                std::future<Eigen::Vector3d> future_result = std::async(std::launch::async, Accumulator_3D, xyz, radial_list, opts.verbose);

                if (future_result.wait_for(std::chrono::milliseconds(60000)) == std::future_status::ready) {

                    estimated_center_mm = future_result.get();
                }
                else {
                    cout << "Accumulator timed out\n\tImage: " << test_img << "\tKeypoint: " << keypoint_count << endl;
                    break;
                }
            }
            catch (const std::exception& e) {
				cout << "Accumulator failed\n";
				break;
			}

            auto acc_end = chrono::high_resolution_clock::now();

            acc_time += chrono::duration_cast<chrono::milliseconds>(acc_end - acc_start).count();

            // Print the number of centers returned and the estimate if verbose option is enabled
            if (opts.verbose) {
                cout << "\tAcc Space Time: " << chrono::duration_cast<chrono::milliseconds>(acc_end - acc_start).count() << "ms" << endl;
                cout << "\tEstimate: " << estimated_center_mm[0] << " " << estimated_center_mm[1] << " " << estimated_center_mm[2] << endl << endl;
            }


            // Define a vector for storing the transformed pointcloud
            Eigen::Vector3d transformed_gt_center_mm_vector(transformed_gt_center_mm(0, 0), transformed_gt_center_mm(0, 1), transformed_gt_center_mm(0, 2));

            // Calculate the offset
            Eigen::Vector3d diff = transformed_gt_center_mm_vector - estimated_center_mm;


            double center_off_mm = diff.norm();


            if (opts.verbose) {
                cout << "Estimated offset: " << center_off_mm << endl << endl;
            }


            // Save the estimation to the centers 
            for (int i = 0; i < 3; i++) {
                estimated_kpts[keypoint_count - 1][i] = estimated_center_mm[i];
            }



            filename_list.push_back(test_img);

            iteration_count++;

            keypoint_count++;

            if (keypoint_count == 4) {
                break;
            }
        }


        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (estimated_kpts[i][j] == 0) {
                    cout << "Error: estimated_kpts is empty" << endl;
                    break;
                }
            }
        }

        const int num_keypoints = 3;

        double RT[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                RT[i][j] = 0;
            }
        }

        double kpts[3][3];
        for (int i = 0; i < num_keypoints; i++) {
            for (int j = 0; j < 3; j++) {
                kpts[i][j] = keypoints[i + 1][j] * 1000;
            }
        }



        // Calculate the RT matrix
        lmshorn(kpts, estimated_kpts, num_keypoints, RT);


        Eigen::MatrixXd RT_matrix = Eigen::MatrixXd::Zero(3, 4);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                RT_matrix(i, j) = RT[i][j];
            }
        }

        matrix xy, xyz_load_est_transformed;



        // Project the estimated position 
        project(xyz_load_matrix * 1000, linemod_K_matrix, RT_matrix, xy, xyz_load_est_transformed);

        // If in demo mode, display the estimated position
        if (opts.demo_mode) {
            int out_of_range = 0;
            cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
            cv::Mat img_only_points;
            img_only_points = cv::Mat::zeros(img.size(), CV_8UC3);

            for (int i = 0; i < xy.rows(); i++) {
                int y = static_cast<int>(xy(i, 1));
                int x = static_cast<int>(xy(i, 0));
                if (0 <= y && y < img.rows && 0 <= x && x < img.cols) {
                    img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
                    img_only_points.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
                }
                else {
                    out_of_range++;
                }
            }
            cv::imshow("Image with point overlay", img);
            cv::waitKey(10);
        }

        if (opts.verbose) {
            cout << string(50, '-') << endl;
        }

        vector<Eigen::Vector3d> pointsGT, pointsEst, pointEst_py, pointEst_gt;

        auto icp_start = chrono::high_resolution_clock::now();
        // Load the ground truth points
        for (int i = 0; i < xyz_load_transformed.rows(); ++i) {
            pointsGT.push_back(1000 * Eigen::Vector3d(xyz_load_transformed(i, 0), xyz_load_transformed(i, 1), xyz_load_transformed(i, 2)));
        }
        // Load the estimated points
        for (int i = 0; i < xyz_load_est_transformed.rows(); ++i) {
            pointsEst.push_back(Eigen::Vector3d(xyz_load_est_transformed(i, 0), xyz_load_est_transformed(i, 1), xyz_load_est_transformed(i, 2)));
        }

        // Create point clouds for the ground truth and estimated points
        geometry::PointCloud sceneGT;
        geometry::PointCloud sceneEst;

        sceneGT.points_ = pointsGT;
        sceneEst.points_ = pointsEst;

        // Paint the point clouds
        //sceneGT.PaintUniformColor({ 0,0,1 });
        ////This line throws errors in debug mode
        //sceneEst.PaintUniformColor({ 1,0,0 }); 

        // Compute the distance between the ground truth and estimated point clouds
        auto distances = sceneGT.ComputePointCloudDistance(sceneEst);
        double distance_bf_icp = accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
        double min_distance_bf_icp = *min_element(distances.begin(), distances.end());
        double max_distance_bf_icp = *max_element(distances.begin(), distances.end());
        double median_distance_bf_icp = distances[distances.size() / 2];
        double standard_deviation_bf_icp = 0.0;
        for (auto& d : distances) {
            standard_deviation_bf_icp += (d - distance_bf_icp) * (d - distance_bf_icp);
        }

        bf_icp_distances.push_back(distance_bf_icp);
        bf_icp_min_distances.push_back(min_distance_bf_icp);
        bf_icp_max_distances.push_back(max_distance_bf_icp);
        bf_icp_median_distances.push_back(median_distance_bf_icp);
        bf_icp_std_deviations.push_back(standard_deviation_bf_icp);


        // If the distance is less than the threshold, increment the number of correct matches
        if (opts.verbose) {
            cout << "Distance Threshold: " << add_threshold[class_name] * 1000 << endl;
            cout << "Distance: " << distance_bf_icp << endl;
        }
        if (distance_bf_icp <= add_threshold[class_name] * 1000) {
            bf_icp += 1;
            if (opts.verbose) {
                cout << "\tICP not needed" << endl;
            }
        }
        else {
            if (opts.verbose) {
                cout << "\tICP needed" << endl;
                
                //cin.get();
                
            }
        }


        // Perform ICP
        Eigen::Matrix4d trans_init = Eigen::Matrix4d::Identity();

        double threshold = distance_bf_icp;

        open3d::pipelines::registration::ICPConvergenceCriteria criteria(2000000);


        auto reg_p2p = open3d::pipelines::registration::RegistrationICP(
            sceneGT, sceneEst, threshold, trans_init,
            open3d::pipelines::registration::TransformationEstimationPointToPoint(),
            criteria);


        sceneGT.Transform(reg_p2p.transformation_);

        distances = sceneGT.ComputePointCloudDistance(sceneEst);

        double distance_af_icp = accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
        double min_distance_af_icp = *min_element(distances.begin(), distances.end());
        double max_distance_af_icp = *max_element(distances.begin(), distances.end());
        double median_distance_af_icp = distances[distances.size() / 2];
        double standard_deviation_af_icp = 0.0;
        for (auto& d : distances) {
            standard_deviation_af_icp += (d - distance_af_icp) * (d - distance_af_icp);
        }

        af_icp_distances.push_back(distance_af_icp);
        af_icp_min_distances.push_back(min_distance_af_icp);
        af_icp_max_distances.push_back(max_distance_af_icp);
        af_icp_median_distances.push_back(median_distance_af_icp);
        af_icp_std_deviations.push_back(standard_deviation_af_icp);


        auto icp_end = chrono::high_resolution_clock::now();
        auto icp_duration = chrono::duration_cast<chrono::milliseconds>(icp_end - icp_start);
        icp_time += icp_duration.count();

        if (opts.verbose) {
            cout << "Distance after ICP: " << distance_af_icp << endl;
        }
        // If the distance is less than the threshold, increment the number of correct matches
        if (distance_af_icp <= add_threshold[class_name] * 1000) {
            af_icp += 1;
            if (opts.verbose) {
                cout << "\tCorrect match after ICP" << endl;
            }

        }
        else {
            if (opts.verbose) {
                cout << "\tIncorrect match after ICP" << endl;
            }
        }

        general_counter += 1;

        auto img_end = chrono::high_resolution_clock::now();
        auto img_duration = chrono::duration_cast<chrono::milliseconds>(img_end - img_start);
        auto sec = (img_duration.count() / 1000.0) - (img_duration.count() % 1000) / 1000.0;
        auto ms = img_duration.count() % 1000;

        if (opts.verbose) {
            double avg_distance_bf_icp = 0;
            double avg_distance_af_icp = 0;
            double avg_min_distance_bf_icp = 0;
            double avg_min_distance_af_icp = 0;
            double avg_max_distance_bf_icp = 0;
            double avg_max_distance_af_icp = 0;
            double avg_median_distance_bf_icp = 0;
            double avg_median_distance_af_icp = 0;
            double avg_std_deviation_bf_icp = 0;
            double avg_std_deviation_af_icp = 0;

            for (double dist : bf_icp_distances) {
                avg_distance_bf_icp += dist;
            }
            avg_distance_bf_icp = avg_distance_bf_icp / bf_icp_distances.size();

            for (double dist : af_icp_distances) {
                avg_distance_af_icp += dist;
            }
            avg_distance_af_icp = avg_distance_af_icp / af_icp_distances.size();

            for (double dist : bf_icp_min_distances) {
                avg_min_distance_bf_icp += dist;
            }
            avg_min_distance_bf_icp = avg_min_distance_bf_icp / bf_icp_min_distances.size();

            for (double dist : af_icp_min_distances) {
                avg_min_distance_af_icp += dist;
            }
            avg_min_distance_af_icp = avg_min_distance_af_icp / af_icp_min_distances.size();
            for (double dist : bf_icp_max_distances) {
                avg_max_distance_bf_icp += dist;
            }
            avg_max_distance_bf_icp = avg_max_distance_bf_icp / bf_icp_max_distances.size();
            for (double dist : af_icp_max_distances) {
                avg_max_distance_af_icp += dist;
            }
            avg_max_distance_af_icp = avg_max_distance_af_icp / af_icp_max_distances.size();
            for (double dist : bf_icp_median_distances) {
                avg_median_distance_bf_icp += dist;
            }
            avg_median_distance_bf_icp = avg_median_distance_bf_icp / bf_icp_median_distances.size();
            for (double dist : af_icp_median_distances) {
                avg_median_distance_af_icp += dist;
            }
            avg_median_distance_af_icp = avg_median_distance_af_icp / af_icp_median_distances.size();
            for (double dist : bf_icp_std_deviations) {
                avg_std_deviation_bf_icp += dist;
            }
            avg_std_deviation_bf_icp = avg_std_deviation_bf_icp / bf_icp_std_deviations.size();
            for (double dist : af_icp_std_deviations) {
                avg_std_deviation_af_icp += dist;
            }
            avg_std_deviation_af_icp = avg_std_deviation_af_icp / af_icp_std_deviations.size();

            ofstream myfile;
            string save_path = "data/img_" + test_img + ".txt";
            myfile.open(save_path, ios::app);
            myfile << "Image number " << test_img << " took " << sec << " seconds and " << ms << " miliseconds to calculate offset." << endl;
            myfile << "Image Count: " << general_counter << endl;
            myfile << "--------------------------------------------" << endl;
            myfile << "Before ICP Count " << bf_icp << endl;
            myfile << "After ICP Count " << af_icp << endl;
            myfile << "Distances: " << endl;
            myfile << "\tBF ICP: " << distance_bf_icp << "\tAF ICP: " << distance_af_icp << endl;
            myfile << "Min Distances: " << endl;
            myfile << "\tBF ICP: " << min_distance_bf_icp << "\tAF ICP: " << min_distance_af_icp << endl;
            myfile << "Max Distances: " << endl;
            myfile << "\tBF ICP: " << max_distance_bf_icp << "\tAF ICP: " << max_distance_af_icp << endl;
            myfile << "Median Distances: " << endl;
            myfile << "\tBF ICP: " << median_distance_bf_icp << "\tAF ICP: " << median_distance_af_icp << endl;
            myfile << "Standard Deviation Distances: " << endl;
            myfile << "\tBF ICP: " << standard_deviation_bf_icp << "\tAF ICP: " << standard_deviation_af_icp << endl;
            myfile << "---------------------------------------------" << endl;
            myfile << "\t\t\tCurrent Averages : " << endl;
            myfile << "\tBefore ICP: " << endl;
            myfile << "\tCurrent Avg Mean Distance before ICP: " << avg_distance_bf_icp << endl;
            myfile << "\tCurrent Avg Median Distance before ICP: " << avg_median_distance_bf_icp << endl;
            myfile << "\tCurrent Avg Min Distance before ICP: " << avg_min_distance_bf_icp << endl;
            myfile << "\tCurrent Avg Max Distance before ICP: " << avg_max_distance_bf_icp << endl;
            myfile << "\tCurrent Avg Standard Deviation Distance before ICP: " << avg_std_deviation_bf_icp << endl;
            myfile << "\tAfter ICP: " << endl;
            myfile << "\tCurrent Avg Mean Distance after ICP: " << avg_distance_af_icp << endl;
            myfile << "\tCurrent Avg Median Distance after ICP: " << avg_median_distance_af_icp << endl;
            myfile << "\tCurrent Avg Min Distance after ICP: " << avg_min_distance_af_icp << endl;
            myfile << "\tCurrent Avg Max Distance after ICP: " << avg_max_distance_af_icp << endl;
            myfile << "\tCurrent Avg Standard Deviation Distance after ICP: " << avg_std_deviation_af_icp << endl;
            myfile << "---------------------------------------------" << endl;


            myfile.close();
            cout << string(75, '-') << endl;
        }

        double avg_correct_bf_icp, avg_correct_af_icp, percent_processed;

        avg_correct_bf_icp = (bf_icp / static_cast<double>(general_counter)) * 100;
        avg_correct_af_icp = (af_icp / static_cast<double>(general_counter)) * 100;
        percent_processed = (static_cast<double>(general_counter) / static_cast<double>(total_num_img)) * 100;

        if (opts.verbose) {
            cout << "Image Count: " << general_counter << endl;
            cout << "Image number " << test_img << " took " << sec << " seconds and " << ms << " miliseconds to calculate offset." << endl;
            cout << "Before ICP Count " << bf_icp << endl;
            cout << "After ICP Count " << af_icp << endl;
            cout << "Before ICP ADDs " << avg_correct_bf_icp << "%" << endl;
            cout << "After ICP ADDs " << avg_correct_af_icp << "%" << endl;
            cout << "Processed: " << percent_processed << "%" << endl;
        }
        else {
            cout << "\r" << string(100, ' ') << "\r";
            cout << "ADD: " << (static_cast<double>(bf_icp)/ static_cast<double>(general_counter)) * 100.0 << "%\t";
            cout << "| " << to_string(general_counter) << "/" << to_string(total_num_img) << " ";
            printProgressBar(general_counter, total_num_img, 50);
 
        }
    }
    cout << endl;



    double total_avg_correct_bf_icp, total_avg_correct_af_icp;

    total_avg_correct_bf_icp = bf_icp / static_cast<double>(general_counter) * 100;
    total_avg_correct_af_icp = af_icp / static_cast<double>(general_counter) * 100;

    auto acc_end = chrono::high_resolution_clock::now();

    cout << "ADD of " << class_name << " before ICP: " << total_avg_correct_bf_icp << endl;
    cout << "ADD of " << class_name << " after ICP: " << total_avg_correct_af_icp << endl;
    auto duration = chrono::duration_cast<chrono::seconds>(acc_end - acc_start);
    auto hours = duration.count() / 3600;
    auto min = (duration.count() % 3600) / 60;
    auto sec = (duration.count() % 3600) % 60;

    cout << "Total Time: " << hours << " hours, " << min << " minutes, and " << sec << " seconds." << endl;
    cout << "Avg Time per image: " << (static_cast<double>(duration.count()) / static_cast<double>(general_counter)) << " seconds." << endl;
    cout << "Avg Accumulator time: " << (static_cast<double>(acc_time) / static_cast<double>(general_counter)) / 1000.0 << " seconds." << endl;
    cout << "Avg Backend Time: " << (static_cast<double>(backend_net_time) / static_cast<double>(general_counter)) / 1000.0 << " seconds." << endl;
    cout << "Avg ICP Time: " << (static_cast<double>(icp_time) / static_cast<double>(general_counter)) / 1000.0 << " seconds." << endl;
    cout << "Avg Input Size to Accumulator Space: " << static_cast<double>(avg_acc_length) / static_cast<double>(general_counter) << endl;
    return;
    
}

void estimate_6d_pose(const Options& opts, DenseFCNResNet152& model, cv::Mat& img, cv::Mat& depth, const vector<vector<double>>& keypoints, const vector<Vertex>& orig_point_cloud )
{
    auto start = chrono::high_resolution_clock::now();

    cv::Mat img_tmp;
    if (opts.demo_mode) {
        img_tmp = img.clone();
    }

    bool use_cuda = torch::cuda::is_available();
    torch::DeviceType device_type = use_cuda ? torch::kCUDA : torch::kCPU;
    torch::Device device(device_type);
    model->to(device);
    model->eval();

    torch::Tensor semantic, radial1, radial2, radial3;

    FCResBackbone(model, img, semantic, radial1, radial2, radial3, device_type, false);

    cv::Mat sem_cv = torch_tensor_to_cv_mat(semantic);
    cv::Mat rad1_cv = torch_tensor_to_cv_mat(radial1);
    cv::Mat rad2_cv = torch_tensor_to_cv_mat(radial2);
    cv::Mat rad3_cv = torch_tensor_to_cv_mat(radial3);

    cv::transpose(sem_cv, sem_cv);
    cv::transpose(rad1_cv, rad1_cv);
    cv::transpose(rad2_cv, rad2_cv);
    cv::transpose(rad3_cv, rad3_cv);

    cv::normalize(sem_cv, sem_cv, 0, 1, cv::NORM_MINMAX);

    cv::Mat threshold;
    cv::threshold(sem_cv, threshold, opts.mask_threshold, 1, cv::THRESH_BINARY);
    threshold.convertTo(sem_cv, sem_cv.type());
    threshold.release();

    cv::Mat sem_tmp, depth_tmp, rad1_tmp, rad2_tmp, rad3_tmp;

    sem_cv.convertTo(sem_tmp, CV_32F);
    depth.convertTo(depth_tmp, CV_32F);
    rad1_cv.convertTo(rad1_tmp, CV_32F);
    rad2_cv.convertTo(rad2_tmp, CV_32F);
    rad3_cv.convertTo(rad3_tmp, CV_32F);

    rad1_tmp = rad1_tmp.mul(sem_tmp);
    rad2_tmp = rad2_tmp.mul(sem_tmp);
    rad3_tmp = rad3_tmp.mul(sem_tmp);
    depth_tmp = depth_tmp.mul(sem_tmp);

    rad1_cv = rad1_tmp.clone();
    rad2_cv = rad2_tmp.clone();
    rad3_cv = rad3_tmp.clone();
    depth = depth_tmp.clone();

    sem_tmp.release();
    depth_tmp.release();
    rad1_tmp.release();
    rad2_tmp.release();

    vector<Vertex> pixel_coor;
    for (int i = 0; i < sem_cv.rows; i++) {
        for (int j = 0; j < sem_cv.cols; j++) {
            if (sem_cv.at<float>(i, j) == 1) {
                Vertex v;
                v.x = i;
                v.y = j;
                v.z = 1;

                pixel_coor.push_back(v);
            }
        }
    }

    vector<double> radial1_list, radial2_list, radial3_list;

    for (auto cord : pixel_coor) {
        radial1_list.push_back(static_cast<double>(rad1_cv.at<float>(cord.x, cord.y)));
        radial2_list.push_back(static_cast<double>(rad2_cv.at<float>(cord.x, cord.y)));
        radial3_list.push_back(static_cast<double>(rad3_cv.at<float>(cord.x, cord.y)));
    }

    vector<Vertex> xyz = rgbd_to_point_cloud(linemod_K, depth);

    depth.release();
    sem_cv.release();
    rad1_cv.release();
    rad2_cv.release();
    rad3_cv.release();

    for (int i = 0; i < xyz.size(); i++) {
        xyz[i].x = xyz[i].x / 1000;
        xyz[i].y = xyz[i].y / 1000;
        xyz[i].z = xyz[i].z / 1000;
    }

  

    Eigen::Vector3d keypoint1 = Accumulator_3D(xyz, radial1_list, false);
    Eigen::Vector3d keypoint2 = Accumulator_3D(xyz, radial2_list, false);
    Eigen::Vector3d keypoint3 = Accumulator_3D(xyz, radial3_list, false);

    double estimated_kpts[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == 0) {
				estimated_kpts[i][j] = keypoint1(j);
			}
            else if (i == 1) {
				estimated_kpts[i][j] = keypoint2(j);
			}
            else {
				estimated_kpts[i][j] = keypoint3(j);
			}
            
        }
    }

    double RT[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            RT[i][j] = 0;
        }
    }


    double kpts[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            kpts[i][j] = keypoints[i + 1][j] * 1000;
        }
    }

    lmshorn(kpts, estimated_kpts, 3, RT);

    Eigen::MatrixXd RT_matrix = Eigen::MatrixXd::Zero(4, 4);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            RT_matrix(i, j) = RT[i][j];
        }
	}

    Eigen::MatrixXd xyz_load_mat = convertToEigenMatrix(orig_point_cloud);
    Eigen::MatrixXd linemod_k_matrix = convertToEigenMatrix(linemod_K);

    Eigen::MatrixXd xy, xyz_load_est;
    
    project(xyz_load_mat * 1000, linemod_k_matrix, RT_matrix, xy, xyz_load_est);


    if (opts.verbose) {
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        auto ms = duration.count() % 1000;
        auto sec = duration.count() / 1000;
        cout << "Took " << sec << " seconds and " << ms << " miliseconds to estimate 6d pose for image." << endl;
    }

    if (opts.demo_mode) {
        int out_of_range = 0;
        for (int i = 0; i < xy.rows(); i++) {
            int y = static_cast<int>(xy(i, 1));
            int x = static_cast<int>(xy(i, 0));
            if (0 <= y && y < img_tmp.rows && 0 <= x && x < img_tmp.cols) {
                img_tmp.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
            }
            else {
                out_of_range++;
            }
		}
        cout << out_of_range << " points out of range." << endl;
        cv::imshow("Image with points overlayed", img_tmp);
        cv::waitKey(100);

    }


}







