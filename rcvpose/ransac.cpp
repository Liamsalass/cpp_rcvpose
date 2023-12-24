#include "ransac.h"

using namespace std;
using namespace Eigen;

struct Vote {
    Sphere s;
    double error = 0;
};


Vector3d centerest(const vector<Sphere>& spheres) {
    assert(spheres.size() >= 4);

    MatrixXd A(spheres.size(), 5);


    for (int i = 0; i < spheres.size(); i++) {
        const Vector3d& p = spheres[i].center;
        double r = spheres[i].radius;

        A(i, 0) = -2 * p(0);
        A(i, 1) = -2 * p(1);
        A(i, 2) = -2 * p(2);
        A(i, 3) = 1;
        A(i, 4) = (p.x()*p.x()) + (p.y()*p.y()) + (p.z()*p.z()) - (r*r);
    }

    Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd X = svd.matrixV().col(4);
    
    X /= X(4);

    return Vector3d(X(0), X(1), X(2));
}




/**
 * Finds the intersection of three spheres and calculates two possible spheres defined
 * by three points of equal radius. This function employs mathematical triangulation technique
 *
 * @param s1, s2, s3: Input spheres with centers and radii.
 * @param result1, result2: Output spheres representing the possible intersections.
 * @param debug: A flag indicating whether to print debug information (default is false).
 *
 * The math and technique involve:
 * 1. Establishing a new coordinate system based on the centers of the input spheres.
 * 2. Calculating direction vectors (ex, ey, and ez) in the new coordinate system.
 * 3. Determining distances and dot products to find x and y, which are used to find
 *    the centers of the two new spheres.
 * 4. Using these centers, calculating B to check if real solutions exist.
 * 5. If solutions exist, computing z and the two possible sphere centers.
 * 6. Finally, calculating the radii of the resulting spheres and returning the results.
 *
 * @return true if solutions are found, false otherwise.
 */
bool findIntersectionSphere(const Sphere& s1, const Sphere& s2, const Sphere& s3, Sphere& result1, Sphere& result2, const bool& debug = false) {
    // Finds the two possible spheres defined by 3 points of equal radius. Returns true if solutions can be found, returns false if not
    Vector3d P1 = s1.center;
    Vector3d P2 = s2.center;
    Vector3d P3 = s3.center;

    // Calculate the direction vectors ex, ey, and ez for a new coordinate system
    Vector3d ex = (P2 - P1).normalized();
    double i = ex.dot(P3 - P1);
    Vector3d ey = (P3 - P1 - i * ex).normalized();
    Vector3d ez = ex.cross(ey);

    // Calculate the distance between P1 and P2 (d) and the dot product between ey and P3-P1 (j)
    double d = (P2 - P1).norm();
    double j = ey.dot(P3 - P1);

    // Calculate x and y, which are used to find the new sphere centers
    double x = (pow(s1.radius, 2) - pow(s2.radius, 2) + pow(d, 2)) / (2 * d);
    double y = (pow(s1.radius, 2) - pow(s3.radius, 2) + pow(i, 2) + pow(j, 2)) / (2 * j) - (i / j) * x;

    // Calculate B, which is used to determine if solutions exist
    double B = pow(s1.radius, 2) - pow(x, 2) - pow(y, 2);

    if (B < 0) {
        // If B is negative, there are no real solutions
        if (debug) {
            int thread = omp_get_thread_num();
            cout << "ERROR Thread[" << thread << "]: No real solution\n";
        }
        return false;
    }

    // Calculate z based on B
    double z = sqrt(B);

    // Calculate two possible solution points for the new sphere centers
    Vector3d sol1 = P1 + x * ex + y * ey + z * ez;
    Vector3d sol2 = P1 + x * ex + y * ey - z * ez;

    // Set the centers and radii of the result spheres
    result1.center = sol1;
    result2.center = sol2;

    double radius1 = (sol1 - s1.center).norm();
    double radius2 = (sol2 - s1.center).norm();

    result1.radius = radius1;
    result2.radius = radius2;

    return true;
}



Vector3d Hash_Vote(const vector<Vertex>& xyz, const vector<double>& radial_list, const double& epsilon, const bool& debug) {
    double acc_unit = 10;
   
    vector<Vertex> xyz_mm(xyz.size());

    #pragma omp parallel for
    for (int i = 0; i < xyz.size(); i++) {
        xyz_mm[i].x = xyz[i].x * 1000 / acc_unit;
        xyz_mm[i].y = xyz[i].y * 1000 / acc_unit;
        xyz_mm[i].z = xyz[i].z * 1000 / acc_unit;
    }

    double x_mean_mm = 0;
    double y_mean_mm = 0;
    double z_mean_mm = 0;

    #pragma omp parallel for
    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mean_mm += xyz_mm[i].x;
        y_mean_mm += xyz_mm[i].y;
        z_mean_mm += xyz_mm[i].z;
    }

    x_mean_mm /= xyz_mm.size();
    y_mean_mm /= xyz_mm.size();
    z_mean_mm /= xyz_mm.size();

    #pragma omp parallel for
    for (int i = 0; i < xyz_mm.size(); i++) {
        xyz_mm[i].x -= x_mean_mm;
        xyz_mm[i].y -= y_mean_mm;
        xyz_mm[i].z -= z_mean_mm;
    }

    vector<double> radial_list_mm(radial_list.size());

    #pragma omp parallel for
    for (int i = 0; i < radial_list.size(); ++i) {
        radial_list_mm[i] = radial_list[i] * 100 / acc_unit;
    }


    double x_mm_min = numeric_limits<double>::infinity();
    double y_mm_min = numeric_limits<double>::infinity();
    double z_mm_min = numeric_limits<double>::infinity();

    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mm_min = min(x_mm_min, xyz_mm[i].x);
        y_mm_min = min(y_mm_min, xyz_mm[i].y);
        z_mm_min = min(z_mm_min, xyz_mm[i].z);
    }

    double xyz_mm_min = min(x_mm_min, min(y_mm_min, z_mm_min));

    double radius_max = radial_list_mm[0];


    for (int i = 0; i < radial_list_mm.size(); i++) {
        if (radius_max < radial_list_mm[i]) {
            radius_max = radial_list_mm[i];
        }
    }

    int zero_boundary = static_cast<int>(xyz_mm_min - radius_max) + 1;

    if (zero_boundary < 0) {
    #pragma omp parallel for
        for (int i = 0; i < xyz_mm.size(); i++) {
            xyz_mm[i].x -= zero_boundary;
            xyz_mm[i].y -= zero_boundary;
            xyz_mm[i].z -= zero_boundary;
        }
    }

    vector<Sphere> sphere_list(xyz.size());


    #pragma omp parallel for
    for (int i = 0; i < xyz_mm.size(); i++) {
        sphere_list[i].center = Vector3d(xyz_mm[i].x, xyz_mm[i].y, xyz_mm[i].z);
        sphere_list[i].radius = radial_list_mm[i];
    }


    unordered_map<int, vector<Sphere>> sphere_map;

    for (int i = 0; i < sphere_list.size(); i++) {
        int key = static_cast<int>(floor(sphere_list[i].radius / epsilon));
        sphere_map[key].push_back(sphere_list[i]);
    }

    int sphere_count = 0, three_count = 0;
    vector<int> keys_w_three;

    for (auto [key, spheres] : sphere_map) {
        int size = spheres.size();
        sphere_count += size;
        if (size >= 3) {
            keys_w_three.push_back(key);
            three_count++;
        }
    }

    if (debug) {
        cout << "\tSpheres : " << sphere_count << endl;
        cout << "\tRadial Levels: " << sphere_map.size() << endl;
        cout << "\tRadial Levels with 3 or more points: " << three_count << endl;
    }
    
    unordered_map<string, unsigned int> point_votes;

    int iterations = 5000;

    #pragma omp parallel for
    for (int i = 0; i < iterations; i++) {

        int keys_w_three_index = rand() % keys_w_three.size();

        int key_index = keys_w_three[keys_w_three_index];

    
        vector<Sphere> spheres = sphere_map[key_index];

        if (spheres.size() < 3) {
            cerr << "Error: Unexpected spheres size in iteration: " << i << ". Size: " << spheres.size() << endl;
            continue;
        }


        if (spheres.size() < 3) {
            if (debug) {
                cout << "Error in Hash Vote: sphere size less than 3\n";
                cout << "\t\tKey: " << key_index << "\n";
                cout << "\t\tSize: " << sphere_map[key_index].size();
            }
            continue;
        }

        int p1_index, p2_index, p3_index, p4_index;

        p1_index = rand() % spheres.size();

        do {
            p2_index = rand() % spheres.size();
        } while (p2_index == p1_index);

        do {
            p3_index = rand() % spheres.size();
        } while (p3_index == p1_index || p3_index == p2_index);

        if (p1_index >= spheres.size() || p2_index >= spheres.size() || p3_index >= spheres.size()) {
            cerr << "Error: Sphere index out of bounds in iteration: " << i << endl;
            continue;
        }


        Sphere p1 = spheres[p1_index];
        Sphere p2 = spheres[p2_index];
        Sphere p3 = spheres[p3_index];
        Sphere r1, r2;

        if (!findIntersectionSphere(p1, p2, p3, r1, r2)) {
            continue;
        }


        int vote1 = 0, vote2 = 0;
        for (auto [key, spheres] : sphere_map) {
            for (const Sphere& p : spheres) {
                double dist1 = (r1.center - p.center).norm();
                double dist2 = (r2.center - p.center).norm();
                if (abs(dist1 - p.radius) <= epsilon) {
                    vote1++;
                }
                if (abs(dist2 - p.radius) <= epsilon) {
                    vote2++;
                }
            }
        }

        Sphere best_center;
        if (vote1 > vote2) {
            best_center = r1;
        }
        else {
            best_center = r2;
        }

        best_center.center[0] = round(best_center.center[0]);
        best_center.center[1] = round(best_center.center[1]);
        best_center.center[2] = round(best_center.center[2]);

        stringstream ss;
        ss << fixed << setprecision(4) << best_center.center[0] << "_" << best_center.center[1] << "_" << best_center.center[2];
        string center_string = ss.str();

        #pragma omp critical 
        {
            if (point_votes.find(center_string) == point_votes.end()) {
                point_votes[center_string] = 1;
            }
            else {
                point_votes[center_string]++;
            }
        }
    }



    if (point_votes.size() == 0) {
        cerr << "Hash Vote failed: no center found" << endl;
        return Vector3d(0, 0, 0);
    }

    unsigned int max_vote = 0;
    string max_vote_center;

    for (auto [center, vote] : point_votes) {
        if (vote > max_vote) {
            max_vote = vote;
            max_vote_center = center;
        }
    }

    if (debug) {
        cout << "\tHash max vote: " << max_vote << endl;
    }


    stringstream ss(max_vote_center);
    string token;
    vector<double> center(3);

    int underscore_count = std::count(max_vote_center.begin(), max_vote_center.end(), '_');
    if (underscore_count != 2) {
        cerr << "Error: max_vote_center format is unexpected. Value: " << max_vote_center << endl;
        return Vector3d(0, 0, 0);
    }
    int i = 0;

    while (getline(ss, token, '_')) {
        center[i] = stod(token);
        i++;
    }

    if (debug) {
        cout << "\tUnshifted Center: [" << center[0] << ", " << center[1] << ", " << center[2] << "]\n";
    }

    Vector3d center_vec(center[0], center[1], center[2]);

    if (zero_boundary < 0) {
        center_vec.array() += zero_boundary;
    }

    center_vec[0] = (center_vec[0] + x_mean_mm + 0.5) * acc_unit;
    center_vec[1] = (center_vec[1] + y_mean_mm + 0.5) * acc_unit;
    center_vec[2] = (center_vec[2] + z_mean_mm + 0.5) * acc_unit;

    return center_vec;
}





Vector3d Ransac_3D(const vector<Vertex>& xyz, const vector<double>& radial_list, const double& epsilon, const bool& debug) {
    double acc_unit = 10;

    if (debug) {
        cout << "\tEpsilon: " << epsilon << endl;
        cout << "\tAccuracy Unit: " << acc_unit << endl;
    }

    vector<Vertex> xyz_mm(xyz.size());

    #pragma omp parallel for
    for (int i = 0; i < xyz.size(); i++) {
        xyz_mm[i].x = xyz[i].x * 1000 / acc_unit;
        xyz_mm[i].y = xyz[i].y * 1000 / acc_unit;
        xyz_mm[i].z = xyz[i].z * 1000 / acc_unit;
    }

    double x_mean_mm = 0;
    double y_mean_mm = 0;
    double z_mean_mm = 0;

    #pragma omp parallel for
    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mean_mm += xyz_mm[i].x;
        y_mean_mm += xyz_mm[i].y;
        z_mean_mm += xyz_mm[i].z;
    }

    x_mean_mm /= xyz_mm.size();
    y_mean_mm /= xyz_mm.size();
    z_mean_mm /= xyz_mm.size();

    #pragma omp parallel for
    for (int i = 0; i < xyz_mm.size(); i++) {
        xyz_mm[i].x -= x_mean_mm;
        xyz_mm[i].y -= y_mean_mm;
        xyz_mm[i].z -= z_mean_mm;
    }

    vector<double> radial_list_mm(radial_list.size());

    #pragma omp parallel for
    for (int i = 0; i < radial_list.size(); ++i) {
        radial_list_mm[i] = radial_list[i] * 100 / acc_unit;
    }


    double x_mm_min = numeric_limits<double>::infinity();
    double y_mm_min = numeric_limits<double>::infinity();
    double z_mm_min = numeric_limits<double>::infinity();

    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mm_min = min(x_mm_min, xyz_mm[i].x);
        y_mm_min = min(y_mm_min, xyz_mm[i].y);
        z_mm_min = min(z_mm_min, xyz_mm[i].z);
    }

    double xyz_mm_min = min(x_mm_min, min(y_mm_min, z_mm_min));

    double radius_max = radial_list_mm[0];


    for (int i = 0; i < radial_list_mm.size(); i++) {
        if (radius_max < radial_list_mm[i]) {
            radius_max = radial_list_mm[i];
        }
    }

    int zero_boundary = static_cast<int>(xyz_mm_min - radius_max) + 1;

    if (zero_boundary < 0) {
    #pragma omp parallel for
        for (int i = 0; i < xyz_mm.size(); i++) {
            xyz_mm[i].x -= zero_boundary;
            xyz_mm[i].y -= zero_boundary;
            xyz_mm[i].z -= zero_boundary;
        }
    }

    vector<Sphere> sphere_list(xyz.size());

    vector<Vector3d> sphere_centers(xyz.size());

    #pragma omp parallel for
    for (int i = 0; i < xyz_mm.size(); i++) {
        Vector3d center(xyz_mm[i].x, xyz_mm[i].y, xyz_mm[i].z);
        sphere_centers[i] = center;
        sphere_list[i].center = center;
        sphere_list[i].radius = radial_list_mm[i];
    }


    unordered_map<int, vector<Sphere>> sphere_map;

    for (int i = 0; i < sphere_list.size(); i++) {
        int key = static_cast<int>(floor(sphere_list[i].radius / epsilon));
        sphere_map[key].push_back(sphere_list[i]);
    }

    sphere_list.clear();

    int sphere_count = 0, three_count = 0;
    vector<int> keys_w_three;

    for (auto [key, spheres] : sphere_map) {
        int size = spheres.size();
        sphere_count += size;
        if (size >= 3) {
            keys_w_three.push_back(key);
            three_count++;
        }
    }

    if (debug) {
        cout << "\tSpheres : " << sphere_count << endl;
        cout << "\tRadial Levels: " << sphere_map.size() << endl;
        cout << "\tRadial Levels with 3 or more points: " << three_count << endl;
    }

    if (three_count == 0) {
        return Vector3d(0, 0, 0);
    }

    //unordered_map<string, unsigned int> sphere_votes;
    vector<Vote> sphere_votes;


    int iterations = 100;
    
    #pragma omp parallel for
    for (int i = 0; i < iterations; i++) {
        assert (keys_w_three.size() > 0);
    
        int keys_w_three_index = rand() % keys_w_three.size();
    
        int key_index = keys_w_three[keys_w_three_index];
    
        vector<Sphere> spheres = sphere_map[key_index];
    
        if (spheres.size() < 3) {
            if (debug) {
                cout << "Error in RANSAC: sphere size less than 3\n";
            }
            continue;
        }
    
        int p1_index, p2_index, p3_index;
    
        p1_index = rand() % spheres.size();
    
        do {
            p2_index = rand() % spheres.size();
        } while (p2_index == p1_index);
    
        do {
            p3_index = rand() % spheres.size();
        } while (p3_index == p1_index || p3_index == p2_index);
    
    
     
        Sphere p1 = spheres[p1_index];
        Sphere p2 = spheres[p2_index];
        Sphere p3 = spheres[p3_index];
        Sphere r1, r2;
     
        if (!findIntersectionSphere(p1, p2, p3, r1, r2)) {
            continue;
        }
        
     
        double avg1 = 0, avg2 = 0;
        for (auto [key, spheres] : sphere_map) {
            for (const Sphere& p : spheres) {
                double dist1 = (r1.center - p.center).norm();
                double dist2 = (r2.center - p.center).norm();

                avg1 += abs(dist1 - p.radius);
                avg2 += abs(dist2 - p.radius);
            }
		}

        avg1 /= sphere_count;
        avg2 /= sphere_count;
        
        Vote v1, v2;
        v1.s = r1;
        v2.s = r2;
        v1.error = avg1;
        v2.error = avg2;
    
        #pragma omp critical
        {
            sphere_votes.push_back(v1);
            sphere_votes.push_back(v2);
        }
    }

    if (sphere_votes.size() == 0) {
        cerr << "RANSAC failed: no center found" << endl;
        return Vector3d(0, 0, 0);
    }

    sort(sphere_votes.begin(), sphere_votes.end(), [](const Vote& v1, const Vote& v2) {return v1.error < v2.error; });
    
    Sphere max_vote_sphere = sphere_votes[0].s;
    double avg_error1 = sphere_votes[0].error;

    Vector3d vote_center = max_vote_sphere.center;

    if (debug) {
        cout << "\tCenter through avg error: [" << vote_center[0] << ", " << vote_center[1] << ", " << vote_center[2] << "], Average Error: " << avg_error1 << "\n";
    }

    vector<Sphere> spheres_no_outliers;
    

    for (auto [key, spheres] : sphere_map) {
        for (const Sphere& s : spheres) {
			double dist = (vote_center - s.center).norm();
            if (abs(dist - s.radius) < avg_error1) {
                spheres_no_outliers.push_back(s);
			} 
		}
	}
    
    if (debug) {
        cout << "\tRemaining non-outlier spheres: " << spheres_no_outliers.size() << endl;
    }

    double x_shift = 0, y_shift = 0, z_shift = 0;

    
    for (int i = 0; i < spheres_no_outliers.size(); i++) {
        const Sphere& s = spheres_no_outliers[i];
        double x_dist = s.center[0] - vote_center[0];
        double y_dist = s.center[1] - vote_center[1];
        double z_dist = s.center[2] - vote_center[2];
        double radius = s.radius; 

        double dist = sqrt(x_dist * x_dist + y_dist * y_dist + z_dist * z_dist);

        #pragma omp critical 
        {
            x_shift += x_dist * (radius - dist) / dist;
            y_shift += y_dist * (radius - dist) / dist;
            z_shift += z_dist * (radius - dist) / dist;
        }
	}
	
    x_shift /= spheres_no_outliers.size();
    y_shift /= spheres_no_outliers.size();
    z_shift /= spheres_no_outliers.size();

    Vector3d new_centerest(vote_center[0] + x_shift, vote_center[1] + y_shift, vote_center[2] + z_shift);
	
    double avg_error2 = 0;
    for (auto [key, spheres] : sphere_map) {
        for (const Sphere& sphere : spheres) {
            avg_error2 += abs((sphere.center - new_centerest).norm() - sphere.radius);
        }
    }

    sphere_map.clear();
    avg_error2 /= sphere_count;

    if (debug) {
        cout << "\tNew Center: [" << new_centerest[0] << ", " << new_centerest[1] << ", " << new_centerest[2] << ", Average Error: " << avg_error2 << "\n";
    }

    if (zero_boundary < 0) {
        new_centerest.array() += zero_boundary;
    }

    new_centerest[0] = (new_centerest[0] + x_mean_mm + 0.5) * acc_unit;
    new_centerest[1] = (new_centerest[1] + y_mean_mm + 0.5) * acc_unit;
    new_centerest[2] = (new_centerest[2] + z_mean_mm + 0.5) * acc_unit;

    return new_centerest;
}

Eigen::Vector3d Ransac_3D_greenspan(const std::vector<Vertex>& xyz, const std::vector<double>& radial_list, const double& epsilon, const bool& debug)
{
    double acc_unit = 10;

    if (debug) {
        cout << "\tEpsilon: " << epsilon << endl;
        cout << "\tAccuracy Unit: " << acc_unit << endl;
    }

    vector<Vertex> xyz_mm(xyz.size());

    #pragma omp parallel for
    for (int i = 0; i < xyz.size(); i++) {
        xyz_mm[i].x = xyz[i].x * 1000 / acc_unit;
        xyz_mm[i].y = xyz[i].y * 1000 / acc_unit;
        xyz_mm[i].z = xyz[i].z * 1000 / acc_unit;
    }

    double x_mean_mm = 0;
    double y_mean_mm = 0;
    double z_mean_mm = 0;

    #pragma omp parallel for
    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mean_mm += xyz_mm[i].x;
        y_mean_mm += xyz_mm[i].y;
        z_mean_mm += xyz_mm[i].z;
    }

    x_mean_mm /= xyz_mm.size();
    y_mean_mm /= xyz_mm.size();
    z_mean_mm /= xyz_mm.size();

    #pragma omp parallel for
    for (int i = 0; i < xyz_mm.size(); i++) {
        xyz_mm[i].x -= x_mean_mm;
        xyz_mm[i].y -= y_mean_mm;
        xyz_mm[i].z -= z_mean_mm;
    }

    vector<double> radial_list_mm(radial_list.size());

    #pragma omp parallel for
    for (int i = 0; i < radial_list.size(); ++i) {
        radial_list_mm[i] = radial_list[i] * 100 / acc_unit;
    }


    double x_mm_min = numeric_limits<double>::infinity();
    double y_mm_min = numeric_limits<double>::infinity();
    double z_mm_min = numeric_limits<double>::infinity();

    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mm_min = min(x_mm_min, xyz_mm[i].x);
        y_mm_min = min(y_mm_min, xyz_mm[i].y);
        z_mm_min = min(z_mm_min, xyz_mm[i].z);
    }

    double xyz_mm_min = min(x_mm_min, min(y_mm_min, z_mm_min));

    double radius_max = radial_list_mm[0];


    for (int i = 0; i < radial_list_mm.size(); i++) {
        if (radius_max < radial_list_mm[i]) {
            radius_max = radial_list_mm[i];
        }
    }

    int zero_boundary = static_cast<int>(xyz_mm_min - radius_max) + 1;

    if (zero_boundary < 0) {
    #pragma omp parallel for
        for (int i = 0; i < xyz_mm.size(); i++) {
            xyz_mm[i].x -= zero_boundary;
            xyz_mm[i].y -= zero_boundary;
            xyz_mm[i].z -= zero_boundary;
        }
    }

    vector<Sphere> sphere_list(xyz.size());

    vector<Vector3d> sphere_centers(xyz.size());

    #pragma omp parallel for
    for (int i = 0; i < xyz_mm.size(); i++) {
        Vector3d center(xyz_mm[i].x, xyz_mm[i].y, xyz_mm[i].z);
        sphere_centers[i] = center;
        sphere_list[i].center = center;
        sphere_list[i].radius = radial_list_mm[i];
    }



    if (debug) {
        cout << "\tSpheres : " << sphere_list.size() << endl;

    }

    if (sphere_list.size() < 3) {
        if (debug) {
            cout << "Error in RANSAC: sphere size less than 3\n";
        }
        return Vector3d(0, 0, 0);
    }


    vector<Vote> sphere_votes;


    int iterations = 200;

    #pragma omp parallel for
    for (int i = 0; i < iterations; i++) {
  
        int p1_index, p2_index, p3_index, p4_index;

        p1_index = rand() % sphere_list.size();

        do {
            p2_index = rand() % sphere_list.size();
        } while (p2_index == p1_index);

        do {
            p3_index = rand() % sphere_list.size();
        } while (p3_index == p1_index || p3_index == p2_index);

        do {
            p4_index = rand() % sphere_list.size();
        } while (p4_index == p1_index || p4_index == p2_index || p4_index == p3_index);


        vector<Sphere> tmp_list(4);
        tmp_list[0] = sphere_list[p1_index];
        tmp_list[1] = sphere_list[p2_index];
        tmp_list[2] = sphere_list[p3_index];
        tmp_list[3] = sphere_list[p4_index];

        Vector3d calculated_center = centerest(tmp_list);

        Vote vote;

        vote.s.center = calculated_center;

        double avg_error = 0;
        for (const Sphere& p : sphere_list) {
            avg_error += abs((calculated_center - p.center).norm() - p.radius);
        }

        avg_error /= sphere_list.size();

        vote.error = avg_error;

        #pragma omp critical
        {
            sphere_votes.push_back(vote);
        }
    }

    if (sphere_votes.size() == 0) {
        cerr << "RANSAC failed: no center found" << endl;
        return Vector3d(0, 0, 0);
    }


    sort(sphere_votes.begin(), sphere_votes.end(), [](const Vote& v1, const Vote& v2) {return v1.error < v2.error; });

    Sphere max_vote_sphere = sphere_votes[0].s;
    double avg_error1 = sphere_votes[0].error;

    Vector3d vote_center = max_vote_sphere.center;

    if (debug) {
        cout << "\tCenter Before Shift: [" << vote_center[0] << ", " << vote_center[1] << ", " << vote_center[2] << "], Average Error: " << avg_error1 << "\n";
    }

    vector<Sphere> spheres_no_outliers;

    for (const Sphere& p : sphere_list) {
        double dist = (vote_center - p.center).norm();
        if (abs(dist - p.radius) < avg_error1) {
			spheres_no_outliers.push_back(p);
		}
    }


    if (debug) {
		cout << "\tRemaining non-outlier spheres: " << spheres_no_outliers.size() << endl;
	}

    Vector3d new_centerest = centerest(spheres_no_outliers);    

    double avg_error2 = 0;

    for (const Sphere& p : sphere_list) {
        avg_error2 += abs((new_centerest - p.center).norm() - p.radius);	
    }

    avg_error2 /= sphere_list.size();

    if (debug) {
        cout << "\tNew Center: [" << new_centerest[0] << ", " << new_centerest[1] << ", " << new_centerest[2] << ", Average Error: " << avg_error2 << "\n";
    }


    if (zero_boundary < 0) {
        new_centerest.array() += zero_boundary;
    }

    new_centerest[0] = (new_centerest[0] + x_mean_mm + 0.5) * acc_unit;
    new_centerest[1] = (new_centerest[1] + y_mean_mm + 0.5) * acc_unit;
    new_centerest[2] = (new_centerest[2] + z_mean_mm + 0.5) * acc_unit;

    return new_centerest;
}



Vector3d Ransac_Accumulator(const std::vector<Vertex>& xyz, const std::vector<double>& radial_list, const double& epsilon, const bool& debug)
{
    double acc_unit = 10;

    if (debug) {
        cout << "\tEpsilon: " << epsilon << endl;
        cout << "\tAccuracy Unit: " << acc_unit << endl;
    }

    vector<Vertex> xyz_mm(xyz.size());

#pragma omp parallel for
    for (int i = 0; i < xyz.size(); i++) {
        xyz_mm[i].x = xyz[i].x * 1000 / acc_unit;
        xyz_mm[i].y = xyz[i].y * 1000 / acc_unit;
        xyz_mm[i].z = xyz[i].z * 1000 / acc_unit;
    }

    double x_mean_mm = 0;
    double y_mean_mm = 0;
    double z_mean_mm = 0;

#pragma omp parallel for
    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mean_mm += xyz_mm[i].x;
        y_mean_mm += xyz_mm[i].y;
        z_mean_mm += xyz_mm[i].z;
    }

    x_mean_mm /= xyz_mm.size();
    y_mean_mm /= xyz_mm.size();
    z_mean_mm /= xyz_mm.size();

#pragma omp parallel for
    for (int i = 0; i < xyz_mm.size(); i++) {
        xyz_mm[i].x -= x_mean_mm;
        xyz_mm[i].y -= y_mean_mm;
        xyz_mm[i].z -= z_mean_mm;
    }

    vector<double> radial_list_mm(radial_list.size());

#pragma omp parallel for
    for (int i = 0; i < radial_list.size(); ++i) {
        radial_list_mm[i] = radial_list[i] * 100 / acc_unit;
    }


    double x_mm_min = numeric_limits<double>::infinity();
    double y_mm_min = numeric_limits<double>::infinity();
    double z_mm_min = numeric_limits<double>::infinity();

    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mm_min = min(x_mm_min, xyz_mm[i].x);
        y_mm_min = min(y_mm_min, xyz_mm[i].y);
        z_mm_min = min(z_mm_min, xyz_mm[i].z);
    }

    double xyz_mm_min = min(x_mm_min, min(y_mm_min, z_mm_min));

    double radius_max = radial_list_mm[0];


    for (int i = 0; i < radial_list_mm.size(); i++) {
        if (radius_max < radial_list_mm[i]) {
            radius_max = radial_list_mm[i];
        }
    }

    int zero_boundary = static_cast<int>(xyz_mm_min - radius_max) + 1;

    if (zero_boundary < 0) {
#pragma omp parallel for
        for (int i = 0; i < xyz_mm.size(); i++) {
            xyz_mm[i].x -= zero_boundary;
            xyz_mm[i].y -= zero_boundary;
            xyz_mm[i].z -= zero_boundary;
        }
    }

    vector<Sphere> sphere_list(xyz.size());

    vector<Vector3d> sphere_centers(xyz.size());

#pragma omp parallel for
    for (int i = 0; i < xyz_mm.size(); i++) {
        Vector3d center(xyz_mm[i].x, xyz_mm[i].y, xyz_mm[i].z);
        sphere_centers[i] = center;
        sphere_list[i].center = center;
        sphere_list[i].radius = radial_list_mm[i];
    }


    unordered_map<int, vector<Sphere>> sphere_map;

    for (int i = 0; i < sphere_list.size(); i++) {
        int key = static_cast<int>(floor(sphere_list[i].radius / epsilon));
        sphere_map[key].push_back(sphere_list[i]);
    }

    sphere_list.clear();

    int sphere_count = 0, three_count = 0;
    vector<int> keys_w_three;

    for (auto [key, spheres] : sphere_map) {
        int size = spheres.size();
        sphere_count += size;
        if (size >= 3) {
            keys_w_three.push_back(key);
            three_count++;
        }
    }

    if (debug) {
        cout << "\tSpheres : " << sphere_count << endl;
        cout << "\tRadial Levels: " << sphere_map.size() << endl;
        cout << "\tRadial Levels with 3 or more points: " << three_count << endl;
    }

    if (three_count == 0) {
        return Vector3d(0,0,0);
    }

    //unordered_map<string, unsigned int> sphere_votes;
    vector<Vote> sphere_votes;


    int iterations = 100;

#pragma omp parallel for
    for (int i = 0; i < iterations; i++) {
        assert(keys_w_three.size() > 0);

        int keys_w_three_index = rand() % keys_w_three.size();

        int key_index = keys_w_three[keys_w_three_index];

        vector<Sphere> spheres = sphere_map[key_index];

        if (spheres.size() < 3) {
            if (debug) {
                cout << "Error in RANSAC: sphere size less than 3\n";
            }
            continue;
        }

        int p1_index, p2_index, p3_index;

        p1_index = rand() % spheres.size();

        do {
            p2_index = rand() % spheres.size();
        } while (p2_index == p1_index);

        do {
            p3_index = rand() % spheres.size();
        } while (p3_index == p1_index || p3_index == p2_index);



        Sphere p1 = spheres[p1_index];
        Sphere p2 = spheres[p2_index];
        Sphere p3 = spheres[p3_index];
        Sphere r1, r2;

        if (!findIntersectionSphere(p1, p2, p3, r1, r2)) {
            continue;
        }


        double avg1 = 0, avg2 = 0;
        for (auto [key, spheres] : sphere_map) {
            for (const Sphere& p : spheres) {
                double dist1 = (r1.center - p.center).norm();
                double dist2 = (r2.center - p.center).norm();

                avg1 += abs(dist1 - p.radius);
                avg2 += abs(dist2 - p.radius);
            }
        }

        avg1 /= sphere_count;
        avg2 /= sphere_count;

        Vote v1, v2;
        v1.s = r1;
        v2.s = r2;
        v1.error = avg1;
        v2.error = avg2;

    #pragma omp critical
        {
            sphere_votes.push_back(v1);
            sphere_votes.push_back(v2);
        }
    }

    if (sphere_votes.size() == 0) {
        cerr << "RANSAC failed: no center found" << endl;
        return Vector3d(0, 0, 0);
    }

    sort(sphere_votes.begin(), sphere_votes.end(), [](const Vote& v1, const Vote& v2) {return v1.error < v2.error; });

    Sphere max_vote_sphere = sphere_votes[0].s;
    double avg_error1 = sqrt(sphere_votes[0].error);

    Vector3d vote_center = max_vote_sphere.center;

    if (debug) {
        cout << "\tCenter through avg error: [" << vote_center[0] << ", " << vote_center[1] << ", " << vote_center[2] << "], Average Error: " << avg_error1 << "\n";
    }

    vector<Sphere> spheres_no_outliers;

    for (auto [key, spheres] : sphere_map) {
        for (const Sphere& s : spheres) {
            double dist = (vote_center - s.center).norm();
            if (abs(dist - s.radius) < avg_error1) {
                spheres_no_outliers.push_back(s);
            }
        }
    }

    if (spheres_no_outliers.size() == 0) {
        return Vector3d(0, 0, 0);
    }

    if (zero_boundary < 0) {
        for (int i = 0; i < spheres_no_outliers.size(); i++) {
			spheres_no_outliers[i].center.array() += zero_boundary;
		}
    }

    for (int i = 0; i < spheres_no_outliers.size(); i++) {
		spheres_no_outliers[i].center[0] = (spheres_no_outliers[i].center[0] + x_mean_mm) * acc_unit / 1000;
		spheres_no_outliers[i].center[1] = (spheres_no_outliers[i].center[1] + y_mean_mm) * acc_unit / 1000;
		spheres_no_outliers[i].center[2] = (spheres_no_outliers[i].center[2] + z_mean_mm) * acc_unit / 1000;
        spheres_no_outliers[i].radius = spheres_no_outliers[i].radius * acc_unit / 100;
	}

    if (debug) {
        cout << "\tRemaining non-outlier spheres: " << spheres_no_outliers.size() << endl;
    }

    vector<Vertex> new_xyz(spheres_no_outliers.size());
    vector<double> new_radial_list(spheres_no_outliers.size());


    for (int i = 0; i < spheres_no_outliers.size(); i++) {
        new_xyz[i].x = spheres_no_outliers[i].center[0];
        new_xyz[i].y = spheres_no_outliers[i].center[1];
        new_xyz[i].z = spheres_no_outliers[i].center[2];
        new_radial_list[i] = spheres_no_outliers[i].radius;
    }

    Vector3d new_centerest = Accumulator_3D(new_xyz, new_radial_list, debug);
    
    return new_centerest;
}



vector<Vote> random_centerest(const vector<Sphere>& sphere_list, const int& iterations, const bool& debug)
{
	vector<Vote> sphere_votes;

    #pragma omp parallel for 
    for (int i = 0; i < iterations; i++) {
		int p1_index, p2_index, p3_index, p4_index;

		p1_index = rand() % sphere_list.size();

        do {
			p2_index = rand() % sphere_list.size();
		} while (p2_index == p1_index);

        do {
			p3_index = rand() % sphere_list.size();
		} while (p3_index == p1_index || p3_index == p2_index); 

        do {
            p4_index = rand() % sphere_list.size();
        } while (p4_index == p1_index || p4_index == p2_index || p4_index == p3_index);

    
        vector<Sphere> tmp_list = { sphere_list[p1_index], sphere_list[p2_index], sphere_list[p3_index], sphere_list[p4_index] };

        Vector3d est = centerest(tmp_list);

        Vote vote;
        double total_error = 0;
        for (int j = 0; j < sphere_list.size(); j++) {
			double dist = (est - sphere_list[j].center).norm();
            total_error += abs(dist - sphere_list[j].radius);

		}
		vote.error = total_error / sphere_list.size();
		vote.s.center = est;
   
        #pragma omp critical 
        {
            sphere_votes.push_back(vote);
        }
    }

    sort(sphere_votes.begin(), sphere_votes.end(), [](const Vote& v1, const Vote& v2) {return v1.error < v2.error; });

    return sphere_votes;
}



vector<Sphere> accumulate_inliers(vector<Sphere>& sphere_list, const Vote& best_vote) {
    vector<Sphere> inliers;

    #pragma omp parallel for 
    for (int i = 0; i < 200; i++) {
        int rand_sphere_idx = rand() % sphere_list.size();
        const Sphere& s = sphere_list[rand_sphere_idx];

        double dist = (best_vote.s.center - s.center).norm();

        if (abs(dist - s.radius) < best_vote.error) {
			#pragma omp critical
            {
				inliers.push_back(s);
			}
		}

    }

    return inliers;
}




Vector3d RANSAC_3D_3(const std::vector<Vertex>& xyz, const std::vector<double>& radial_list, const int& iterations, const bool& debug)
{
	double acc_unit = 10;
    double itr_split = 0.66;

    int first_itr = static_cast<int>(iterations * itr_split);
    int second_itr = iterations - first_itr;


    if (debug) {
        cout << "\tFirst Iteration Count: " << first_itr << endl;
        cout << "\tSecond Iteration Count: " << second_itr << endl;
    }

    vector<Vertex> xyz_mm(xyz.size());

    #pragma omp parallel for
    for (int i = 0; i < xyz.size(); i++) {
        xyz_mm[i].x = xyz[i].x * 1000 / acc_unit;
        xyz_mm[i].y = xyz[i].y * 1000 / acc_unit;
        xyz_mm[i].z = xyz[i].z * 1000 / acc_unit;
    }

    double x_mean_mm = 0;
    double y_mean_mm = 0;
    double z_mean_mm = 0;

    #pragma omp parallel for
    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mean_mm += xyz_mm[i].x;
        y_mean_mm += xyz_mm[i].y;
        z_mean_mm += xyz_mm[i].z;
    }

    x_mean_mm /= xyz_mm.size();
    y_mean_mm /= xyz_mm.size();
    z_mean_mm /= xyz_mm.size();

    #pragma omp parallel for
    for (int i = 0; i < xyz_mm.size(); i++) {
        xyz_mm[i].x -= x_mean_mm;
        xyz_mm[i].y -= y_mean_mm;
        xyz_mm[i].z -= z_mean_mm;
    }

    vector<double> radial_list_mm(radial_list.size());

    #pragma omp parallel for
    for (int i = 0; i < radial_list.size(); ++i) {
        radial_list_mm[i] = radial_list[i] * 100 / acc_unit;
    }


    double x_mm_min = numeric_limits<double>::infinity();
    double y_mm_min = numeric_limits<double>::infinity();
    double z_mm_min = numeric_limits<double>::infinity();

    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mm_min = min(x_mm_min, xyz_mm[i].x);
        y_mm_min = min(y_mm_min, xyz_mm[i].y);
        z_mm_min = min(z_mm_min, xyz_mm[i].z);
    }

    double xyz_mm_min = min(x_mm_min, min(y_mm_min, z_mm_min));

    double radius_max = radial_list_mm[0];


    for (int i = 0; i < radial_list_mm.size(); i++) {
        if (radius_max < radial_list_mm[i]) {
            radius_max = radial_list_mm[i];
        }
    }

    int zero_boundary = static_cast<int>(xyz_mm_min - radius_max) + 1;

    if (zero_boundary < 0) {
    #pragma omp parallel for
        for (int i = 0; i < xyz_mm.size(); i++) {
            xyz_mm[i].x -= zero_boundary;
            xyz_mm[i].y -= zero_boundary;
            xyz_mm[i].z -= zero_boundary;
        }
    }

    vector<Sphere> sphere_list(xyz.size());

    vector<Vector3d> sphere_centers(xyz.size());

    #pragma omp parallel for
    for (int i = 0; i < xyz_mm.size(); i++) {
        Vector3d center(xyz_mm[i].x, xyz_mm[i].y, xyz_mm[i].z);
        sphere_centers[i] = center;
        sphere_list[i].center = center;
        sphere_list[i].radius = radial_list_mm[i];
    }

    if (debug) {
        cout << "\tSpheres : " << sphere_list.size() << endl;
    }

    if (sphere_list.size() < 3) {
        if (debug) {
            cout << "Error in RANSAC: sphere size less than 3\n";
        }
        return Vector3d(0, 0, 0);
    }

    vector<Vote> sorted_vote_list = random_centerest(sphere_list, first_itr, debug);

    Vote best_vote = sorted_vote_list[0];

    if (debug) {
        cout << "\tBest Vote: [" << best_vote.s.center[0] << ", " << best_vote.s.center[1] << ", " << best_vote.s.center[2] << "], Average Error: " << best_vote.error << "\n";
    }

    vector<Sphere> inlier_list = accumulate_inliers(sphere_list, best_vote);

    int inlier_count = inlier_list.size();

    if (debug) {
        cout << "\tInlier Count: " << inlier_count << endl;
    }

    Vector3d center = { 0, 0, 0 };

    if (inlier_count > 4) {
        sorted_vote_list = random_centerest(inlier_list, second_itr, debug);
        if (debug) {
            cout << "\tSecond Iteration Best Vote: [" << sorted_vote_list[0].s.center[0] << ", " << sorted_vote_list[0].s.center[1] << ", " << sorted_vote_list[0].s.center[2] << "], Average Error: " << sorted_vote_list[0].error << "\n";
        }
        center = sorted_vote_list[0].s.center;
    }
    else if (inlier_count == 4) {
        center = centerest(inlier_list);
        if (debug) {
            cout << "\tCenter through centerest: [" << center[0] << ", " << center[1] << ", " << center[2] << "]\n";
        }
    }
    else {
        center = best_vote.s.center;
        if (debug) {
            cout << "\tCenter through best vote: [" << center[0] << ", " << center[1] << ", " << center[2] << "]\n";
        }
    }

    

    if (zero_boundary < 0) {
        center.array() += zero_boundary;
    }

    center[0] = (center[0] + x_mean_mm + 0.5) * acc_unit;
    center[1] = (center[1] + y_mean_mm + 0.5) * acc_unit;
    center[2] = (center[2] + z_mean_mm + 0.5) * acc_unit;

    return center;

}