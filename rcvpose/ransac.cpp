#include "ransac.h"

using namespace std;
using namespace Eigen;

bool intersecting_spheres(const Sphere& a, const Sphere& b) {
    double dist = (a.center - b.center).norm();
    return dist <= (a.radius + b.radius);
}

vector<Vector3d> sphere_intersection_points(const Sphere& a, const Sphere& b) {
    Vector3d d = b.center - a.center;
    double d_norm = d.norm();

    if (d_norm > a.radius + b.radius) {
        return {};
    }

    double a1 = (a.radius * a.radius - b.radius * b.radius + d_norm * d_norm) / (2 * d_norm);
    double a2 = d_norm - a1;
    double h = sqrt(a.radius * a.radius - a1 * a1);

    Vector3d p2 = a.center + (a1 / d_norm) * d;
    Vector3d p3 = p2 + (h / d_norm) * Vector3d(d[1], -d[0], 0);
    Vector3d p4 = p2 + (h / d_norm) * Vector3d(-d[1], d[0], 0);

    return { p3, p4 };

}

Vector3d lines_intersection(const Vector3d& p1, const Vector3d& p2, const Vector3d& p3, const Vector3d& p4) {
    Vector3d p13 = p1 - p3;
    Vector3d p43 = p4 - p3;
    Vector3d p21 = p2 - p1;

    //cout << "\t\tP13: " << p13[0] << " " << p13[1] << " " << p13[2] << endl;
    //cout << "\t\tP43: " << p43[0] << " " << p43[1] << " " << p43[2] << endl;
    //cout << "\t\tP21: " << p21[0] << " " << p21[1] << " " << p21[2] << endl;

    double d1343 = p13[0] * p43[0] + p13[1] * p43[1] + p13[2] * p43[2];
    double d4321 = p43[0] * p21[0] + p43[1] * p21[1] + p43[2] * p21[2];
    double d1321 = p13[0] * p21[0] + p13[1] * p21[1] + p13[2] * p21[2];
    double d4343 = p43[0] * p43[0] + p43[1] * p43[1] + p43[2] * p43[2];
    double d2121 = p21[0] * p21[0] + p21[1] * p21[1] + p21[2] * p21[2];

    //cout << "\t\tD1343: " << d1343 << endl;
    //cout << "\t\tD4321: " << d4321 << endl;
    //cout << "\t\tD1321: " << d1321 << endl;
    //cout << "\t\tD4343: " << d4343 << endl;
    //cout << "\t\tD2121: " << d2121 << endl;

    double denom = d2121 * d4343 - d4321 * d4321;

    //cout << "\t\tDenominator: " << denom << endl;

    if (abs(denom) < 1e-6) {
        return Vector3d(0, 0, 0);
    }


    double numer = d1343 * d4321 - d1321 * d4343;

    //cout << "\t\tNumerator: " << numer << endl;

    double mua = numer / denom;
    double mub = (d1343 + d4321 * mua) / d4343;

    //cout << "\t\tMua: " << mua << endl;
    //cout << "\t\tMub: " << mub << endl;

    Vector3d pa = p1 + mua * p21;
    Vector3d pb = p3 + mub * p43;

    //cout << "\t\tPA: " << pa[0] << " " << pa[1] << " " << pa[2] << endl;
    //cout << "\t\tPB: " << pb[0] << " " << pb[1] << " " << pb[2] << endl;

    return (pa + pb) / 2.0;

}

Sphere find_circumcenter(const Sphere& A, const Sphere& B, const Sphere& C) {
    if (!intersecting_spheres(A, B) || !intersecting_spheres(B, C) || !intersecting_spheres(A, C)) {
        return Sphere{ Vector3d(0, 0, 0), 0 };
    }

    auto A_B_intersection = sphere_intersection_points(A, B);
    auto B_C_intersection = sphere_intersection_points(B, C);

    //cout << "\t\tAB: " << A_B_intersection[0][0] << " " << A_B_intersection[0][1] << " " << A_B_intersection[0][2] << endl;
    //cout << "\t\tBC: " << B_C_intersection[0][0] << " " << B_C_intersection[0][1] << " " << B_C_intersection[0][2] << endl;

    //auto A_C_intersection = sphere_intersection_points(A, C);

    Vector3d circumcenter = lines_intersection(A_B_intersection[0], A_B_intersection[1], B_C_intersection[0], B_C_intersection[1]);
    if (circumcenter[0] == 0 && circumcenter[1] == 0 && circumcenter[1] == 0) {
        return Sphere{ Vector3d(0, 0, 0), 0 };
    }

    //cout << "\t\tCircumcenter: " << circumcenter[0] << " " << circumcenter[1] << " " << circumcenter[2] << endl;

    double radius = (circumcenter - A.center).norm();

    //cout << "\t\tCalculated radius: " << radius << "\tActual radius: " << A.radius << endl;

    return Sphere{ circumcenter, radius };
}



Vector3d Ransac_3D(const vector<Vertex>& xyz, const vector<double>& radial_list, const double& epsilon, const bool& debug, std::atomic<bool>& flag) {
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


    map<int, vector<Sphere>> sphere_map;

#pragma omp parallel for
    for (int i = 0; i < sphere_list.size(); i++) {
        int key = static_cast<int>(sphere_list[i].radius / epsilon);
#pragma omp critical
        {
            sphere_map[key].push_back(sphere_list[i]);
        }
    }

    int three_count = 0;


    for (auto [key, spheres] : sphere_map) {
        if (spheres.size() < 3) {
            sphere_map.erase(key);
        }
        else {
            three_count += floor(spheres.size() / 3);
        }
    }

    unordered_map<string, int> sphere_votes;

    int iterations = 10000;

    if (!debug) {
        cout << "\tIterations: " << iterations << endl;

        for (int i = 0; i < iterations; i++) {
            int key_index = rand() % sphere_map.size();

            auto it = sphere_map.begin();
            advance(it, key_index);
            vector<Sphere> spheres = it->second;

            if (spheres.size() < 3) {
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


            cout << "Input Spheres:\n";
            cout << "\t[" << p1.center[0] << ", " << p1.center[1] << ", " << p1.center[3] << "] r = " << p1.radius << "\n";
            cout << "\t[" << p2.center[0] << ", " << p2.center[1] << ", " << p2.center[3] << "] r = " << p2.radius << "\n";
            cout << "\t[" << p3.center[0] << ", " << p3.center[1] << ", " << p3.center[3] << "] r = " << p3.radius << endl << endl;

            Sphere circumcenter_sphere = find_circumcenter(p1, p2, p3);

            cout << "Circumcenter Sphere:\n";
            cout << "\t[" << circumcenter_sphere.center[0] << ", " << circumcenter_sphere.center[1] << ", " << circumcenter_sphere.center[2] << "] r = " << circumcenter_sphere.radius << "\n";
            cin.get();

            if (circumcenter_sphere.radius == 0) {
                continue;
            }

            circumcenter_sphere.center[0] = round(circumcenter_sphere.center[0]);
            circumcenter_sphere.center[1] = round(circumcenter_sphere.center[1]);
            circumcenter_sphere.center[2] = round(circumcenter_sphere.center[2]);


            stringstream ss;
            ss << fixed << setprecision(8) << circumcenter_sphere.center[0] << "_" << circumcenter_sphere.center[1] << "_" << circumcenter_sphere.center[2];
            string circumcenter_string = ss.str();

            if (sphere_votes.find(circumcenter_string) == sphere_votes.end()) {
                sphere_votes[circumcenter_string] = 1;
            }
            else {
                sphere_votes[circumcenter_string]++;
            }
        }
    }
    else {
        #pragma omp parallel for
        for (int i = 0; i < iterations; i++) {
            int key_index = rand() % sphere_map.size();

            if (flag.load()) {
                break;
            }

            auto it = sphere_map.begin();
            advance(it, key_index);
            vector<Sphere> spheres = it->second;

            if (spheres.size() < 3) {
                continue;
            }

            int p1_index, p2_index, p3_index;

            p1_index = rand() % spheres.size();

            do {
                p2_index = rand() % spheres.size();
                if (flag.load()) {
                    break;
                }
            } while (p2_index == p1_index);

            do {
                if (flag.load()) {
                    break;
                }
                p3_index = rand() % spheres.size();
            } while (p3_index == p1_index || p3_index == p2_index);

            Sphere p1 = spheres[p1_index];
            Sphere p2 = spheres[p2_index];
            Sphere p3 = spheres[p3_index];

            Sphere circumcenter_sphere = find_circumcenter(p1, p2, p3);

            circumcenter_sphere.center[0] = round(circumcenter_sphere.center[0]);
            circumcenter_sphere.center[1] = round(circumcenter_sphere.center[1]);
            circumcenter_sphere.center[2] = round(circumcenter_sphere.center[2]);


            stringstream ss;
            ss << fixed << setprecision(8) << circumcenter_sphere.center[0] << "_" << circumcenter_sphere.center[1] << "_" << circumcenter_sphere.center[2];
            string circumcenter_string = ss.str();

            if (sphere_votes.find(circumcenter_string) == sphere_votes.end()) {
                #pragma omp critical
                {
                    sphere_votes[circumcenter_string] = 1;
                }
            }
            else {
                #pragma omp atomic
                sphere_votes[circumcenter_string]++;
            }
        }

    }
    if (sphere_votes.size() == 0) {
        cerr << "RANSAC failed: no center found" << endl;
        return Vector3d(0, 0, 0);
    }

    int max_vote = 0;
    string max_vote_center;

    for (auto [center, vote] : sphere_votes) {

        if (vote > max_vote) {
            max_vote = vote;
            max_vote_center = center;
        }
    }

    if (debug) {
        cout << "\tRANSAC max vote: " << max_vote << endl;
    }


    stringstream ss(max_vote_center);
    string token;
    vector<double> center(3);


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
