#include "AccumulatorSpace.h"


Matrix dot(Matrix A, Matrix B) {
    int n = A.size(), m = A[0].size(), p = B[0].size();
    Matrix C(n, std::vector<float>(p, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < m; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}
//TODO: Check implementation
std::vector<Matrix, Matrix> project(Matrix xyz, Matrix K, Matrix RT) {
    int N = xyz.size();

    Matrix xyz_T(3, std::vector<float>(N, 0));

    Matrix RT_T = dot(K, RT);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 3; j++) {
            xyz_T[j][i] = xyz[i][j];
        }
    }

    Matrix actual_xyz_T = dot(RT_T, xyz_T);

    Matrix actual_xyz(N, std::vector<float>(3, 0));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 3; j++) {
            actual_xyz[i][j] = actual_xyz_T[j][i];
        }
    }

    Matrix xyz_K_T = dot(K, actual_xyz_T);

    Matrix xy(N, std::vector<float>(2, 0));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 2; j++) {
            xy[i][j] = xyz_K_T[j][i] / xyz_K_T[2][i];
        }
    }

    return { xy, actual_xyz };
}