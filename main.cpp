#include "matrix.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

using namespace std;

void cuda_multiply(const std::vector<int>& A,
    const std::vector<int>& B,
    std::vector<int>& C,
    int N, int blockSizeX, int blockSizeY);

int main() {
    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };

    vector<pair<int, int>> blocks = {
        {8, 8},
        {16, 8},
        {16, 16},
        {32, 8},
        {32, 16},
        {32, 32}
    };

    ofstream results_file("results_cuda.txt");

    for (int n : sizes) {
        Matrix A = generate_random_matrix(n);
        Matrix B = generate_random_matrix(n);

        A.write_to_file("matrix_1_" + to_string(n) + ".txt");
        B.write_to_file("matrix_2_" + to_string(n) + ".txt");

        vector<int> A_flat(n * n);
        vector<int> B_flat(n * n);
        vector<int> C_flat(n * n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A_flat[i * n + j] = A.get(i, j);
                B_flat[i * n + j] = B.get(i, j);
            }
        }

        for (auto& block : blocks) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            cuda_multiply(A_flat, B_flat, C_flat, n, block.first, block.second);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);

            results_file << "Size: " << n << "x" << n
                << ", Block: " << block.first << "x" << block.second
                << ", Time: " << fixed << setprecision(3) << ms
                << " ms\n";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        Matrix result(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result.set(i, j, C_flat[i * n + j]);
            }
        }
        result.write_to_file("result_" + to_string(n) + ".txt");
        results_file << "\n";
    }

    results_file.close();
    return 0;
}