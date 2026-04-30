#include "matrix.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <mpi.h>

using namespace std;
using namespace chrono;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<size_t> sizes = { 200, 400, 800, 1200, 1600, 2000 };

    ofstream results_file;
    if (rank == 0) {
        results_file.open("results_" + to_string(size) + "_processes.txt");
    }

    for (size_t n : sizes) {
        if (n % size != 0) {
            if (rank == 0) {
                results_file << "Size: " << n << "x" << n << ", Time: NO " << "\n";
            }
            continue;
        }

        Matrix A, B;

        if (rank == 0) {
            A = generate_random_matrix(n);
            B = generate_random_matrix(n);

            A.write_to_file("matrix_1_" + to_string(n) + ".txt");
            B.write_to_file("matrix_2_" + to_string(n) + ".txt");

        }

        auto start_time = high_resolution_clock::now();
        Matrix C = A.multiply_mpi(B);
        auto end_time = high_resolution_clock::now();
        auto elapsed = duration<double>(end_time - start_time).count();

        C.write_to_file("result_" + to_string(n) + ".txt");

        if (rank == 0) {
            results_file << "Size: " << n << "x" << n << ", Time: "
                << fixed << setprecision(5) << elapsed << " seconds\n";
        }
    }

    if (rank == 0) {
        results_file.close();
    }

    MPI_Finalize();
    return 0;
}

