#include "matrix.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream> 


int main() {
    try {
        vector<size_t> sizes = {200, 400, 800, 1200, 1600, 2000};

        ofstream results_file("results.txt");

        for (size_t size : sizes) {

            Matrix matrix_1 = generate_random_matrix(size);
            Matrix matrix_2 = generate_random_matrix(size);

            matrix_1.write_to_file("matrix_1_" + to_string(size) + ".txt");
            matrix_2.write_to_file("matrix_2_" + to_string(size) + ".txt");

            auto start = chrono::high_resolution_clock::now();
            Matrix result = matrix_1 * matrix_2;
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> dif = end - start;

            result.write_to_file("result_" + to_string(size) + ".txt");

            results_file << "Size: " << size << "x" << size << ", ";
            results_file << "Time: " << fixed << setprecision(5) << dif.count() << " seconds" << "\n";

        }

        results_file.close();

    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}