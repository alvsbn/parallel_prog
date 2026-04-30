#pragma once

#include <vector>
#include <string>

using namespace std;

class Matrix {
private:
	vector<int> _data;
	size_t _rows;
	size_t _cols;

public:
	Matrix();
	Matrix(size_t rows, size_t cols);
	Matrix(const Matrix& other);
	Matrix& operator=(const Matrix& other);

	size_t rows() const;
	size_t cols() const;
	int get(size_t i, size_t j) const;
	void set(size_t i, size_t j, int value);

	void write_to_file(const string& filename) const;

	Matrix multiply_mpi(const Matrix& other) const;

};

Matrix read_from_file(const string& filename);
Matrix generate_random_matrix(size_t size, int min_val = 0, int max_val = 100);

std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
