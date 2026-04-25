#include "matrix.h"
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <random>


Matrix::Matrix() : _rows(0), _cols(0) {}
Matrix::Matrix(size_t rows, size_t cols) : _data(rows * cols, 0), _rows(rows), _cols(cols) {}
Matrix::Matrix(const Matrix& other) : _data(other._data), _rows(other._rows), _cols(other._cols) {}

Matrix& Matrix::operator=(const Matrix& other) {
	if (this != &other) {
		_data = other._data;
		_rows = other._rows;
		_cols = other._cols;
	}
	return *this;
}

size_t Matrix::rows() const {
	return _rows;
}

size_t Matrix::cols() const {
	return _cols;
}

int Matrix::get(size_t i, size_t j) const {
	if (i >= _rows || j >= _cols) {
		throw out_of_range("Index out of range");
	}
	return _data[i * _cols + j];
}

void Matrix::set(size_t i, size_t j, int value) {
	if (i >= _rows || j >= _cols) {
		throw out_of_range("Index out of range");
	}
	_data[i * _cols + j] = value;
}

Matrix Matrix::operator*(const Matrix& other) const {
	if (_cols != other._rows) {
		throw invalid_argument("Invalid matrix dimensions");
	}

	Matrix result(_rows, other._cols);

	#pragma omp parallel for
	for (int i = 0; i < _rows; ++i) {
		for (int j = 0; j < other._cols; ++j) {
			int sum = 0;
			for (int k = 0; k < _cols; ++k) {
				sum += _data[i * _cols + k] * other._data[k * other._cols + j];
			}
			result._data[i * other._cols + j] = sum;
		}
	}

	return result;
}

Matrix generate_random_matrix(size_t size, int min_value, int max_value) {
	Matrix result(size, size);

	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> dist(min_value, max_value);

	for (size_t i = 0; i < size; ++i) {
		for (size_t j = 0; j < size; ++j) {
			result.set(i, j, dist(gen));
		}
	}

	return result;
}

Matrix read_from_file(const string& filename) {
	ifstream file(filename);

	if (!file.is_open()) {
		throw runtime_error("The file cannot be opened");
	}

	size_t rows, cols;
	file >> rows >> cols;

	Matrix matrix(rows, cols);

	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			int value;
			file >> value;
			matrix.set(i, j, value);
		}
	}

	file.close();
	return matrix;
}

void Matrix::write_to_file(const string& filename) const {
	ofstream file(filename);

	if (!file.is_open()) {
		throw runtime_error("The file cannot be opened");
	}

	file << _rows << " " << _cols << "\n";
	for (size_t i = 0; i < _rows; ++i) {
		for (size_t j = 0; j < _cols; ++j) {
			file << get(i, j) << " ";
		}
		file << "\n";
	}

	file.close();
}

ostream& operator<<(ostream& stream, const Matrix& matrix) {
	for (size_t i = 0; i < matrix.rows(); ++i) {
		for (size_t j = 0; j < matrix.cols(); ++j) {
			stream << matrix.get(i, j) << "\t";
		}
		stream << "\n";
	}
	return stream;
}

