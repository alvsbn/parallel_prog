#include "matrix.h"
#include "iostream"
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <random>
#include <mpi.h>

using namespace std;


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


Matrix Matrix::multiply_mpi(const Matrix& other) const {

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (_cols != other._rows) {
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	int rows_a, cols_a, rows_b, cols_b;

	if (rank == 0) {
		rows_a = _rows;
		cols_a = _cols;
		rows_b = other._rows;
		cols_b = other._cols;
	}

	MPI_Bcast(&rows_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&cols_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&rows_b, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&cols_b, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int rows_per_proc_base = rows_a / size;
	int remainder_rows = rows_a % size;

	vector<int> send_counts(size), send_displs(size);
	for (int i = 0; i < size; ++i) {
		int rows_for_proc = rows_per_proc_base + (i < remainder_rows ? 1 : 0);
		send_counts[i] = rows_for_proc * cols_a;
		send_displs[i] = (i == 0 ? 0 : send_displs[i - 1] + send_counts[i - 1]);
	}

	int local_rows = send_counts[rank] / cols_a;

	vector<int> flat_matrix_a;
	if (rank == 0) {
		flat_matrix_a.resize(rows_a * cols_a);
		copy(_data.begin(), _data.end(), flat_matrix_a.begin());
	}

	vector<int> local_a(local_rows * cols_a);
	MPI_Scatterv(flat_matrix_a.data(), send_counts.data(), send_displs.data(), MPI_INT,
		local_a.data(), send_counts[rank], MPI_INT, 0, MPI_COMM_WORLD);

	vector<int> flat_matrix_b(rows_b * cols_b);
	if (rank == 0) {
		copy(other._data.begin(), other._data.end(), flat_matrix_b.begin());
	}

	MPI_Bcast(flat_matrix_b.data(), rows_b * cols_b, MPI_INT, 0, MPI_COMM_WORLD);

	vector<int> local_c(local_rows * cols_b, 0);
	for (int i = 0; i < local_rows; ++i) {
		for (int j = 0; j < cols_b; ++j) {
			int sum = 0;
			for (int k = 0; k < cols_a; ++k) {
				sum += local_a[i * cols_a + k] * flat_matrix_b[k * cols_b + j];
			}
			local_c[i * cols_b + j] = sum;
		}
	}

	vector<int> recv_counts(size), recv_displs(size);

	for (int i = 0; i < size; ++i) {
		int rows_for_proc = send_counts[i] / cols_a;
		recv_counts[i] = rows_for_proc * cols_b;
		recv_displs[i] = (i == 0 ? 0 : recv_displs[i - 1] + recv_counts[i - 1]);
	}

	vector<int> flat_result;
	if (rank == 0) flat_result.resize(rows_a * cols_b);
	MPI_Gatherv(local_c.data(), recv_counts[rank], MPI_INT,
		flat_result.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
		0, MPI_COMM_WORLD);

	if (rank == 0) {
		Matrix result(rows_a, cols_b);
		copy(flat_result.begin(), flat_result.end(), result._data.begin());
		return result;
	}

	return Matrix();
}

