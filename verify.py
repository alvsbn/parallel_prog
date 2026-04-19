import numpy as np


def read_matrix_from_file(filename):
    with open(filename, 'r') as file:
        rows, cols = map(int, file.readline().split())
        matrix = np.loadtxt(file, dtype=int)
        return matrix.reshape(rows, cols)


def verify_matrix_multiplication(size, result_file):
    matrix_1 = read_matrix_from_file(f"matrix_1_{size}.txt")
    matrix_2 = read_matrix_from_file(f"matrix_2_{size}.txt")
    result_cpp = read_matrix_from_file(f"result_{size}.txt")

    result_py = np.dot(matrix_1, matrix_2)


    result_file.write(f"Размер матрицы {size}x{size}:")
    if np.array_equal(result_py, result_cpp):
        result_file.write("Результаты совпадают\n")
    else:
        result_file.write("Результаты не совпадают\n")


if __name__ == "__main__":
    sizes = [200, 400, 800, 1200, 1600, 2000]

    with open("verify_results.txt", "w", encoding="utf-8") as result_file:
        result_file.write(f"Проверка умножения матриц:\n\n")
        for size in sizes:
            verify_matrix_multiplication(size, result_file)
