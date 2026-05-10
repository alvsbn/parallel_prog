#include <cuda_runtime.h>
#include <vector>

__global__ void matrix_multiply_kernel(const int* A, const int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void cuda_multiply(const std::vector<int>& A,
    const std::vector<int>& B,
    std::vector<int>& C,
    int N, int blockSizeX, int blockSizeY)
{
    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C = nullptr;

    size_t size = N * N * sizeof(int);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(blockSizeX, blockSizeY);
    dim3 numBlocks((N + blockSizeX - 1) / blockSizeX, (N + blockSizeY - 1) / blockSizeY);

    matrix_multiply_kernel <<< numBlocks, threadsPerBlock >>> (d_A, d_B, d_C, N);

    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}