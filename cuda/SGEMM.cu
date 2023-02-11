#include "header.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixMul(const float *A, const float *B, float *C, int M, int N, int K) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    if (tx < M && ty < N) {
        float c = 0;
        for (int i = 0; i < K; i++) {
            c += A[tx * K + i] * B[i * N + ty];
        }
        C[tx * N + ty] = c;
    }
}

void matrixMulCpu(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float c = 0;
            for (int k = 0; k < K; k++) {
                c += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = c;
        }
    }
}

int main(int argc, char **argv) {
    initDevice(0);

    int nx = 1 << 8;
    int nk = 1 << 8;
    int ny = 1 << 8;
    int num = nx * ny;

    // malloc
    float *A_host = (float *)malloc(num * sizeof(float));
    float *B_host = (float *)malloc(num * sizeof(float));
    float *C_host = (float *)malloc(num * sizeof(float));
    float *C_from_gpu = (float *)malloc(num * sizeof(float));
    initialData(A_host, num);
    initialData(B_host, num);

    // cuMalloc
    float *A_device = NULL;
    float *B_device = NULL;
    float *C_device = NULL;
    CHECK(cudaMalloc((void **)&A_device, num * sizeof(float)));
    CHECK(cudaMalloc((void **)&B_device, num * sizeof(float)));
    CHECK(cudaMalloc((void **)&C_device, num * sizeof(float)));

    // move input data from host to device(GPU)
    CHECK(cudaMemcpy(A_device, A_host, num * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_device, B_host, num * sizeof(float), cudaMemcpyHostToDevice));

    // cpu compute (get the correct answer)
    printf("CPU Execution starting \n");
    double istart = cpuSecond();
    matrixMulCpu(A_host, B_host, C_host, nx, ny, nk);
    double iElaps = cpuSecond() - istart;
    printf("CPU Execution no configuration<<<>>> Time elapsed %f sec\n", iElaps);

    // gpu compute (to check the kernel)
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);

    istart = cpuSecond();
    matrixMul<<<grid, block>>>(A_device, B_device, C_device, nx, ny, nk);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - istart;
    printf("GPU Execution configuration<<<(%d,%d),(%d,%d)>>> Time elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

    CHECK(cudaMemcpy(C_from_gpu, C_device, num * sizeof(float), cudaMemcpyDeviceToHost));
    checkResult(C_host, C_from_gpu, num);

    // free pointer
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
    free(A_host);
    free(B_host);
    free(C_host);
    cudaDeviceReset();

    return EXIT_SUCCESS;
}