#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void MatrixProduct(float* A, float* B, float* C, int rows, int cols_A, int cols_B)
{
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < rows && col_idx < cols_B) {
        float c_value = 0;
        for (int k = 0; k < cols_A; k++) {
            c_value += A[row_idx * cols_A + k] * B[k * cols_B + col_idx];
        }
        C[row_idx * cols_B + col_idx] = c_value;
    }
}

// Function to display a matrix
void display_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int rows_A = 1024; // number of rows in A and C
    int cols_A = 512; // number of columns in A and number of rows in B
    int cols_B = 2048; // number of columns in B and C

    // Allocate memory for input and output matrices on host
    float *h_A = (float*)malloc(rows_A * cols_A * sizeof(float));
    float *h_B = (float*)malloc(cols_A * cols_B * sizeof(float));
    float *h_C = (float*)malloc(rows_A * cols_B * sizeof(float));

    // Initialize input matrices with random values
    srand(time(NULL));
    for (int i = 0; i < rows_A * cols_A; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < cols_A * cols_B; i++) {
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate memory for input and output matrices on device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, rows_A * cols_A * sizeof(float));
    cudaMalloc((void**)&d_B, cols_A * cols_B * sizeof(float));
    cudaMalloc((void**)&d_C, rows_A * cols_B * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, rows_A * cols_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, cols_A * cols_B * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel function with appropriate thread layout and block size
    int block_size = 16; 
    dim3 threadsPerBlock(block_size, block_size);
    dim3 numBlocks(ceil(cols_B / (float)block_size), ceil(rows_A / (float)block_size));
    
    cudaEvent_t start, stop;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    MatrixProduct<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rows_A, cols_A, cols_B);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_C, d_C, rows_A * cols_B * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Elapsed time: %f ms\n", elapsed_time);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}