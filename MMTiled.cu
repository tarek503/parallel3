#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
__global__ void MatrixMulKernel(float* M, float* N, float* P,int rowm , int colm , int coln )
{
    __shared__ float ds_M[16][16];
    __shared__ float ds_N[16][16];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    float Pvalue = 0;
    for( int p = 0; p < (coln -1 ) / 16 +1 ; p++ ){ //
        if(Row<rowm && tx* 16+tx<colm)
        {
            ds_M[ty][tx] = M[Row * colm + p * 16 + tx];
        }else{
            ds_M[ty][tx] = 0.0;
        }
        if (p*16+ty < colm && Col < coln ) {
            ds_N[ty][tx] = N[(p*16 + ty) * coln+ Col];
        }else{
            ds_N[ty][tx] = 0.0;
        }
        __syncthreads();

        if(Row<rowm && Col < coln){
            for(int i = 0; i < 16; i++){
                Pvalue+= ds_M[ty][i] * ds_N[i][tx];
                __syncthreads();
            }
            
        }
    }
    if(Row<coln&& Col < coln)
        P[Row*coln + Col] = Pvalue;
}

int main()
{
    int rowm = 1024; 
    int colm =512; 
    int coln = 2048; 
    int sizem= rowm * colm; int sizen = colm * coln; int sizep= rowm * coln;
    float *h_M = (float*)malloc(sizem* sizeof(float));
    float *h_N = (float*)malloc(sizen * sizeof(float));
    float *h_P = (float*)malloc(sizep * sizeof(float));

    for (int i = 0; i < rowm * colm; i++) {
        h_M[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < colm * coln; i++) {
        h_N[i] = rand() / (float)RAND_MAX;
    }

    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, sizem*sizeof(float));
    cudaMalloc((void**)&d_N,sizen *sizeof(float));
    cudaMalloc((void**)&d_P, sizep *sizeof(float));
    cudaMemcpy(d_M, h_M, sizem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, sizen* sizeof(float), cudaMemcpyHostToDevice);
    dim3 blockDim(16, 16);
    dim3 blocks((colm + blockDim.x - 1)/ blockDim.x, (rowm + blockDim.y -1) / blockDim.y );
    cudaEvent_t start, stop;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    MatrixMulKernel<<<blocks, blockDim>>>(d_M, d_N, d_P,rowm,colm,coln);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(h_P, d_P, sizep * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Elapsed time: %f ms\n", elapsed_time);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}