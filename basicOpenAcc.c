#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

int main() {
    int numRows = 1024; // number of rows in matrixA and matrixResult
    int numColsA = 512; // number of columns in matrixA and number of rows in matrixB
    int numColsB = 2048; // number of columns in matrixB and matrixResult

    float** matrixA = (float**)malloc(numRows * sizeof(float*));
    for (int i = 0; i < numRows; i++) {
        matrixA[i] = (float*)malloc(numColsA * sizeof(float));
        for (int j = 0; j < numColsA; j++) {
            matrixA[i][j] = rand() / (float)RAND_MAX;
        }
    }

    float** matrixB = (float**)malloc(numColsA * sizeof(float*));
    for (int i = 0; i < numColsA; i++) {
        matrixB[i] = (float*)malloc(numColsB * sizeof(float));
        for (int j = 0; j < numColsB; j++) {
            matrixB[i][j] = rand() / (float)RAND_MAX;
        }
    }

    float** matrixResult = (float**)malloc(numRows * sizeof(float*));
    for (int i = 0; i < numRows; i++) {
        matrixResult[i] = (float*)malloc(numColsB * sizeof(float));
    }

    clock_t startTime = clock();
    // matrixA
    #pragma acc data copyin(matrixA[0:numRows][0:numColsA], matrixB[0:numColsA][0:numColsB]), copyout(matrixResult[0:numRows][0:numColsB])
    {
        #pragma acc parallel loop collapse(2)
        for(int i = 0; i < numRows; i++) {
            for (int j = 0; j < numColsB; j++) {
                matrixResult[i][j] = 0;
                for(int k = 0; k < numColsA; k++) {
                    matrixResult[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }
    }
    clock_t endTime = clock();
    double duration = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
    printf("Execution time: %lf seconds.\n", duration);

    return 0;
}