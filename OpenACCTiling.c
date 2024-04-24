#include <sys/time.h>
#include <stdio.h>

#define SIZE 4000

double matrixA[SIZE][SIZE];
double matrixB[SIZE][SIZE];
double matrixC[SIZE][SIZE];
double matrixD[SIZE][SIZE];

int main()
{
  int i,j,k;
  struct timeval tim;
  double startTime, endTime;
  double temp;
  
  // Initialize matrices.
  for (i = 0; i < SIZE; ++i) {
    for (j = 0; j < SIZE; ++j) {
      matrixA[i][j] = (double)(i + j);
      matrixB[i][j] = (double)(i - j);
      matrixC[i][j] = 0.0f;
      matrixD[i][j] = 0.0f;
    }
  }

  // Time stamp startTime
  gettimeofday(&tim, NULL);
  startTime = tim.tv_sec+(tim.tv_usec/1000000.0);  

  // Compute matrix multiplication.
#pragma acc data copyin(matrixA,matrixB) copy(matrixC)
#pragma acc kernels
#pragma acc loop tile(32,32)
  for (i = 0; i < SIZE; ++i) {
    for (j = 0; j < SIZE; ++j) {
      temp=0.0f;
#pragma acc loop reduction(+:temp)
      for (k = 0; k < SIZE; ++k) {
        temp += matrixA[i][k] * matrixB[k][j];
      }
      matrixC[i][j] = temp;
    }
  }

  // Time stamp endTime, elapsed time OpenACC
  gettimeofday(&tim, NULL);
  endTime = tim.tv_sec+(tim.tv_usec/1000000.0);
  printf("%.6lf seconds with OpenACC \n", endTime-startTime);

#pragma omp parallel for default(none) shared(matrixA,matrixB,matrixD) private(i,j,k)
  for (i = 0; i < SIZE; ++i) 
    for (j = 0; j < SIZE; ++j) 
      for (k = 0; k < SIZE; ++k) 
	matrixD[i][j] += matrixA[i][k] * matrixB[k][j];
 
  // Check the OpenACC result matrix
  for (i = 0; i < SIZE; ++i)
    for (j = 0; j < SIZE; ++j)
      if(matrixC[i][j] != matrixD[i][j]) {
	printf("Error %d %d %f %f \n", i,j, matrixC[i][j], matrixD[i][j]);
	exit(1);
      }
  printf("OpenACC matrix multiplication test was successful!\n");
  
  return 0;
}