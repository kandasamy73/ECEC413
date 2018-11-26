/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 Author: N. Kandasamy, 1/29/2011 

*/

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "vec_mat_mult.h"


__global__ void MatrixMulKernel_prefetch_v2(const float* A, const float* X, float* Y)
{
	//Multiply the two matrices
	// Declare shared memory
	__shared__ float aTile_1[TILE_SIZE][TILE_SIZE];
	__shared__ float xTile_1[TILE_SIZE];

	__shared__ float aTile_2[TILE_SIZE][TILE_SIZE];
	__shared__ float xTile_2[TILE_SIZE]; 

	// Calculate thread index, block index and position in matrix
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockY = blockIdx.y;
	int yInMatrix = TILE_SIZE * blockY + threadY;
	int i, k;
	
	// Clear partialSum for thread
	float partialSum = 0.0f;

	for (i = 0; i < MATRIX_SIZE; i += 2*TILE_SIZE) {
		// Populate shared memory for the two tiles
		aTile_1[threadY][threadX] = A[MATRIX_SIZE * yInMatrix + i + threadX]; // Bring TILE_SIZE elements per row of the A matrix into shared memory 
		if(threadY == 0) xTile_1[threadX] = X[threadX + i]; // Bring TILE_SIZE elements of the vector X into shared memory
					
		aTile_2[threadY][threadX] = A[MATRIX_SIZE * yInMatrix + i + TILE_SIZE + threadX]; // Bring TILE_SIZE elements per row of the A matrix into shared memory 
		if(threadY == 0) xTile_2[threadX] = X[i + TILE_SIZE + threadX]; // Bring TILE_SIZE elements of the vector X into shared memory

		__syncthreads();

		// Compute partialSum for the current Tile
		float aElement1, xElement1;
		float aElement2, xElement2;
		for (k = 0; k < TILE_SIZE; k += 1){	
			aElement1 = aTile_1[threadY][k]; xElement1 = xTile_1[k];			
			aElement2 = aTile_2[threadY][k]; xElement2 = xTile_2[k]; 

			partialSum += aElement1 * xElement1 + aElement2 * xElement2;
		}		
		__syncthreads();
	}

	// Store partialSum
	if (threadX == 0) Y[yInMatrix] = partialSum;
}


__global__ void MatrixMulKernel_sm_prefetch_v1(const float* A, const float* X, float* Y)
{
	//Multiply the two matrices
	// Declare shared memory
	__shared__ float aTile_1[TILE_SIZE][TILE_SIZE];
	__shared__ float xTile_1[TILE_SIZE];

	__shared__ float aTile_2[TILE_SIZE][TILE_SIZE];
	__shared__ float xTile_2[TILE_SIZE]; 

	// Calculate thread index, block index and position in matrix
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockY = blockIdx.y;
	int yInMatrix = TILE_SIZE * blockY + threadY;
	int i, k;
	
	// Clear partialSum for thread
	float partialSum = 0.0f;

	for (i = 0; i < MATRIX_SIZE; i += 2*TILE_SIZE) {
		// Populate shared memory for the first tile
		aTile_1[threadY][threadX] = A[MATRIX_SIZE * yInMatrix + i + threadX]; // Bring TILE_SIZE elements per row of the A matrix into shared memory 
		if(threadY == 0) xTile_1[threadX] = X[threadX + i]; // Bring TILE_SIZE elements of the vector X into shared memory		
		__syncthreads();

		// Compute partialSum for Tile 1
		float aElement1, xElement1;
		float aElement2, xElement2;
		for (k = 0; k < TILE_SIZE; k += 1){	
			aElement1 = aTile_1[threadY][k]; xElement1 = xTile_1[k];			
			partialSum += aElement1 * xElement1;
		}	
		
		// Prefetch the next tile in parallel with the execution of the previous for loop
		aTile_2[threadY][threadX] = A[MATRIX_SIZE * yInMatrix + i + TILE_SIZE + threadX]; // Bring TILE_SIZE elements per row of the A matrix into shared memory 
		if(threadY == 0) xTile_2[threadX] = X[i + TILE_SIZE + threadX]; // Bring TILE_SIZE elements of the vector X into shared memory
		__syncthreads();

		// Compute partial sum for Tile 2	
		for (k = 0; k < TILE_SIZE; k += 1){	
			aElement2 = aTile_2[threadY][k]; xElement2 = xTile_2[k]; 
			partialSum += aElement2 * xElement2;
		}		
	}

	// Store partialSum
	if (threadX == 0) Y[yInMatrix] = partialSum;
}



__global__ void MatrixMulKernel_sm_vanilla(const float* A, const float* X, float* Y)
{
		  // Declare shared memory
	__shared__ float aTile[TILE_SIZE][TILE_SIZE];
	__shared__ float xTile[TILE_SIZE];

	// Calculate thread index, block index and position in matrix
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockY = blockIdx.y;
	int yInMatrix = TILE_SIZE * blockY + threadY;

	// Clear partialSum for thread
	float partialSum = 0.0f;

	for (int i = 0; i < MATRIX_SIZE; i += TILE_SIZE) {
		// Populate shared memory for the current tile if within range
		aTile[threadY][threadX] = A[MATRIX_SIZE * yInMatrix + i + threadX]; // Bring TILE_SIZE elements per row of the A matrix into shared memory 
		if(threadY == 0) xTile[threadX] = X[i + threadX]; // Bring TILE_SIZE elements of the vector X into shared memory

		__syncthreads();

		// Compute partialSum for the current Tile
		/*
		for (int k = 0; k < TILE_SIZE; ++k)
			partialSum += aTile[threadY][k] * xTile[k];
		*/
		float aElement1;
		float xElement1;
		for (int k = 0; k < TILE_SIZE; k += 1){
			aElement1 = aTile[threadY][k]; xElement1 = xTile[k]; 		
			partialSum += aElement1 * xElement1; 
		}

		__syncthreads();
	}

	// Store partialSum
	if (threadX == 0) {
		Y[yInMatrix] = partialSum;
	}
}

/* This kernel uses global memory to compute the result. It uses the concept of prefetching to speed up the process. */
__global__ void MatrixMulKernel_gm_vanilla_prefetch(const float *A, const float *X, float *Y)
{
		  int threadX = threadIdx.x; 
		  int blockY = blockIdx.y;
		  int yInMatrix = THREAD_BLOCK_SIZE * blockY + threadX; // Obtain the corresponding row number

		  float partialSum = 0.0f;

		  float aElement0 = A[MATRIX_SIZE * yInMatrix];
		  float xElement0 = X[0];
		  float aElement1 = A[MATRIX_SIZE * yInMatrix + 1];
		  float xElement1 = X[1];

		  /* Loop is unrolled twice. */
		  for(int i = 0; i < (MATRIX_SIZE - 2); i += 2){

					 /* Compute for the current iteration. */
					 partialSum += aElement0 * xElement0 + aElement1 * xElement1;
					 
					 /* Prefetch values for the next iteration. */
					 aElement0 = A[MATRIX_SIZE * yInMatrix + i + 2];
					 xElement0 = X[i + 2];
					 aElement1 = A[MATRIX_SIZE * yInMatrix + i + 3];
					 xElement1 = X[i + 3];
		  }

		  partialSum += aElement0 * xElement0 + aElement1 * xElement1;
		  Y[yInMatrix] = partialSum;
}

/* This kernel uses global memory to compute the result. */
__global__ void MatrixMulKernel_gm_vanilla(const float *A, const float *X, float *Y)
{
		  int threadX = threadIdx.x; 
		  int blockY = blockIdx.y;
		  int yInMatrix = THREAD_BLOCK_SIZE * blockY + threadX; // Obtain the corresponding row number

		  float partialSum = 0.0f;
		  for(int i = 0; i < MATRIX_SIZE; i++){
					 partialSum += A[MATRIX_SIZE * yInMatrix + i] * X[i];
		  }
		  Y[yInMatrix] = partialSum;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
