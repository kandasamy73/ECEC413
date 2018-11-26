/* Vector-matrix multiplication: Y = A * X.
 * Host code
Author: Naga Kandasamy
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <vec_mat_mult_kernel.cu>

#define MIN_NUMBER 1
#define MAX_NUMBER 4

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void compute_gold(float*, const float*, const float*, unsigned int, unsigned int);

Matrix allocate_matrix_on_gpu(const Matrix M);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost);
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice);
void vec_mat_mult_on_device(const Matrix M, const Matrix N, Matrix P);
void print_matrix(const Matrix M);
float get_random_number(int, int);

void checkCUDAError(const char *msg);
void free_device_matrix(Matrix* M);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	// Matrices for the program
	Matrix  A; // N x N matrix
	Matrix  X; // N x 1 vector
	Matrix  Y; // N x 1 vector
	
	// Initialize the random number generator with a seed value 
	srand(time(NULL));
	
	// Check command line arguments
	if(argc > 1) {
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}		
	 
	// Allocate and initialize the matrices
	printf("Creating the matrices. \n");
	A  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1); // Create a random matrix
	X  = allocate_matrix(MATRIX_SIZE, 1, 1); // Create a random vector 
	Y  = allocate_matrix(MATRIX_SIZE, 1, 0); // Allocate memory for the output vector
	printf("\n");	
	
	// Perform the vector-matrix multiplication on the GPU
	printf("Computing the vector-matrix multiplication on the GPU. \n");
	vec_mat_mult_on_device(A, X, Y);
	printf("\n");

	// compute the vector-matrix multiplication on the CPU for comparison
	printf("Computing the vector-matrix multiplication on the CPU. \n");
	unsigned int timer;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
	Matrix reference = allocate_matrix(MATRIX_SIZE, 1, 0);
	compute_gold(reference.elements, A.elements, X.elements, A.num_rows, A.num_columns);
	cutStopTimer(timer);
	float time = 1e-3 * cutGetTimerValue(timer);
	printf("CPU run time:        %0.10f s\n", time);

	// check if the device result is equivalent to the expected solution
	int size_elements = NUM_ROWS;
	CUTBoolean res = cutComparefe(reference.elements, Y.elements, size_elements, 0.0001f);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	// Free host matrices
	free(A.elements); A.elements = NULL;
	free(X.elements); X.elements = NULL;
	free(Y.elements); Y.elements = NULL;

	return 0;
}

/* 
	Perform the multiplication on the device.
*/
void vec_mat_mult_on_device(const Matrix A, const Matrix X, Matrix Y) 
{	
	/* Initialize timer. */
	unsigned int gpu_timer, kernel_timer;
	cutCreateTimer(&gpu_timer);
	cutCreateTimer(&kernel_timer);

	cutStartTimer(gpu_timer);

	/* Load matrices A and x on to the device. */
	Matrix Ad = allocate_matrix_on_gpu(A);
	copy_matrix_to_device(Ad, A);
	Matrix Xd = allocate_matrix_on_gpu(X);
	copy_matrix_to_device(Xd, X);
	
	/* Allocate Y on the device */
	Matrix Yd = allocate_matrix_on_gpu(Y);
	
	/* Setup the execution configuration for the kernel that uses global memory. We assume 512 threads per thread block. */
	// dim3 threads(THREAD_BLOCK_SIZE, 1, 1);
	// dim3 grid(1, (Yd.num_rows + THREAD_BLOCK_SIZE - 1)/THREAD_BLOCK_SIZE);

	/* Setup the execution configuration for the kernel that uses shared memory. */
	dim3 threads(TILE_SIZE, TILE_SIZE);
	dim3 grid(1, (Yd.num_rows + TILE_SIZE - 1)/TILE_SIZE);
	
	// execute the kernel
	cutStartTimer(kernel_timer);

	// MatrixMulKernel_gm_vanilla<<<grid, threads>>>(Ad.elements, Xd.elements, Yd.elements); // Kernel uses only global memory to compute the result
	// MatrixMulKernel_gm_vanilla_prefetch<<<grid, threads>>>(Ad.elements, Xd.elements, Yd.elements); // Kernel uses only global memory to compute the result
	MatrixMulKernel_sm_vanilla<<< grid, threads >>>(Ad.elements, Xd.elements, Yd.elements);
	// MatrixMulKernel_sm_prefetch_v2<<< grid, threads >>>(Ad.elements, Xd.elements, Yd.elements);
	cudaThreadSynchronize();
	cutStopTimer(kernel_timer);
	
	// check if kernel execution generated an error
	checkCUDAError("Error in kernel");
	
	// Read Y from the device
	copy_matrix_from_device(Y, Yd); 
	
	// Free device matrices
	cudaFree(&Ad);
	cudaFree(&Xd);
	cudaFree(&Yd);
	
	cutStopTimer(gpu_timer);
	
	float GPU_time = 1e-3 * cutGetTimerValue(gpu_timer);
	float kernel_time = 1e-3 * cutGetTimerValue(kernel_timer);
	printf("Total GPU run time:  %0.10f s\n", GPU_time);
	printf("GPU Kernel run time: %0.10f s\n", kernel_time);
	printf("Overhead:            %0.10f s\n", GPU_time - kernel_time);
}

// Allocate a device matrix of same size as M.
Matrix allocate_matrix_on_gpu(const Matrix M) {
	Matrix Mdevice = M;
	int size = M.num_rows * M.num_columns * sizeof(float);
	cudaMalloc((void**)&Mdevice.elements, size);
	return Mdevice;
}

// Free a device matrix.
void free_device_matrix(Matrix* M)
{
	cudaFree(M->elements);
	M->elements = NULL;
}

// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix allocate_matrix(int num_rows, int num_columns, int init) {
		Matrix M;
		M.num_columns = M.pitch = num_columns;
		M.num_rows = num_rows;
		int size = M.num_rows * M.num_columns;
		
	M.elements = (float*) malloc(size*sizeof(float));
	for(unsigned int i = 0; i < size; i++){
		if(init == 0)
			M.elements[i] = 0; 
		else
			M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
	return M;
}	

// Copy a host matrix to a device matrix.
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
	int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
	Mdevice.num_rows = Mhost.num_rows;
	Mdevice.num_columns = Mhost.num_columns;
	Mdevice.pitch = Mhost.pitch;
	cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice) {
	int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
	cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Prints the matrix out to screen
void print_matrix(const Matrix M) {
	for(unsigned int i = 0; i < 5; i++){
	//for(unsigned int i = 0; i < M.num_rows; i++){
		for(unsigned int j = 0; j < M.num_columns; j++)
			printf("%f ", M.elements[i*M.num_columns + j]);
		printf("\n");
	} 
	printf("\n");
}

// Returns a random floating-point number between the specified min and max values 
float get_random_number(int min, int max) {
	return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) 
	{
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}						 
}
