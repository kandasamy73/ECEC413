// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <vector_dot_product_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void run_test(unsigned int);
void run_test_with_pinned_memory(unsigned int);
float compute_on_device_sp(float *, float *,int);
float compute_on_device_kahan(float *, float *,int);
float compute_on_device_dp(float *, float *,int);
float compute_on_device_with_pinned_memory(float *, float *,int);
void check_for_error(char *);

extern "C" float compute_gold_sp( float *, float *, unsigned int);
extern "C" double compute_gold_dp( float *, float *, unsigned int);
extern "C" float compute_gold_kahan( float *, float *, unsigned int);

int main( int argc, char** argv) 
{
	if(argc != 2){
		printf("Usage: vector_dot_product <num elements> \n");
		exit(0);	
	}

	// Check the device properties if our device supports mapping host memory 
	cudaDeviceProp properties;
	int my_device;
	cudaGetDevice(&my_device);
	cudaGetDeviceProperties(&properties, my_device);
	if(properties.canMapHostMemory != 1){
			  printf("The device cannot map host memory. \n");
			  exit(0);
	}

	// Place the CUDA runtime in a state which supports mapping memory on the host 
	cudaSetDeviceFlags(cudaDeviceMapHost);

	unsigned int num_elements = atoi(argv[1]);
	// run_test_with_pinned_memory(num_elements);
	run_test(num_elements);
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Perform vector dot product on the CPU and the GPU and compare results for correctness
////////////////////////////////////////////////////////////////////////////////
void run_test(unsigned int num_elements) 
{
	// Obtain the vector length
	unsigned int vector_size = sizeof(float) * num_elements;

    // Allocate memory on the CPU for the input vectors A and B, and the output vector C
	float *A = (float *)malloc(vector_size);
	float *B = (float *)malloc(vector_size);
	
	// Randomly generate input data. Initialize the input data to be values between 0 and .5 
	srand(time(NULL));
	for(unsigned int i = 0; i < num_elements; i++){
		A[i] = (float)rand()/(float)RAND_MAX - .5;
		B[i] = (float)rand()/(float)RAND_MAX - .5;
	}

	// Generate dot product on the CPU 
	float time;
	unsigned int cpu_timer;
	cutCreateTimer(&cpu_timer);

	cutStartTimer(cpu_timer);

	float reference1 = compute_gold_sp(A, B, num_elements);

	cutStopTimer(cpu_timer);
	time = 1e-3 * cutGetTimerValue(cpu_timer);
	printf("CPU run time using single precision arithmetic: %0.10f s\n", time);

	cutStartTimer(cpu_timer);

	double reference2 = compute_gold_dp(A, B, num_elements);

	cutStopTimer(cpu_timer);
	time = 1e-3 * cutGetTimerValue(cpu_timer);
	printf("CPU run time using double precision arithmetic: %0.10f s\n", time);

	cutStartTimer(cpu_timer);
	
	double reference3 = compute_gold_kahan(A, B, num_elements);

	cutStopTimer(cpu_timer);
	time = 1e-3 * cutGetTimerValue(cpu_timer);
	printf("CPU run time using the Kahan summation technique: %0.10f s\n", time);

	// Compute the result vector on the GPU 
	unsigned int gpu_timer;
	cutCreateTimer(&gpu_timer);	
	
	cutStartTimer(gpu_timer);
	float gpu_result1 = compute_on_device_sp(A, B, num_elements);
	cutStopTimer(gpu_timer);
	time = 1e-3 * cutGetTimerValue(gpu_timer);
	printf("GPU run time using single precision arithmetic: %0.10f s\n", time);


	cutStartTimer(gpu_timer);
	float gpu_result2 = compute_on_device_kahan(A, B, num_elements);
	cutStopTimer(gpu_timer);
	time = 1e-3 * cutGetTimerValue(gpu_timer);
	printf("GPU run time using Kahan summation: %0.10f s\n", time);


	cutStartTimer(gpu_timer);
	float gpu_result3 = compute_on_device_dp(A, B, num_elements);
	cutStopTimer(gpu_timer);
	time = 1e-3 * cutGetTimerValue(gpu_timer);
	printf("GPU run time using double precision arithmetic: %0.10f s\n \n", time);


	printf("CPU result using single precision arithmetic: %f \n", reference1);
	printf("CPU result using double precision arithmetic: %f \n", reference2);
	printf("CPU result using Kahan method for single precision arithmetic: %f \n", reference3);
	printf("GPU result using single precision arithmetic: %f \n", gpu_result1);
	printf("GPU result using kahan summation: %f \n", gpu_result2);
	printf("GPU result using double precision arithmetic: %f \n", gpu_result3);

	// cleanup memory
	free(A);
	free(B);
	
	return;
}

////////////////////////////////////////////////////////////////////////////////
//! Perform vector dot product on the CPU and the GPU and compare results for correctness
// Use pinned memory
////////////////////////////////////////////////////////////////////////////////
void run_test_with_pinned_memory(unsigned int num_elements) 
{
	// Obtain the vector length
	unsigned int vector_size = sizeof(float) * num_elements;

	// Allocate pinned memory on the CPU for the input vectors A and B
	float *A = NULL;
	float *B = NULL;
	cudaHostAlloc((void **)&A, vector_size, cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void **)&B, vector_size, cudaHostAllocWriteCombined | cudaHostAllocMapped);
	check_for_error("ERROR ALLOCATING PINNED MEMORY");

	// Randomly generate input data. Initialize the input data to be integer values between -.5 and +.5 
	// Since A and B are write combined, read/write operations on them are not guranteed to be coherent
	// That is, the writes to A and B are not guaranteed to be visible until a fence operation is called. 
	// In our case, that will happen when the GPU kernel is called.
	srand(time(NULL));
	for(unsigned int i = 0; i < num_elements; i++){
		A[i] = (float)rand()/(float)RAND_MAX - 0.5;
     	B[i] = (float)rand()/(float)RAND_MAX - 0.5;
	}

	// Compute the result vector on the GPU 
	printf("Computing dot product on the GPU...");
	unsigned int gpu_timer;
	float time;
	cutCreateTimer(&gpu_timer);
	cutStartTimer(gpu_timer);

	float gpu_result = compute_on_device_with_pinned_memory(A, B, num_elements);

	cutStopTimer(gpu_timer);
	printf("done. \n");
	time = 1e-3 * cutGetTimerValue(gpu_timer);
	printf("GPU run time: %0.10f s\n", time);

	printf("Generating dot product on the CPU...");
	unsigned int cpu_timer;
	cutCreateTimer(&cpu_timer);
	cutStartTimer(cpu_timer);
	
	// Compute the reference solution on the CPU
	float reference = compute_gold_dp(A, B, num_elements);
    
	cutStopTimer(cpu_timer);
	printf("done. \n");
	time = 1e-3 * cutGetTimerValue(cpu_timer);
	printf("CPU run time: %0.10f s\n", time);
	printf("\n");

	
	printf("Result on CPU: %f, result on GPU: %f. \n", reference, gpu_result);

	// cleanup memory
	cudaFreeHost(A);
	cudaFreeHost(B);
	check_for_error("ERROR FREEING PINNED MEMORY");

	return;
}


float compute_on_device_sp(float *A_on_host, float *B_on_host, int num_elements)
{
	float *A_on_device = NULL;
	float *B_on_device = NULL;
	float *C_on_device = NULL; 

	// Allocate space on the GPU for vectors A and B, and copy the contents of the vectors to the GPU
	cudaMalloc((void**)&A_on_device, num_elements * sizeof(float));
	cudaMemcpy(A_on_device, A_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&B_on_device, num_elements * sizeof(float));
	cudaMemcpy(B_on_device, B_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate space for the result vector on the GPU
	cudaMalloc((void**)&C_on_device, NUM_BLOCKS * sizeof(float));
	cudaMemset(C_on_device, 0.0f, NUM_BLOCKS * sizeof(float));

	int *mutex = NULL;
	cudaMalloc((void **)&mutex, sizeof(int));
	cudaMemset(mutex, 0, sizeof(int));


 	// Set up the execution grid on the GPU
	dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1); // Set the number of threads in the thread block
	dim3 grid(NUM_BLOCKS,1);
	
	// Launch the kernel
	vector_dot_product_kernel_v2_sp<<<grid, thread_block>>>(A_on_device, B_on_device, C_on_device, num_elements, mutex);
	cudaThreadSynchronize();
	check_for_error("KERNEL FAILURE");

	float sum;
	cudaMemcpy(&sum, &C_on_device[0], sizeof(float), cudaMemcpyDeviceToHost);

	// Free memory
	cudaFree(A_on_device);
	cudaFree(B_on_device);
	cudaFree(C_on_device);

	return (float)sum;
}

float compute_on_device_kahan(float *A_on_host, float *B_on_host, int num_elements)
{
	float *A_on_device = NULL;
	float *B_on_device = NULL;
	float *C_on_device = NULL; 

	// Allocate space on the GPU for vectors A and B, and copy the contents of the vectors to the GPU
	cudaMalloc((void**)&A_on_device, num_elements * sizeof(float));
	cudaMemcpy(A_on_device, A_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&B_on_device, num_elements * sizeof(float));
	cudaMemcpy(B_on_device, B_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate space for the result vector on the GPU
	cudaMalloc((void**)&C_on_device, NUM_BLOCKS * sizeof(float));
	cudaMemset(C_on_device, 0.0f, NUM_BLOCKS * sizeof(float));

	int *mutex = NULL;
	cudaMalloc((void **)&mutex, sizeof(int));
	cudaMemset(mutex, 0, sizeof(int));


 	// Set up the execution grid on the GPU
	dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1); // Set the number of threads in the thread block
	dim3 grid(NUM_BLOCKS,1);
	
	// Launch the kernel
	vector_dot_product_kernel_v2_kahan<<<grid, thread_block>>>(A_on_device, B_on_device, C_on_device, num_elements, mutex);
	cudaThreadSynchronize();
	check_for_error("KERNEL FAILURE");

	float sum;
	cudaMemcpy(&sum, &C_on_device[0], sizeof(float), cudaMemcpyDeviceToHost);

	// Free memory
	cudaFree(A_on_device);
	cudaFree(B_on_device);
	cudaFree(C_on_device);

	return (float)sum;
}


float compute_on_device_dp(float *A_on_host, float *B_on_host, int num_elements)
{
	float *A_on_device = NULL;
	float *B_on_device = NULL;
	float *C_on_device = NULL; 

	// Allocate space on the GPU for vectors A and B, and copy the contents of the vectors to the GPU
	cudaMalloc((void**)&A_on_device, num_elements * sizeof(float));
	cudaMemcpy(A_on_device, A_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&B_on_device, num_elements * sizeof(float));
	cudaMemcpy(B_on_device, B_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate space for the result vector on the GPU
	cudaMalloc((void**)&C_on_device, NUM_BLOCKS * sizeof(float));
	cudaMemset(C_on_device, 0.0f, NUM_BLOCKS * sizeof(float));

	int *mutex = NULL;
	cudaMalloc((void **)&mutex, sizeof(int));
	cudaMemset(mutex, 0, sizeof(int));


 	// Set up the execution grid on the GPU
	dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1); // Set the number of threads in the thread block
	dim3 grid(NUM_BLOCKS,1);
	
	// Launch the kernel
	vector_dot_product_kernel_v2_dp<<<grid, thread_block>>>(A_on_device, B_on_device, C_on_device, num_elements, mutex);
	cudaThreadSynchronize();
	check_for_error("KERNEL FAILURE");

	float sum;
	cudaMemcpy(&sum, &C_on_device[0], sizeof(float), cudaMemcpyDeviceToHost);

	// Free memory
	cudaFree(A_on_device);
	cudaFree(B_on_device);
	cudaFree(C_on_device);

	return (float)sum;
}



float compute_on_device_with_pinned_memory(float *A_on_host, float *B_on_host, int num_elements)
{
	float *A_on_device = NULL;
	float *B_on_device = NULL;

	// Get valid GPU pointers using pointers on the pinned memory on the host 
	cudaHostGetDevicePointer(&A_on_device, A_on_host, 0);
	cudaHostGetDevicePointer(&B_on_device, B_on_host, 0);
	check_for_error("ERROR GETTING GPU POINTER");

	// Allocate pinned memory on the CPU for the result
	float *C_on_host = NULL;
	float *C_on_device = NULL;
	cudaHostAlloc((void **)&C_on_host, sizeof(float) * NUM_BLOCKS, cudaHostAllocMapped);
	cudaHostGetDevicePointer(&C_on_device, C_on_host, 0);
	check_for_error("ERROR ALLOCATING PINNED MEMORY");

 	// Set up the execution grid on the GPU
	dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1); // Set the number of threads in the thread block
	dim3 grid(NUM_BLOCKS,1);
	
	// Launch the kernel
	vector_dot_product_kernel<<<grid, thread_block>>>(A_on_device, B_on_device, C_on_device, num_elements);
	cudaThreadSynchronize();
	check_for_error("KERNEL FAILURE");

	// Perform final reduction on the CPU
	double sum = 0.0f;
	for(unsigned int i = 0; i < NUM_BLOCKS; i++)
			  sum += C_on_host[i];

	// Free memory
	cudaFreeHost(C_on_host);
	check_for_error("ERROR FREEING PINNED MEMORY");

	return (float)sum;
}


void check_for_error(char *msg){
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} 
