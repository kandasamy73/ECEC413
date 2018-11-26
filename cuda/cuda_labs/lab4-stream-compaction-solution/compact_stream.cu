// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, project
#include <cutil_inline.h>

#define NUM_ELEMENTS 262144	// MIN = 512, MAX = 33553920
#define MAX_THREADS 512
#define LAST_THREAD (MAX_THREADS - 1)
#define MAX_THREAD_DIVISOR ((float)(1.0 / MAX_THREADS))

// includes, kernels
#include <compact_stream_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void compact_stream(void);


extern "C" void compact_stream_gold(float *reference, float *idata, unsigned int *len);

int main( int argc, char** argv) 
{
    compact_stream();
    cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a scan test for CUDA
////////////////////////////////////////////////////////////////////////////////
void compact_stream(void) 
{
    unsigned int num_elements = NUM_ELEMENTS;
    unsigned int cpuTimer, gpuTimer, kernelTimer;

	cutilCheckError(cutCreateTimer(&cpuTimer));
    cutilCheckError(cutCreateTimer(&gpuTimer));
	cutilCheckError(cutCreateTimer(&kernelTimer));
    
    const unsigned int num_blocks = (num_elements + MAX_THREADS - 1) / MAX_THREADS;
	const unsigned int num_blocks_blocks = (num_blocks + MAX_THREADS - 1) / MAX_THREADS;
    const unsigned int mem_size = sizeof(float) * num_elements;
	const unsigned int int_size = sizeof(int) * MAX_THREADS * num_blocks;
	const unsigned int sum_size = sizeof(int) * MAX_THREADS * num_blocks_blocks;

	// allocate host memory to store the input data
    float *h_data = (float*)malloc(mem_size);
      
    // initialize the input data on the host to be integer values
    // between 0 and 1000, both positive and negative
	float rand_number;
    for (unsigned int i = 0; i < num_elements; ++i) {
		rand_number = rand()/(float)RAND_MAX;
		if(rand_number > 0.5) 
			h_data[i] = floorf(1000 * (rand() / (float)RAND_MAX));
		else 
			h_data[i] = -floorf(1000 * (rand() / (float)RAND_MAX));
    }

    // compute reference solution
	cutStartTimer(cpuTimer);
    float *reference = (float*)malloc(mem_size);
	unsigned int reference_length = num_elements;
    compact_stream_gold(reference, h_data, &reference_length);
	cutStopTimer(cpuTimer);

	// compute gpu solution
	cutStartTimer(gpuTimer);
    // allocate device memory input/output arrays and processing arrays
    float *d_input, *d_output;
    cutilSafeCall(cudaMalloc((void**)&d_input, mem_size));
	cutilSafeCall(cudaMalloc((void**)&d_output, mem_size));
	int *d_flags, *d_scans, *d_sums, *d_increments;
	cutilSafeCall(cudaMalloc((void**)&d_flags, int_size));
	cutilSafeCall(cudaMalloc((void**)&d_scans, int_size));
	cutilSafeCall(cudaMalloc((void**)&d_sums, sum_size));
	cutilSafeCall(cudaMalloc((void**)&d_increments, sum_size));

    // copy host memory to device input array and initialize others
    cutilSafeCall(cudaMemcpy(d_input, h_data, mem_size, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemset(d_output, 0, mem_size));
	cutilSafeCall(cudaMemset(d_flags, 0, int_size));
	cutilSafeCall(cudaMemset(d_scans, 0, int_size));
	cutilSafeCall(cudaMemset(d_sums, 0, sum_size));
	cutilSafeCall(cudaMemset(d_increments, 0, sum_size));

    dim3 grid(num_blocks, 1, 1);
	dim3 grid_sums(num_blocks_blocks, 1, 1);
    dim3 threads(MAX_THREADS, 1, 1);

    // make sure there are no CUDA errors before we start
    cutilCheckMsg("Kernel execution failed");

	// execute the kernels
    cutStartTimer(kernelTimer);
    flag_and_scan<<< grid, threads >>>(d_input, d_scans, d_flags, d_sums);
	cudaThreadSynchronize();
	scan_sums<<< grid_sums, threads >>>(d_sums, d_increments, num_blocks);
	cudaThreadSynchronize();
	add_increments<<< grid, threads >>>(d_scans, d_increments);
	cudaThreadSynchronize();
	compact_stream<<< grid, threads >>>(d_output, d_input, d_flags, d_scans);
    cudaThreadSynchronize();
    cutStopTimer(kernelTimer);

    // check for any errors
    cutilCheckMsg("Kernel execution failed");

    // copy result from device to host
    cutilSafeCall(cudaMemcpy(h_data, d_output, mem_size, cudaMemcpyDeviceToHost));
	cutStopTimer(gpuTimer);

	// perform comparison and print statistics
	float epsilon = 0.0f;
    unsigned int result_regtest = cutComparefe(reference, h_data, reference_length, epsilon);
    printf("%s: Test %s\n\n", "compact_stream", (1 == result_regtest) ? "PASSED" : "FAILED");
	printf("CPU time: %f ms\n", cutGetTimerValue(cpuTimer));
	printf("GPU time: %f ms\n", cutGetTimerValue(gpuTimer));
	printf("Kernel time: %f ms\n", cutGetTimerValue(kernelTimer));
	printf("Overheard: %f ms\n", (cutGetTimerValue(gpuTimer) - cutGetTimerValue(kernelTimer)));

	cutResetTimer(cpuTimer);
	cutResetTimer(gpuTimer);
	cutResetTimer(kernelTimer);

    // cleanup memory
    free(h_data);
    free(reference);
    cutilSafeCall(cudaFree(d_input));
	cutilSafeCall(cudaFree(d_output));
	cutilSafeCall(cudaFree(d_flags));
	cutilSafeCall(cudaFree(d_scans));
	cutilSafeCall(cudaFree(d_sums));
	cutilSafeCall(cudaFree(d_increments));
    cutilCheckError(cutDeleteTimer(cpuTimer));
	cutilCheckError(cutDeleteTimer(gpuTimer));
	cutilCheckError(cutDeleteTimer(kernelTimer));

	cudaThreadExit();
}
