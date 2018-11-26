#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <vector_reduction_kernel.cu>

#define NUM_ELEMENTS 5000000

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);
float computeOnDevice(float* h_data, int array_mem_size);

extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
	runTest( argc, argv);
	return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//  Run naive scan test
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
	CUT_DEVICE_INIT(1, "NULL");
	int num_elements = NUM_ELEMENTS;


	// -- Parse Command Line -------------------------------------------
	if (argc != 1)
		num_elements = atoi(argv[1]);
	// -----------------------------------------------------------------


	// -- Generate Input Data ------------------------------------------
	size_t array_mem_size = sizeof( float) * num_elements;
	float* h_data = (float*) malloc( array_mem_size);
	
	srand(time(NULL)); // Seed the pseudo-random number generator
	printf ("\nNum Elements: %i\n\n", num_elements);
	for( unsigned int i = 0; i < num_elements; ++i) 
		h_data[i] = floorf(2*(rand()/(float)RAND_MAX));
	// -----------------------------------------------------------------

	unsigned int timer;
	cutCreateTimer(&timer);
	cutStartTimer(timer);

	// -- Compute CPU Solution -----------------------------------------
	float reference = 0.0f;  
	computeGold(&reference , h_data, num_elements);

	cutStopTimer(timer);
	float time = 1e-3 * cutGetTimerValue(timer);
	printf("CPU run time:        %0.10f s\n", time);

	// -----------------------------------------------------------------
    

	// -- Compute GPU Solution -----------------------------------------
	float result = computeOnDevice(h_data, num_elements);
	// -----------------------------------------------------------------


	// -- Check the GPU Results ----------------------------------------
	float epsilon = 0.0f;
	unsigned int result_regtest = (abs(result - reference) <= epsilon);
	printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
	printf( "device: %f\n", result);
	printf( "  host: %f\n", reference);
	// -----------------------------------------------------------------

	free( h_data);
}




float computeOnDevice(float* h_data, int num_elements)
{
	// -- Get GPU Pointer & Data Size ----------------------------------
	float* gpu_data;
	size_t gpu_data_size = num_elements * sizeof (float);
	// -----------------------------------------------------------------

	unsigned int gpu_timer, kernel_timer;
	cutCreateTimer(&gpu_timer);
	cutCreateTimer(&kernel_timer);

	cutStartTimer(gpu_timer);

	// -- Allocate GPU Memory & Copy Data Over -------------------------
	cudaMalloc ((void**)&gpu_data, gpu_data_size);
	cudaMemcpy (gpu_data, h_data, gpu_data_size, cudaMemcpyHostToDevice);
	// -----------------------------------------------------------------


	// -- Determine GPU Execution Configuration ------------------------
	int threads_per_block = 512;
	int thread_blocks = (int)ceil((float)num_elements/(float)threads_per_block);
	printf("Number of thread blocks needed to perform the reduction is %d. \n", thread_blocks);
	int grid_dimension_in_blocks = (int)ceil(sqrt((float)thread_blocks)); // Set up a 2D grid
	// -----------------------------------------------------------------

	// -- Define GPU Execution Config ----------------------------------
	printf("Setting up a grid of size %d x %d thread blocks \n", grid_dimension_in_blocks, grid_dimension_in_blocks);
	dim3 dimGrid  (grid_dimension_in_blocks, grid_dimension_in_blocks, 1);
	dim3 dimBlock (threads_per_block, 1, 1);
	int smemSize = threads_per_block * sizeof(float);
	// -----------------------------------------------------------------

	cutStartTimer(kernel_timer);

	// -- Perform the first set of partial sum reductions -------------------------------
	reduction <<<dimGrid, dimBlock, smemSize>>> (gpu_data, num_elements);
	// cudaThreadSynchronize();

	// -----------------------------------------------------------------
	
	// -- Perform more reductions if needed---------
	int num_partial_sums = thread_blocks;
	while(num_partial_sums > threads_per_block){
		// Determine the number of thread blocks needed to process this set of partial sums
		thread_blocks = (int)ceil((float)num_partial_sums/(float)threads_per_block);
		
		// Set up the 2D grid 
		grid_dimension_in_blocks = (int)ceil(sqrt((float)thread_blocks)); // Set up a 2D grid
		dim3 dimGrid  (grid_dimension_in_blocks, grid_dimension_in_blocks, 1);
	   dim3 dimBlock (threads_per_block, 1, 1);
      smemSize = threads_per_block * sizeof(float);
	
		// Perform the reduction 
		reduction <<<dimGrid, dimBlock, smemSize>>> (gpu_data, num_partial_sums);
		// cudaThreadSynchronize();

		// Update the number of partial sums
		num_partial_sums = thread_blocks;
	}

	// -- Perform the final reduction ------------------------------
	if(num_partial_sums <= threads_per_block){ 
		dim3 dimGrid (1, 1, 1);
		dim3 dimBlock (512, 1, 1);
		smemSize = 512 * sizeof (float);
		reduction <<<dimGrid, dimBlock, smemSize>>> (gpu_data, num_partial_sums);
	}

	cudaThreadSynchronize();
	cutStopTimer(kernel_timer);

	// -- Copy Result to CPU Memory ------------------------------------
	cudaMemcpy (h_data, gpu_data, sizeof(float), cudaMemcpyDeviceToHost);
	// -----------------------------------------------------------------

	cutStopTimer(gpu_timer);


	// -- Free the GPU Memory ------------------------------------------
	cudaFree (gpu_data); 
	// -----------------------------------------------------------------

	float GPU_time = 1e-3 * cutGetTimerValue(gpu_timer);
	float kernel_time = 1e-3 * cutGetTimerValue(kernel_timer);
	printf("Total GPU run time:  %0.10f s\n", GPU_time);
	printf("GPU Kernel run time: %0.10f s\n", kernel_time);
	printf("Overhead:            %0.10f s\n", GPU_time - kernel_time);

	return h_data[0];
}
