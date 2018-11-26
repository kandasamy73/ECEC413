/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <compact_stream_kernel.cu>

#define NUM_ELEMENTS 512

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void compact_stream(void);

// regression test functionality
extern "C" unsigned int compare( const float* reference, const float* data, const unsigned int len);
extern "C" void compute_scan_gold( float* reference, float* idata, const unsigned int len);
extern "C" void compact_stream_gold(float *reference, float *idata, unsigned int *len);
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
    compact_stream();
    cutilExit(argc, argv);
}

void compact_stream(void) 
{
    unsigned int num_elements = NUM_ELEMENTS;
    unsigned int timer;

    cutilCheckError( cutCreateTimer(&timer));
    
    const unsigned int mem_size = sizeof( float) * num_elements;

    // allocate host memory to store the input data
    float* h_data = (float*) malloc( mem_size);
      
    // initialize the input data on the host to be integer values
    // between 0 and 1000, both positive and negative
	 srand(time(NULL));
	 float rand_number;
    for( unsigned int i = 0; i < num_elements; ++i) {
		rand_number = rand()/(float)RAND_MAX;
		if(rand_number > 0.5) 
			h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
		else 
			h_data[i] = -floorf(1000*(rand()/(float)RAND_MAX));
    }

    // Compute reference solution. The function compacts the stream and stores the length of the new steam in num_elements
    float* reference = (float*) malloc( mem_size);  
    compact_stream_gold(reference, h_data, &num_elements);

  	// Add your code to perform the stream compaction on the GPU

  
	// Compare the reference solution with the GPU-based solution
	float epsilon = 0.0f;
    unsigned int result_regtest = cutComparefe( reference, h_data, num_elements, epsilon);
    printf( "%s: Test %s\n", "compact_stream", (1 == result_regtest) ? "PASSED" : "FAILED");

    // cleanup memory
    free( h_data);
    free( reference);
    cutilCheckError(cutDeleteTimer(timer));
  }
