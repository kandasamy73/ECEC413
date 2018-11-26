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

#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_

__global__ void flag_and_scan(float *data, int *scans, int *flags, int *sums)
{
	// Allocate shared memory for flags
	__shared__ int flag[MAX_THREADS];

	// Allocate shared memory for scan
	__shared__ int scan[2 * MAX_THREADS];

	// Find position within vector
	int threadID = threadIdx.x;
	int blockID = blockIdx.x;
	int xInVector = blockID * MAX_THREADS + threadID;

	// Populate shared memory
	flag[threadID] = (data[xInVector] > 0) ? 1 : 0;
	__syncthreads();
	scan[threadID] = (threadID > 0) ? flag[threadID - 1] : 0;
	__syncthreads();

	// Initialize scan variables
	int pOut = 0;
	int pIn = 1;

	// Perform scan
	for (int offset = 1; offset < MAX_THREADS; offset <<= 1) {
		pOut = 1 - pOut;
		pIn = 1 - pOut;
		__syncthreads();

		scan[pOut * MAX_THREADS + threadID] = scan[pIn * MAX_THREADS + threadID];

		if (threadID >= offset)
			scan[pOut * MAX_THREADS + threadID] += scan[pIn * MAX_THREADS + threadID - offset];
	}
	__syncthreads();

	// Store information for compact stream
	flags[xInVector] = flag[threadID];
	scans[xInVector] = scan[pOut * MAX_THREADS + threadID];

	if (threadID == LAST_THREAD)
		sums[blockID] = scan[pOut * MAX_THREADS + threadID] + flag[threadID];
}

__global__ void scan_sums(int *sums, int *increments, int n)
{
	// Allocate shared memory for scan
	__shared__  int scan[2 * MAX_THREADS];

	// Find position within vector
	int threadID = threadIdx.x;
	int blockID = blockIdx.x;
	int xInVector = blockID * MAX_THREADS + threadID;

	// Populate shared memory
	scan[threadID] = (threadID > 0) ? sums[xInVector - 1] : 0;
	__syncthreads();

	int last_grid = gridDim.x - 1;
	int limit = (blockID == last_grid) ? n - (blockID * MAX_THREADS) : MAX_THREADS;
	int last_thread = limit - 1;
	__syncthreads();

	// Initialize scan variables
	int pOut = 0;
	int pIn = 1;

	// Perform scan
	for (int offset = 1; offset < MAX_THREADS; offset <<= 1) {
		pOut = 1 - pOut;
		pIn = 1 - pOut;
		__syncthreads();

		scan[pOut * MAX_THREADS + threadID] = scan[pIn * MAX_THREADS + threadID];

		if (threadID >= offset)
			scan[pOut * MAX_THREADS + threadID] += scan[pIn * MAX_THREADS + threadID - offset];
	}
	__syncthreads();

	// Store increments per block
	if (threadID > 0)
		increments[xInVector] = scan[pOut * MAX_THREADS + threadID];
	__syncthreads();

	if (threadID == 0 && blockID != last_grid)
		increments[xInVector + limit] = scan[pOut * MAX_THREADS + last_thread] + sums[xInVector + last_thread];
}

__global__ void add_increments(int *scans, int *increments)
{
	// Find position within vector
	int blockID = blockIdx.x;
	int blockCount = (blockID + MAX_THREADS - 1) * MAX_THREAD_DIVISOR;
	int xInVector = blockID * MAX_THREADS + threadIdx.x;

	// Update sums
	int current = scans[xInVector];
	for (int i = 1; i < blockCount; i++) {
		current += increments[i * MAX_THREADS];
	}
	__syncthreads();
	scans[xInVector] = current + increments[blockID];
}

__global__ void compact_stream(float *output, float *input, int *flags, int *scans)
{
	// Find position within vector
	int xInVector = blockIdx.x * MAX_THREADS + threadIdx.x;

	// Compact stream
	int index = scans[xInVector];
	if (flags[xInVector] == 1)
		output[index] = input[xInVector];
}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
