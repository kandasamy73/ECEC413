#include <stdio.h>
#include <math.h>
#include <float.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" float compute_gold_sp( float *, float *, unsigned int);
extern "C" double compute_gold_dp( float *, float *, unsigned int);
extern "C" float compute_gold_kahan(float *, float *, unsigned int);


float compute_gold_sp( float* A, float* B, unsigned int num_elements)
{
  unsigned int i;
  float dot_product = 0.0; 

  for( i = 0; i < num_elements; i++) 
			 dot_product += A[i] * B[i];

  return dot_product;
}


double compute_gold_dp( float* A, float* B, unsigned int num_elements)
{
  unsigned int i;
  double dot_product = 0.0; 

  for( i = 0; i < num_elements; i++) 
			 dot_product += A[i] * B[i];

  return dot_product;
}



/* This function shows how to use the Kahan summation allllgorithm to reduce precision errors during single-precision floating point arithmetic. */
float compute_gold_kahan(float *A, float *B, unsigned int num_elements)
{
		  unsigned int i;
		  float dot_product = 0.0;
		  volatile float recovered_bits = 0.0;
		  float y, temp;

		  for(i = 0; i < num_elements; i++)
		  {
					 y = (A[i]*B[i]) - recovered_bits;
					 temp = dot_product + y; // If dot_product is big, then the lower-order bits of y are lost
					 recovered_bits = (temp - dot_product) - y; // (temp - dot_product) recovers the higher-order bits of y; subtracting y recovers -(lower part of y)
					 dot_product = temp; // Next time around, the lost lower-order bits will be added to y in a fresh attempt
		  }
		  return dot_product;
}



