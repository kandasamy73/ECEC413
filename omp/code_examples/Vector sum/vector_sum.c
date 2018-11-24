/* Vector addition of two vectors using OpenMP. Shows the use of the nowait clause
 * Author: Naga Kandasamy
 * Date created: 4/4/2011
 * Date of last update: 10/06/2014
 * Compile as follows: gcc -fopenmp vector_sum.c -o vector_sum -std=c99
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define NUM_THREADS 4

// Function prototypes
float compute_using_openmp_v1 (float *, float *, float *, float *, int);
float compute_using_openmp_v2 (float *, float *, float *, float *, int);

int
main (int argc, char **argv)
{
  if (argc != 2){
      printf ("Usage: vector_sum <num elements> \n");
      exit (1);
    }
  int num_elements = atoi (argv[1]);	// Obtain the size of the vector 

  /* Create the vectors A and B and fill them with random numbers between [-.5, .5]. */
  printf("Creating the random vectors...");
  float *vector_a = (float *) malloc (sizeof (float) * num_elements);
  float *vector_b = (float *) malloc (sizeof (float) * num_elements);
  float *vector_c = (float *) malloc (sizeof (float) * num_elements);
  float *vector_d = (float *) malloc (sizeof (float) * num_elements);

  srand (time (NULL));		// Seed the random number generator
  for (int i = 0; i < num_elements; i++) {
      vector_a[i] = ((float) rand () / (float) RAND_MAX) - 0.5;
      vector_b[i] = ((float) rand () / (float) RAND_MAX) - 0.5;
      vector_c[i] = ((float) rand () / (float) RAND_MAX) - 0.5;
      vector_d[i] = ((float) rand () / (float) RAND_MAX) - 0.5;
    }

  printf("done. \n \n");

  /* Compute the vector sum using OpenMP. */      
  struct timeval start, stop;

  printf("Computing the vector sum using OpenMP. \n");
  gettimeofday (&start, NULL);
  
  float sum = compute_using_openmp_v1 (vector_a, vector_b, vector_c, vector_d, num_elements);
  printf("Sum = %f.\n", sum);
  
  gettimeofday (&stop, NULL);
  printf ("Execution time = %fs. \n \n",
	  (float) (stop.tv_sec - start.tv_sec +
		   (stop.tv_usec - start.tv_usec) / (float) 1000000));

  /* Compute the vector sum using OpenMP. Use the nowait clause to speed up the execution by reducing the 
   * barrier synchronization points. */
  printf("Computing the vector sum using OpenMP. Minimizing the barrier sync points using the nowait clause. \n");
  gettimeofday (&start, NULL);
  
  sum = compute_using_openmp_v2 (vector_a, vector_b, vector_c, vector_d, num_elements);
  printf("Sum = %f.\n", sum);
  
  gettimeofday (&stop, NULL);
  printf ("Execution time = %fs. \n \n",
	  (float) (stop.tv_sec - start.tv_sec +
		   (stop.tv_usec - start.tv_usec) / (float) 1000000));
  
  free ((void *) vector_a);
  free ((void *) vector_b);
  free ((void *) vector_c);
  free ((void *) vector_d);
}


/* This function computes the vector sum. Version one. */
float
compute_using_openmp_v1 (float *vector_a, 
        float *vector_b, 
        float *vector_c,
        float *vector_d, 
        int num_elements)
{
  int i, j;
  double sum = 0.0;

  omp_set_num_threads (NUM_THREADS);	// Set the number of threads

#pragma omp parallel private(i, j)
  {
#pragma omp for
    for (i = 0; i < num_elements; i++)
      vector_a[i] = vector_a[i] + vector_b[i];

#pragma omp for
    for (j = 0; j < num_elements; j++)
      vector_c[j] = vector_c[j] + vector_d[j];

#pragma omp for reduction(+: sum)
    for (i = 0; i < num_elements; i++)
      sum = sum + vector_a[i] + vector_c[i];
  }

  return (float) sum;
}


/* This function computes the vector sum. Version uses the nowait clause. */
float
compute_using_openmp_v2 (float *vector_a, 
        float *vector_b, 
        float *vector_c,
        float *vector_d, 
        int num_elements)
{
  int i, j;
  double sum = 0.0;

  omp_set_num_threads (NUM_THREADS);	// Set the number of threads

#pragma omp parallel private(i, j)
  {
#pragma omp for  nowait
    for (i = 0; i < num_elements; i++)
      vector_a[i] = vector_a[i] + vector_b[i];

#pragma omp for nowait
    for (j = 0; j < num_elements; j++)
      vector_c[j] = vector_c[j] + vector_d[j];
#pragma omp barrier

#pragma omp for reduction(+: sum) nowait
    for (i = 0; i < num_elements; i++)
      sum = sum + vector_a[i] + vector_c[i];
  }

  return (float) sum;
}
