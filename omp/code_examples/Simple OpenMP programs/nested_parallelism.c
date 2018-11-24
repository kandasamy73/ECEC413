/* OpenMP program showing the use of nested parallelism. Unfortunately, the implementation on Linux does not support 
 * nested parallelism.
 * 
 * Author: Naga Kandasamy
 * Date: 04/25/2011
 * 
 * Compile as follows: gcc -o nested_parallelism nested_parallelism.c -fopenmp -std=c99
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


void report_num_threads(int level)
{
#pragma omp single 
	printf("Level %d: number of threads in the team = %d. \n", level, omp_get_num_threads());
}

int
main (int argc, char **argv)
{

  if (argc != 2)
    {
      printf ("Usage: nested_parallelism <num threads> \n");
      exit (0);
    }
  int thread_count = atoi (argv[1]);	// Obtain the number of threads to be created as a command variable argument

  // Check if the OpenMP implementation supports nested parallelism
  printf ("Nested parallelism is %s \n",
	  omp_get_nested ()? "supported" : "not supported");

  int thread_id;
  // OpenMP block here
#pragma omp parallel num_threads(thread_count) private(thread_id)
  {
    thread_id = omp_get_thread_num ();
    report_num_threads(1); // Report the number of threads at this level
    printf ("Thread %d executes outer parallel region \n", thread_id);

#pragma omp parallel num_threads(thread_count) firstprivate(thread_id)
    {
	    report_num_threads(2);
      printf ("   Parent Thread %d: Thread %d executes inner parallel region \n",
	      thread_id, omp_get_thread_num ());
    }
  }

  return 0;
}
