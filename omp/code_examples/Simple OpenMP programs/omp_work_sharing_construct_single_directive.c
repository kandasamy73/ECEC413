/* OpenMP example of the work sharing construct. This construct shows how to use the single directive. 
 * Author: Naga Kandasamy, 04/15/2011
 * Date last modified: 10/06/2014
 *
 * Compile as follows: 
 * gcc -o work_sharing_construct_with_single_directive  work_sharing_construct_with_single_directive.c -fopenmp -std=c99
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int
main (int argc, char **argv)
{
  if (argc != 2){
      printf ("Usage: openmp_example_2 <num threads> \n");
      exit (0);
  }
  
  int thread_count = atoi (argv[1]);

  int a;
  int b[10];
  int i;
  int n = 10;

  /* Start of parallel region. */
#pragma omp parallel num_threads(thread_count) shared(a, b) private(i)
  {
	  /* Only a single thread in the team executes this code. Useful when dealing with sections of code that are 
	   * not thread safe (such as I/O)*/
#pragma omp single
    {
      a = 10;
      printf ("Single construct executed by thread %d. \n", omp_get_thread_num ());

    } /* This is an implicit barrier sync. */

    /* We parallelize the iterations within the for loop over the available threads. */ 
#pragma omp for
    for (i = 0; i < n; i++){
        b[i] = a;
    }
  }     /* End of parallel region. */

  for (i = 0; i < n; i++)
    printf ("b[%d] = %d. \n", i, b[i]);

  return 0;
}
