/* A hello word program that uses OpenMP. This provides an example of the parallel construct or MIMD style 
 * parallelism. 
 * Author: Naga Kandasamy
 * Date created: 04/15/2011
 * Date last modified: 10/06/2014
 * Compile as follows: gcc -o omp_hello_world omp_hello_worid_v2.c -fopenmp -std=c99
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int
main (int argc, char **argv)
{
  if (argc != 2){
      printf ("Usage: hello_world <num threads> \n");
      exit (0);
  }
  
  int thread_count = atoi (argv[1]); /* Number of threads to be created. */

  /* OpenMP block here. */
#pragma omp parallel num_threads(thread_count)
  {
    int thread_id = omp_get_thread_num ();

    printf ("The parallel region executed by thread %d. \n",
	    omp_get_thread_num ());

    if (thread_id == 4)
      {
	printf ("Thread %d does things differently. \n",
		omp_get_thread_num ());
      }

    if (thread_id == 2)
      {
	printf ("Thread %d does things differently. \n",
		omp_get_thread_num ());
      }
  } /* Barrier sync here. */

  return 0;
}
