/* A hello word program that uses OpenMP. 
 * For a good tutorial on OpenMP, please refer to 
 * computing.llnl.gov/tutorials/openMP
 * 
 * Author: Naga Kandasamy
 * Date created: 04/15/2011
 * Date of last update: 10/06/2014
 * Compile as follows: gcc -o omp_hello_world omp_hello_world_v1.c -fopenmp -std=c99
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void Hello (void);

int
main (int argc, char **argv)
{
  if (argc != 2){
      printf ("Usage: hello_world <num threads> \n");
      exit (0);
  }
  
  int thread_count = atoi(argv[1]);	/* Number of threads to create. */

  /* OpenMP block here. */
#pragma omp parallel num_threads(thread_count)
  {
    Hello ();
  } /* OpemMP enforces an implicit barrier sync at the end of the parallel construct. */

  return 0;
}

void
Hello (void)
{
  int my_id = omp_get_thread_num ();	/* Obtain thread ID. */
  int thread_count = omp_get_num_threads ();

  printf ("Hello from thread %d of %d threads. \n", my_id, thread_count);
}
