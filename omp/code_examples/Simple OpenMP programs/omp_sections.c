/* OpenMP example of the section construct. 
 * Author: Naga Kandasamy
 * Date created: 04/15/2011
	* Date last modified: 10/06/2014
 * Compile as follows: gcc -o openmp_sections openmp_sections.c -fopenmp -std=c99
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void function_a (void);
void function_b (void);
void function_c (void);

int
main (int argc, char **argv)
{
  if (argc != 2){
      printf ("Usage: openmp_example_3 <num threads> \n");
      exit (0);
    }
  
  int thread_count = atoi (argv[1]);	// Obtain the number of threads to be created as a command variable argument


  /* Start of parallel region. The SECTIONS directive is a non-iterative work-sharing construct, specifying that the enclosed 
   * section(s) of code are to be divided among the threads in the team. Independent SECTION directives are nested within a SECTIONS 
   * directive and each SECTION is executed once by a thread in the team. Different sections may be executed by different threads. 
   * It is possible for a thead to execute more than one section if it is quick enough and the implementation permits such. */
#pragma omp parallel num_threads(thread_count)
  {
#pragma omp sections
    {
#pragma omp section
      function_a ();

#pragma omp section
      function_b ();

#pragma omp section
      function_c();
    }
  } /* End of parallel region. */               
  return 0;
}

void
function_a (void)
{
  printf ("Thread %d is executing function A. \n", omp_get_thread_num ());
}

void
function_b (void)
{
  printf ("Thread %d is executing function B. \n", omp_get_thread_num ());
}

void 
function_c (void)
{							
    printf ("Thread %d is executing function C. \n", omp_get_thread_num());
}
