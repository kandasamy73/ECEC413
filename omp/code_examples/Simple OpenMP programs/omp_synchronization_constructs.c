/* This code computes the vector dot product A.*B using OpenMP. Illustrates the use of various synchronization constructs 
 * available in OpenMP.
 *
 * Author: Naga Kandasamy
 * Date created: 4/24/2011
 * Date of last update: 10/06/2014
 *
 * Compile as follows: gcc -fopenmp omp_synchronization_constructs.c -o omp_synchronization_constructs -std=c99
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define NUM_THREADS 4
#define NUM_ELEMENTS 100000000

// Function prototypes
float compute_gold (float *, float *, int);
float compute_using_openmp_v1 (float *, float *, int);
float compute_using_openmp_v2 (float *, float *, int);
float compute_using_openmp_v3 (float *, float *, int);
float compute_using_openmp_v4 (float *, float *, int);
float compute_using_openmp_v5 (float *, float *, int);
void *dot_product (void *);

int
main (int argc, char **argv)
{
  if (argc != 1){
      printf ("Usage: This program takes no arguments. \n");
      exit (1);
  }
  
  int num_elements = NUM_ELEMENTS;	/* Obtain the size of the vector. */

  /* Create the vectors A and B and fill them with random numbers between [-.5, .5]. */
  printf("Creating two random vectors with %d elements each...", num_elements);
  float *vector_a = (float *) malloc (sizeof (float) * num_elements);
  float *vector_b = (float *) malloc (sizeof (float) * num_elements);
  srand (time (NULL));		/* Seed the random number generator. */
  for (int i = 0; i < num_elements; i++){
      vector_a[i] = ((float) rand () / (float) RAND_MAX) - 0.5;
      vector_b[i] = ((float) rand () / (float) RAND_MAX) - 0.5;
    }
  printf("done. \n \n");

  /* Compute the dot product using the reference, single-threaded solution. */
  printf("Dot product of vectors with %d elements using the single-threaded implementation. \n", num_elements);
  struct timeval start, stop;
  gettimeofday (&start, NULL);
  float reference = compute_gold (vector_a, vector_b, num_elements);
  gettimeofday (&stop, NULL);

  printf ("Reference solution = %f. \n", reference);
  printf ("Execution time = %fs. \n \n",
	  (float) (stop.tv_sec - start.tv_sec +
		   (stop.tv_usec - start.tv_usec) / (float) 1000000));

  float omp_result; 
  printf("Dot product of vectors with %d elements using OpenMP with the critical construct. \n", num_elements);
  gettimeofday (&start, NULL);
  omp_result = compute_using_openmp_v1 (vector_a, vector_b, num_elements);
  gettimeofday (&stop, NULL);

  printf ("OpenMP solution = %f. \n", omp_result);
  printf ("Execution time = %fs. \n \n",
	  (float) (stop.tv_sec - start.tv_sec +
		   (stop.tv_usec - start.tv_usec) / (float) 1000000));

  printf("Dot product of vectors with %d elements using OpenMP with the atomic construct. \n", num_elements);
  gettimeofday (&start, NULL);
  omp_result = compute_using_openmp_v2 (vector_a, vector_b, num_elements);
  gettimeofday (&stop, NULL);

  printf ("OpenMP solution = %f. \n", omp_result);
  printf ("Execution time = %fs. \n \n",
	  (float) (stop.tv_sec - start.tv_sec +
		   (stop.tv_usec - start.tv_usec) / (float) 1000000));

  printf("Dot product of vectors with %d elements using OpenMP with locks. \n", num_elements);
  gettimeofday (&start, NULL);
  omp_result = compute_using_openmp_v3 (vector_a, vector_b, num_elements);
  gettimeofday (&stop, NULL);

  printf ("OpenMP solution = %f. \n", omp_result);
  printf ("Execution time = %fs. \n \n",
	  (float) (stop.tv_sec - start.tv_sec +
		   (stop.tv_usec - start.tv_usec) / (float) 1000000));

  printf("Dot product of vectors with %d elements using OpenMP with barrier and master constructs. \n", num_elements);
  gettimeofday (&start, NULL);
  omp_result = compute_using_openmp_v4 (vector_a, vector_b, num_elements);
  gettimeofday (&stop, NULL);

  printf ("OpenMP solution = %f. \n", omp_result);
  printf ("Execution time = %fs. \n \n",
	  (float) (stop.tv_sec - start.tv_sec +
		   (stop.tv_usec - start.tv_usec) / (float) 1000000));

  printf("Dot product of vectors with %d elements using OpenMP with the reduction clause. \n", num_elements);
  gettimeofday (&start, NULL);
  omp_result = compute_using_openmp_v5 (vector_a, vector_b, num_elements);
  gettimeofday (&stop, NULL);

  printf ("OpenMP solution = %f. \n", omp_result);
  printf ("Execution time = %fs. \n \n",
	  (float) (stop.tv_sec - start.tv_sec +
		   (stop.tv_usec - start.tv_usec) / (float) 1000000));


  free ((void *) vector_a);
  free ((void *) vector_b);
}

/* This function computes the reference soution using a single thread. */
float
compute_gold (float *vector_a, float *vector_b, int num_elements)
{
  double sum = 0.0;
  for (int i = 0; i < num_elements; i++)
    sum += vector_a[i] * vector_b[i];

  return (float) sum;
}

/* This function computes the dot product using multiple threads. It illustrates the use of the "critical" construct in OpenMP. */
float
compute_using_openmp_v1 (float *vector_a, float *vector_b, int num_elements)
{
  int i;
  double sum = 0.0;		/* Variable to hold the final dot product. */
  double local_sum;

#pragma omp parallel private(i, local_sum) shared(sum) num_threads(NUM_THREADS)
  {
    local_sum = 0.0;
    /* Parallelize the iterations of the for loop over the available threads. */
#pragma omp for
    for (i = 0; i < num_elements; i++){
	local_sum += vector_a[i] * vector_b[i];
    }		

    /* The CRITICAL directive specifies a region of code that must be executed by only one thread at a time. If a thread is currently 
	* executing inside a CRITICAL region and another thread reaches that CRITICAL region and attempts to execute it, it will block 
	* until the first thread exits that CRITICAL region. 
    * */
#pragma omp critical
    {
      sum += local_sum;
      printf ("Thread ID %d: local_sum = %f, sum = %f\n", omp_get_thread_num (), local_sum, sum);
    }       /* End critical region. */
  }         /* End parallel region */

  return (float) sum;
}


/* This function computes the dot product using multiple threads. It illustrates the use of the "atomic" construct in OpenMP. 
	* The ATOMIC directive specifies that a specific memory location must be updated atomically, rather than letting multiple threads 
	* attempt to write to it. In essence, this directive provides a mini-CRITICAL section. This directive applies only to a single, 
	* immediately following statement. */
float
compute_using_openmp_v2 (float *vector_a, float *vector_b, int num_elements)
{
  int i;
  double sum = 0.0;
  double local_sum;

#pragma omp parallel private(i, local_sum) shared(sum) num_threads(NUM_THREADS)
  {
    local_sum = 0.0;

#pragma omp for
    for (i = 0; i < num_elements; i++){
        local_sum += vector_a[i] * vector_b[i];
    }				

#pragma omp critical
    printf ("Thread ID %d: local_sum = %f, sum = %f\n", omp_get_thread_num (), local_sum, sum);

    /* Only a single statement can follow after the atomic construct. */
#pragma omp atomic
    sum += local_sum;
  }         /* End parallel region. */

  return (float) sum;
}

/* This function computes the dot product using multiple threads. It illustrates the use of the "lock" construct in OpenMP. */
float
compute_using_openmp_v3 (float *vector_a, float *vector_b, int num_elements)
{
  int i;
  double sum = 0.0;
  omp_lock_t lock;          /* The lock variable. */
  double local_sum;

  omp_init_lock (&lock);	/* Initialize the lock. */

#pragma omp parallel private(i, local_sum) shared(lock) num_threads(NUM_THREADS)
  {
    local_sum = 0.0;

#pragma omp for
    for (i = 0; i < num_elements; i++){
        local_sum += vector_a[i] * vector_b[i];
    }		

    omp_set_lock (&lock);	/* Attempt to gain the lock. Force the executing thread to block until the specified lock is available. */

    sum += local_sum;		/* Accumulate into the global sum */
    printf ("Thread ID %d: local_sum = %f, sum = %f\n", omp_get_thread_num (), local_sum, sum);

    omp_unset_lock (&lock);	    /* Release the lock. */
  }				                /* End parallel region. */

  omp_destroy_lock (&lock);	    /* Destroy the lock. */

  return (float) sum;
}

/* This function computes the dot product using multiple threads. It illustrates the use of the "barrier" and "master" constructs in OpenMP. */
float
compute_using_openmp_v4 (float *vector_a, float *vector_b, int num_elements)
{
  int i, j;
  double sum;
  double local_sum[NUM_THREADS];
  int thread_id;

#pragma omp parallel num_threads(NUM_THREADS) private(thread_id, i, j) shared(sum, local_sum)
  {
    thread_id = omp_get_thread_num ();

    /* The MASTER directive specifies a region that is to be executed only by the master thread of the team. 
     * All other threads on the team skip this section of code. Note that there is no implied barrier associated 
     * with this directive. 
     * */
#pragma omp master
    {
      printf ("Master thread %d is performing some initialization\n", omp_get_thread_num ());
      sum = 0.0;
      for (i = 0; i < NUM_THREADS; i++)
          local_sum[i] = 0.0;
    }
    
    /* Synchronize all threads in the team. When a BARRIER directive is reached, a thread will wait at that point until all other 
     * threads have reached that barrier. All threads then resume executing in parallel the code that follows the barrier. */
#pragma omp barrier

#pragma omp for
    for (i = 0; i < num_elements; i++){
        local_sum[thread_id] += vector_a[i] * vector_b[i];
    }			

    printf ("Thread %d is at the barrier. local_sum = %f\n", thread_id, local_sum[thread_id]);

    /* The barrier. Each barrier in your code must be encountered by all threads or none at all. */
#pragma omp barrier
    
    /* The master generates the final sum. */
#pragma omp master
    {
      printf ("Master thread %d is computing the final sum\n", omp_get_thread_num ());
      for (j = 0; j < NUM_THREADS; j++)
          sum += local_sum[j];
    }
  }				/* End parallel region. */

  return (float) sum;
}

/* This function computes the dot product using multiple threads. It illustrates the use of the reduction and schedule clauses. */
float
compute_using_openmp_v5 (float *vector_a, float *vector_b, int num_elements)
{
  int i;
  double sum = 0.0;

  omp_set_num_threads (NUM_THREADS);	/* Set the number of threads. */

  /* The REDUCTION clause performs a reduction on the variables that appear in its list. A private copy for each list variable 
   * is created for each thread. At the end of the reduction, the reduction variable is applied to all private copies of the shared 
   * variable, and the final result is written to the global shared variable. 
   *
   * The SCHEDULE clause describes how iterations of the loop are divided among the threads in the team. The default schedule is 
   * implementation dependent.
   * 
   * STATIC: Loop iterations are divided into pieces of size chunk and then statically assigned to threads. If chunk is not specified, 
   * the iterations are evenly (if possible) divided contiguously among the threads.
   *
   * DYNAMIC: Loop iterations are divided into pieces of size chunk, and dynamically scheduled among the threads; when a thread finishes 
   * one chunk, it is dynamically assigned another. The default chunk size is 1.
   *
   * GUIDED: Iterations are dynamically assigned to threads in blocks as threads request them until no blocks remain to be assigned. 
   * Similar to DYNAMIC except that the block size decreases each time a parcel of work is given to a thread. The size of the initial 
   * block is proportional to: number_of_iterations / number_of_threads. 
   * Subsequent blocks are proportional to number_of_iterations_remaining / number_of_threads
   * The chunk parameter defines the minimum block size. The default chunk size is 1.
   * */
#pragma omp parallel for reduction(+:sum) schedule(guided)
  for (i = 0; i < num_elements; i++){
      sum += vector_a[i] * vector_b[i];
  }

  return (float) sum;
}
