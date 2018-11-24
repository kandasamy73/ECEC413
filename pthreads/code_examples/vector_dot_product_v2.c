/* Vector dot product A.B using pthreads. This version shows how to use lock variables. 
 * Author: Naga Kandasamy
 * Date: 4/4/2011
 * Compile as follows: gcc -o vector_dot_product_v2 vecctor_dot_product_v2.c -std=c99 -lpthread -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>

// Magic values 
#define NUM_THREADS 64

// Data structures
typedef struct args_for_thread_s{
		  int thread_id; // The thread ID
		  int num_elements; // Number of elements in the vector
		  float *vector_a; // Starting address of vector_a
		  float *vector_b; // Starting address of vector_b
		  int offset; // Starting offset for thread within the vectors 
		  int chunk_size; // Chunk size
		  double *sum; // Location of the shared variable sum
		  pthread_mutex_t *mutex_for_sum; // Location of the lock variable protecting sum
} ARGS_FOR_THREAD;

// Function prototypes
float compute_gold(float *, float *, int);
float compute_using_pthreads(float *, float *, int);
void *dot_product(void *);
void print_args(ARGS_FOR_THREAD *);


int main(int argc, char **argv)
{
	if(argc != 2){
		printf("Usage: vector_dot_product <num elements> \n");
		exit(1);
	}
	int num_elements = atoi(argv[1]); // Obtain the size of the vector 

	// Create the vectors A and B and fill them with random numbers between [-.5, .5]
	float *vector_a = (float *)malloc(sizeof(float) * num_elements);
	float *vector_b = (float *)malloc(sizeof(float) * num_elements); 
	srand(time(NULL)); // Seed the random number generator
	for(int i = 0; i < num_elements; i++){
		vector_a[i] = ((float)rand()/(float)RAND_MAX) - 0.5;
		vector_b[i] = ((float)rand()/(float)RAND_MAX) - 0.5;
	}
	// Compute the dot product using the reference, single-threaded solution
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	float reference = compute_gold(vector_a, vector_b, num_elements); 
	gettimeofday(&stop, NULL);

	printf("Reference solution = %f. \n", reference);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	printf("\n");

	gettimeofday(&start, NULL);
	float result = compute_using_pthreads(vector_a, vector_b, num_elements);
	gettimeofday(&stop, NULL);

	printf("Pthread solution = %f. \n", result);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	printf("\n");

	// Free memory here 
	free((void *)vector_a);
	free((void *)vector_b);

	pthread_exit(NULL);
}

/* This function computes the reference soution using a single thread. */
float compute_gold(float *vector_a, float *vector_b, int num_elements)
{
	double sum = 0.0;
	for(int i = 0; i < num_elements; i++)
			  sum += vector_a[i] * vector_b[i];
	
	return (float)sum;
}


/* This function computes the dot product using multiple threads. This version uses mutex locks. */
float compute_using_pthreads(float *vector_a, float *vector_b, int num_elements)
{
		  pthread_t thread_id[NUM_THREADS]; // Data structure to store the thread IDs
		  pthread_attr_t attributes; // Thread attributes
		  pthread_mutex_t mutex_for_sum; // Lock for the shared variable sum

		  pthread_attr_init(&attributes); // Initialize the thread attributes to the default values 
		  pthread_mutex_init(&mutex_for_sum, NULL); // Initialize the mutex

		  // Allocate memory on the heap for the required data structures and create the worker threads
		  int i;
		  double sum = 0; 
		  ARGS_FOR_THREAD *args_for_thread[NUM_THREADS];
		  int chunk_size = (int)floor((float)num_elements/(float)NUM_THREADS); // Compute the chunk size
		  for(i = 0; i < NUM_THREADS; i++){
					 args_for_thread[i] = (ARGS_FOR_THREAD *)malloc(sizeof(ARGS_FOR_THREAD));
					 args_for_thread[i]->thread_id = i; // Provide thread ID
					 args_for_thread[i]->num_elements = num_elements; 
					 args_for_thread[i]->vector_a = vector_a; // Starting address of vector_a
					 args_for_thread[i]->vector_b = vector_b; // Starting address of vector_b
					 args_for_thread[i]->offset = i * chunk_size; // Starting offset for thread within the vectors 
					 args_for_thread[i]->chunk_size = chunk_size; // Chunk size
					 args_for_thread[i]->sum = &sum;
					 args_for_thread[i]->mutex_for_sum = &mutex_for_sum;
		  }

		  for(i = 0; i < NUM_THREADS; i++)
					 pthread_create(&thread_id[i], &attributes, dot_product, (void *)args_for_thread[i]);
					 
		  // Wait for the workers to finish
		  for(i = 0; i < NUM_THREADS; i++)
					 pthread_join(thread_id[i], NULL);
		  

		  // Free data structures
		  for(i = 0; i < NUM_THREADS; i++)
					 free((void *)args_for_thread[i]);

		  return (float)sum;
}

/* This function is executed by each thread to compute the overall dot product */
void *dot_product(void *args)
{
		  ARGS_FOR_THREAD *args_for_me = (ARGS_FOR_THREAD *)args; // Typecast the argument to a pointer the the ARGS_FOR_THREAD structure
		  // print_args(args_for_me);
		  
		  // Compute the partial sum that this thread is responsible for
		  double partial_sum = 0.0; 
		  if(args_for_me->thread_id < (NUM_THREADS - 1)){
					 for(int i = args_for_me->offset; i < (args_for_me->offset + args_for_me->chunk_size); i++)
								partial_sum += args_for_me->vector_a[i] * args_for_me->vector_b[i];
		  } 
		  else{ // This takes care of the number of elements that the final thread must process
					  for(int i = args_for_me->offset; i < args_for_me->num_elements; i++)
								 partial_sum += args_for_me->vector_a[i] * args_for_me->vector_b[i];
		  }

		  // Accumulate partial sums into the shared variable 
		  pthread_mutex_lock(args_for_me->mutex_for_sum);
		  *(args_for_me->sum) += partial_sum;
		  pthread_mutex_unlock(args_for_me->mutex_for_sum);

		  pthread_exit((void *)0);
}

/* Helper function */
void print_args(ARGS_FOR_THREAD *args_for_thread)
{
		  printf("Thread ID: %d \n", args_for_thread->thread_id);
		  printf("Num elements: %d \n", args_for_thread->num_elements); 
		  printf("Address of vector A on heap: %x \n", args_for_thread->vector_a);
		  printf("Address of vector B on heap: %x \n", args_for_thread->vector_b);
		  printf("Offset within the vectors for thread: %d \n", args_for_thread->offset);
		  printf("Chunk size to operate on: %d \n", args_for_thread->chunk_size);
		  printf("\n");
}

