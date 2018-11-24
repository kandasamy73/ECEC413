/* 
   Author: Naga Kandasamy, 1/21/2009

   Program to illustrate basic thread management operations. 

   Compile as follows: 
   gcc -o multiple_threads multiple_threads.c -std=c99 -lpthread 
*/

#define _REENTRANT // Make sure the library functions are MT (muti-thread) safe
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

#define NUM_THREADS 8

/* Function prototype for the thread routines */
void *my_func(void *);


// Declare a structure data type that will be used to pass arguments to the worker threads
typedef struct args_for_thread_t{
		  int threadID; // thread ID
		  int arg1; // First argument
		  float arg2; // Second argument
		  int processing_time; // Third argument
} ARGS_FOR_THREAD; 

int main()
{
		  pthread_t main_thread;
		  pthread_t worker_thread[NUM_THREADS];
		  ARGS_FOR_THREAD *args_for_thread;
		  int i;

		  main_thread = pthread_self(); // Returns the thread ID for the calling thread 
		  printf("Main thread = %u is creating %d worker threads \n", (int)main_thread, NUM_THREADS);
		  // Create NUM_THREADS worker threads and ask them to execute my_func that takes a structure as an argument
		  for(i = 0; i < NUM_THREADS; i++){

					 args_for_thread = (ARGS_FOR_THREAD *)malloc(sizeof(ARGS_FOR_THREAD)); // Allocate memory for the structure that will be used to pack the arguments
					 args_for_thread->threadID = i; // Fill the structure with some dummy arguments
					 args_for_thread->arg1 = 5; 
					 args_for_thread->arg2 = 2.5;
					 args_for_thread->processing_time = 2; 

					 if((pthread_create(&worker_thread[i], NULL, my_func, (void *)args_for_thread)) != 0){
								printf("Cannot create thread \n");
								exit(0);
					 }
		  }

		  // Wait for all the worker threads to finish 
		  for(i = 0; i < NUM_THREADS; i++)
					 pthread_join(worker_thread[i], NULL);

		  printf("Main thread exiting \n");
		  pthread_exit((void *)main_thread);
}


/* Function that will be executed by all the worker threads */
void *my_func(void *this_arg){
		 ARGS_FOR_THREAD *args_for_me = (ARGS_FOR_THREAD *)this_arg; // Typecast the argument passed to this function to the appropriate type
		
		 // Simulate some processing
		 printf("Thread %d is using args %d and %f \n", args_for_me->threadID, args_for_me->arg1, args_for_me->arg2);
		 int processing_time = args_for_me->processing_time;
		 sleep(processing_time);
		 printf("Thread %d is done \n", args_for_me->threadID);

		 free((void *)args_for_me); // Free up the structure
		 pthread_exit(NULL);
}







