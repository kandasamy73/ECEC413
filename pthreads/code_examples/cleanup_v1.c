/* Example code to illustrate the use of a cleanup handler that is invoked upon thread cancellation.
 * Author: Naga Kandasamy
 * Date: 04/10/2011
 * Compile as follows: gcc -o cleanup_v1 cleanup_v1.c -lpthread -std=c99 
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#define NUM_THREADS 5

// The dummy control structure type
typedef struct control_struct{
		  int counter, busy;
		  pthread_mutex_t mutex;
		  pthread_cond_t cv;
} control_t;

control_t control = {0, 1, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER};

/* The cleanup handler associated with the thread. This routine is installed around the pthread_cond_wait call which is a cancellation point. */
void cleanup_handler(void *args)
{
		  control_t *s = (control_t *)args;
		  s->counter--;
		  printf("Cleanup handle: counter = %d. \n", s->counter);
		  pthread_mutex_unlock(&(s->mutex));
}

void *my_thread(void *args)
{
		  // Associate the cleanup handler
		  pthread_cleanup_push(cleanup_handler, (void *)&control); 
		  pthread_mutex_lock(&control.mutex);
		  control.counter++;

		  // Simulate a thread being blocked indefinitely
		  while(control.busy){
					 pthread_cond_wait(&control.cv, &control.mutex);
		  }

		  // Run the cleanup handler during normal termination as well, that is, when cancellation does not occur
		  pthread_cleanup_pop(1);
} 

int main(int argc, char **argv)
{
		  pthread_t thread_id[NUM_THREADS];
		  int i;
		  void *result;

		  /* Create the threads. The threads are created with their cancellation flag set to 
			* DEFERRED as the default setting. */
		  printf("Creating the threads. \n");
		  for(i = 0; i < NUM_THREADS; i++)
					 pthread_create(&thread_id[i], NULL, my_thread, NULL);

		  /* Simulate some processing. */
		  sleep(5); // Sleep for five seconds

		  printf("Cancelling the threads. \n");
		  for(i = 0; i < NUM_THREADS; i++){
					 pthread_cancel(thread_id[i]);
					 
					 pthread_join(thread_id[i], &result);
					 if(result == PTHREAD_CANCELED)
								printf("Thread was cancelled. \n");
					 else
								printf("Thread was not cancelled. \n");
		  }
		  pthread_exit(NULL);
}

