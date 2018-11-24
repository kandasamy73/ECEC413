/* Example code that shows how to use semaphores to signal between threads. 
 * Author: Naga Kandasamy
 * Date: 04/05/2011
 * Compile as follows: gcc -o signalling_using_semaphores signalling_using_semaphores.c -std=c99 -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <semaphore.h>
#include <pthread.h>

typedef struct my_struct_s{
		  sem_t semaphore; // Signals a change to the value
		  pthread_mutex_t mutex; // Protects access to the value
		  int value; // The value itself
} MY_STRUCT;

MY_STRUCT data; 

// Function prototypes
void *my_thread(void *);


int main(int argc, char **argv)
{
		  pthread_t thread_id;
		 
		  /* Initialize the semaphore and value within data */
		  data.value = 0;
		  sem_init(&data.semaphore, 0, 0); // Semaphore is not shared among processes and is initialized to 0
		  pthread_mutex_init(&data.mutex, NULL); // Initialize the mutex

		  /* Create a thread */
		  pthread_create(&thread_id, NULL, my_thread, NULL);
		 		  
		  pthread_mutex_lock(&data.mutex);
		  // Test the predicate, that is the data and wait for a condition to become true if neccessary 
		  if(data.value == 0){
					 printf("Value of data = %d. Waiting for someone to change it and signal me. \n");
					 pthread_mutex_unlock(&data.mutex);
					 sem_wait(&data.semaphore); // Probe the semaphore, P(). If the value is zero, then we block, else we decrement the semaphore by one and proceed
		  }

		  // Someone changed the value of data to one
		  pthread_mutex_lock(&data.mutex);
		  if(data.value != 0){
					 printf("Change in variable state was signalled. \n");
					 printf("Value of data = %d. \n", data.value);
		  }
		  pthread_mutex_unlock(&data.mutex);

		  pthread_join(thread_id, NULL); // Wait for the thread to join us
		  pthread_exit(NULL);
}

/* The function executed by the thread */
void *my_thread(void *args)
{
		  int status;
		  sleep(5); // Sleep for five seconds
		  
		  pthread_mutex_lock(&data.mutex);
		  data.value = 1; // Change the state of the variable
		  pthread_mutex_unlock(&data.mutex);
		  
		  sem_post(&data.semaphore); // Signal the change to the blocked thread

		  pthread_exit(NULL);
}
