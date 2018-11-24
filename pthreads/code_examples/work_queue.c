/* This code shows an example of how to use a thread pool or work queue. 
 * Author: Naga Kandasamy
 * Date: 04/12/2011
 * Compile as follows: gcc -o work_queue work_queue.c -std=c99 -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <math.h>

#define NUM_ITEMS 6
#define NUM_THREADS 2
#define MAX_NUM_THREADS 50

// Define data structures for the work queue which is a singly linked list comprising of work items
typedef struct item_s{
   int id; // Work ID
	int work; // Work item
	struct item_s *next; // Pointer to next item in the list
} item_t;

typedef struct queue_s{
	item_t *head;
	item_t *tail;
	int num_workers;
	int num_items;
} queue_t;

// Create the work queue
queue_t work_queue;
sem_t work_available; // Counting semaphore to indicate that there is some work available
sem_t work_queue_lock ; // Binary semaphore to protect the work queue

// This function removes a work item from the work queue and returns the pointer to that item
item_t *remove_item(queue_t *this_queue)
{
		  item_t *this_item;

		  // Remove item from the head of the queue
		  if(this_queue->head != NULL){
					 this_item = this_queue->head;
					 this_queue->head = (this_queue->head)->next;
					 this_queue->num_items--; 
					 return this_item;
		  }
		  printf("Something is wrong. Queue is empty!. \n");
		  return NULL;
}

// This function adds a work item to the work queue
void add_item(queue_t *this_queue, item_t *this_item)
{
		  if(this_queue->head == NULL){
					 this_queue->head = this_item;
					 this_queue->tail = this_item;
		  }
		  else{
					 // Add item to end of the queue
					 (this_queue->tail)->next = this_item;
					 this_queue->tail = (this_queue->tail)->next;
		  }
		  this_queue->num_items++;
}

// Helper function to print the contents of the queue 
void print_queue(queue_t *this_queue)
{
		  item_t *current = this_queue->head;
		  while(current != NULL){
					 printf("Work ID: %d. \n", current->id); 
					 printf("Processing time: %d. \n", current->work);
					 printf("\n");
					 current = current->next;
		  }
}

// Cleanup function for the workers
void cleanup_handler(void *args)
{
		  int thread_id = (int)args;
		  printf("Worker %d shutting down. \n", thread_id);
}

// This function is executed by the worker threads
void *worker(void *args)
{
		  int thread_id = (int) args;
		  item_t *this_item;
		  int state;

		  pthread_cleanup_push(cleanup_handler, (void *)thread_id);
		  while(1){
					 printf("Worker %d is waiting for work. \n", thread_id);
					 sem_wait(&work_available);

					 // Obtain item from the queue
					 sem_wait(&work_queue_lock);
					 this_item = remove_item(&work_queue);
					 sem_post(&work_queue_lock);
					
					 pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &state); // Disable cancellation in this section of the code
					 printf("Worker %d is busy for %d seconds with work ID %d. \n", thread_id, this_item->work, this_item->id);
					 sleep(this_item->work); // Simulate some processing
					 free((void *)this_item);
					 pthread_setcancelstate(state, &state); // Re-enable cancellation
					 pthread_testcancel(); // Test for cancellation
		  }

		  pthread_cleanup_pop(0);
}

// This function is executed by the contractor thread. It creates the work crew and distributes work to it
void *contractor(void *args)
{
	// Initialize the work queue
	work_queue.head = work_queue.tail = NULL;
	work_queue.num_items = 0;
	work_queue.num_workers = NUM_THREADS;

	// Initialize the semaphores 
	sem_init(&work_available, 0, 0);
	sem_init(&work_queue_lock, 0, 1);

	// Create the work crew and work items for the crew
	int i, j;
	pthread_t thread_id[MAX_NUM_THREADS];

	for(j = 0; j < work_queue.num_workers; j++)
			  pthread_create(&thread_id[j], NULL, worker, (void *)j);
	
	for(i = 0; i < NUM_ITEMS; i++){
			  int sleep_time = (int)ceil((float)rand()/(float)RAND_MAX * 1); // Sleep for some random time between 0 and 5 seconds
			  sleep(sleep_time);

			  printf("Contractor: Creating work for the work queue. \n");
			  int processing_time = (int)ceil((float)rand()/(float)RAND_MAX * 10); // Processing time per work item 
			  item_t *new_item = (item_t *)malloc(sizeof(item_t));
			  new_item->id = i;
			  new_item->work = processing_time;
			  new_item->next = NULL;

			  sem_wait(&work_queue_lock);
			  add_item(&work_queue, new_item);
			  sem_post(&work_queue_lock);
			  sem_post(&work_available); // Signal the workers that some work is available 
	}
	
	// print_queue(&work_queue);

	// Check to see if the workers are done
	while(1){
			  sem_wait(&work_queue_lock);
			  if(work_queue.num_items > 0){
						 sem_post(&work_queue_lock);
						 sched_yield(); // Give up the CPU
			  } else
						 break;
	}
	// A worker woke us up indicating that the queue has been depleted. Cancel the workers
	void *result;	
	for(i = 0; i < work_queue.num_workers; i++){
			  pthread_cancel(thread_id[i]);
			  pthread_join(thread_id[i], result);
			  if(result == PTHREAD_CANCELED)
						 printf("Thread %d cancelled. \n", i);
	}

	pthread_exit(NULL);
}

int main(int argc, char **argv)
{
	pthread_t thread_id;

	// Create the contractor thread
	pthread_create(&thread_id, NULL, contractor, NULL);  

	pthread_join(thread_id, NULL);
	pthread_exit(NULL);
}
