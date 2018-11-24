/* Example code illustrating the producer consumer synchronization problem. 
 * Compile as follows: 
 *
 * gcc -o producer_consumer producer_consumer.c -std=c99 -lpthread -lm
 *
 * Author: Naga Kandasamy
 * Date: 15 July 2011
 * Last undate:
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>


#define QUEUE_SIZE 5
#define NUM_ITEMS 10

/* Variables to store the final statistics. */
int num_items_produced = 0;
int num_items_consumed = 0;

/* Define the queue data structure type. */
typedef struct queue_t{
		  int buffer[QUEUE_SIZE]; // The buffer
		  int head, tail; // Indices within the queue for the producer and the consumer
		  int full, empty; // Predicates to indicate if the queue is full or empty
		  pthread_mutex_t lock; // Lock to protect the queue structure
		  pthread_cond_t not_full, not_empty; // Condition signalling variables 
} QUEUE;


/* Function prototypes for the functions to be executed by the threads. */
void *producer(void *args);
void *consumer(void *args);

/* Fuction prototypes for queue operations. */
QUEUE *create_queue(int);
void delete_queue(QUEUE *);
void add_to_queue(QUEUE *, int);
int remove_from_queue(QUEUE *);

/* Other helper functions. */
int UD(int, int);


int main(int argc, char **argv)
{
		  /* Create and initialize the queue data structure */ 
		  QUEUE *queue = create_queue(QUEUE_SIZE);
		  if(queue == NULL){
					 printf("Error creating the queue data structure. Exiting. \n");
					 exit(-1);
		  }

		  /* Create the producer and consumer threads. */
		  pthread_t producer_id, consumer_id;
		  pthread_create(&producer_id, NULL, producer, (void *)queue);
		  pthread_create(&consumer_id, NULL, consumer, (void *)queue);

		  /* Wait for the producer and consumer threads to finish and join the main thread. */
		  pthread_join(producer_id, NULL);
		  pthread_join(consumer_id, NULL);

		  delete_queue(queue);
		  pthread_exit(NULL);
}


/* The function executed by the producer thread. */
void *producer(void *args)
{
		  QUEUE *this_queue = (QUEUE *)args; // Typecast back to the queue data structure
		  for(int i = 0; i < NUM_ITEMS; i++){
					 int item = UD(1, 10); // We produce an item which simulates a processing time between 2 and 5 seconds
					 pthread_mutex_lock(&(this_queue->lock)); 
					 while(this_queue->full == 1){
								printf("Producer: THE QUEUE IS FULL. \n");
								pthread_cond_wait(&(this_queue->not_full), &(this_queue->lock)); // Producer blocks here. relinquishes the lock
					 }
					 /* The producer unblocks here. The mutex has already been acquired. */
					 printf("Producer: adding item %d to queue with a processing time of %d. \n", i, item);
					 add_to_queue(this_queue, item);
					 pthread_mutex_unlock(&(this_queue->lock));
					 pthread_cond_signal(&(this_queue->not_empty)); // Signal to the consumer in case it is sleeping on an empty queue
					 sleep(UD(1, 2)); // The producer sleeps for some random time between 2 and 5 seconds
		  }
		  pthread_exit(NULL);
}

/* The function executed by the consumer thread. */
void *consumer(void *args)
{
		  QUEUE *this_queue = (QUEUE *)args;
		  for(int i = 0; i < NUM_ITEMS; i++){
					 pthread_mutex_lock(&(this_queue->lock));
					 while(this_queue->empty == 1){
								printf("Consumer: THE QUEUE IS EMPTY. \n");
								pthread_cond_wait(&(this_queue->not_empty), &(this_queue->lock)); // The consumer blocks here, relinquishes the lock
					 }
					 /* The consumer unblocks here. The mutex has already been acquired. */
					 int item = remove_from_queue(this_queue);
					 pthread_mutex_unlock(&(this_queue->lock));

					 printf("Consumer: processing item %d from the queue with a processing time of %d. \n", i, item);
					 pthread_cond_signal(&(this_queue->not_full)); // Signal the producer in case it is sleeping on a full queue
					 sleep(item); // Simulate some processing
		  }
		  pthread_exit(NULL);
}


/* This function adds an item to the queue. */
void add_to_queue(QUEUE *this_queue, int item)
{
		  this_queue->buffer[this_queue->tail] = item; 
		  this_queue->tail++;
		  if(this_queue->tail == QUEUE_SIZE) 
					 this_queue->tail = 0; // Wrap around the circular buffer
		  if(this_queue->tail == this_queue->head)
					 this_queue->full = 1; // Queue is full

		  this_queue->empty = 0;
		  return;
}

/* This function removes an item from the queue. */
int remove_from_queue(QUEUE *this_queue)
{
		  int item = this_queue->buffer[this_queue->head];
		  this_queue->head++;
		  if(this_queue->head == QUEUE_SIZE)
					 this_queue->head = 0; // Warp around the circular buffer
		  if(this_queue->head == this_queue->tail)
					 this_queue->empty = 1; // Queue is empty

		  this_queue->full = 0;
		  return item;
}

/* Helper function to create and initialize the queue. */
QUEUE *create_queue(int size)
{
		  QUEUE *queue = (QUEUE *)malloc(sizeof(QUEUE));
		  if(queue == NULL) return NULL;

		  /* Initialize the members of the structure. */
		  queue->head = queue->tail = 0;
		  queue->full = 0;
		  queue->empty = 1;
		  pthread_mutex_init(&(queue->lock), NULL);
		  pthread_cond_init(&(queue->not_full), NULL);
		  pthread_cond_init(&(queue->not_empty), NULL);

		  return queue;
}

/* Helper function to delete the queue */
void delete_queue(QUEUE *this_queue)
{
		  pthread_mutex_destroy(&(this_queue->lock));
		  pthread_cond_destroy(&(this_queue->not_full));
		  pthread_cond_destroy(&(this_queue->not_empty));

		  free((void *)this_queue);
		  return;
}

/* Returns a random number between min and max */
int UD(int min, int max){
  return((int)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX))));
}

