/* 
   Author: Naga Kandasamy, 1/21/2009

   Program to illustrate basic thread management operations. 

   Compile as follows: 
   gcc -o simple_thread simple_thread.c -std=c99 -lpthread 
*/

#define _REENTRANT // Make sure the library functions are MT (muti-thread) safe
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

/* Function prototypes for the thread routines */
void *func_a(void *);
void *func_b(void *);

pthread_t thr_a, thr_b; /* Variables that store thread IDs */
int a = 10;

int main()
{
		  pthread_t main_thread;
		  main_thread = pthread_self(); // Returns the thread ID for the calling thread 
		  printf("Main thread = %u \n", (int)main_thread);

		  // Create a new thread and ask it to execute func_a that takes no arguments
		  if((pthread_create(&thr_a, NULL, func_a, NULL)) != 0){
					 printf("Cannot create thread \n");
					 exit(0);
		  }

		  printf("Main thread is thread %u: Creating thread %u \n", pthread_self(), thr_a);
		  pthread_join(thr_a, NULL); // Wait for thread to finish

		  printf("Value of a is %d\n", a);
  
		  printf("Main thread exiting \n");
		  pthread_exit((void *)main_thread);
}

/* Function A */
void *func_a(void *arg){

		  thr_a = pthread_self(); // Obtain thread ID
		  int args_for_thread = 5;

		  // Create a new thread and ask it to execute func_b
		  if((pthread_create(&thr_b, NULL, func_b, (void *)args_for_thread)) != 0){
					 printf("Cannot create thr_a\n");
					 exit(0);
		  }
		  
		  // Simulate some processing 
		  printf("Thread %u is processing. \n", thr_a);
		  for(int i = 0; i < 5; i++) {
					 a = a + 1;
					 sleep(1);
		  }

		  pthread_join(thr_b, NULL); // Wait for thread B to finish 

		  printf("Thread %u is exiting \n", thr_a);
		  pthread_exit((void *)thr_a);
}

/* Function B */
void *func_b(void *arg){
		 int args_for_me = (int) arg;
		 pthread_t thr_b = pthread_self();

		 // Simulate some processing
		 printf("Thread %u is using args %d. \n", thr_b, args_for_me);
  
		 printf("Thread %u is processing. \n", thr_b);
		 for(int i = 0; i < 5; i++) {
					sleep(2);
		 }

		 printf("Thread %u is exiting \n", thr_b);
		 pthread_exit((void *)thr_b);
}







