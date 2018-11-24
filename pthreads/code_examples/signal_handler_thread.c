/* Author: Naga Kandasamy, 1/22/2009
   
   This code illustrates how to create a signal handler thread.
   Compile as follows: gcc -o signal_handler_thread signal_handler_thread.c -lpthread 

*/

#define _REENTRANT
#include <stdio.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

#define TRUE 1
#define FALSE 0

void *signal_handler(void *); /* The signal handler prototype */


int main(int agrc, char **argv){

  sigset_t set; /* Define the set of signals */
  pthread_t thread_id;
  
  /* Block all signals in the main thread. Threads created after this will 
     inherit this, i.e, also block all signals */
  sigfillset(&set);
  sigprocmask(SIG_BLOCK, &set, NULL);

  /* Create a signal_handler thread to handle all non-directed signals */
  if((pthread_create(&thread_id, NULL, signal_handler, NULL)) != 0){
					 printf("Cannot create signal handler thread. \n");
					 exit(-1);
		  }

  printf("Main thread processing \n");
  while(TRUE){
    /* Simulate some processing in the main thread. */
  }

  return(0);
}

/* The signal handler thread. */
void *signal_handler(void *arg){
  sigset_t set;
  int sig;

  sigfillset(&set); /* Catch all signals */

  while(TRUE){
			 /* Wait for some signal to arrive and handle the signal appropriately. */
			 if(sigwait(&set, &sig) != 0){
						printf("Error waiting on a signal. \n");
						break;
			 };
			
			 switch(sig){
						case SIGINT: 
								  printf("Ctrl+C received ... quiting \n");
								  exit(0);

						case SIGUSR1: 
								  printf("Received SIGUSR1 signal \n");
								  break;

						case SIGUSR2:
								  printf("Received SIGUSR2 signal \n");
								  break;

						default:
								  printf("Received some signal \n");
			 } /* End switch */
  } /*End while */
  return ((void *)0);
}
