/* 
   Author: Naga Kandasamy, 10/3/2011

   Program to illustrate basic thread management operations. 

   Compile as follows: 
   gcc -o alarm_thread alarm_thread.c -lpthread 
*/

#define _REENTRANT // Make sure the library functions are MT (muti-thread) safe
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>


#define MAX_LENGTH 256

typedef struct alarm_tag{
		  int seconds;
		  char message[MAX_LENGTH];

} alarm_t;

void *alarm_thread(void *arg);

int main(int argc, char **argv)
{
		  int status;
		  char message[MAX_LENGTH];
		  alarm_t *alarm;
		  pthread_t thread_id;

		  while(1){
					 printf("Alarm> ");
					 if(fgets(message, MAX_LENGTH, stdin) == NULL){
								exit(0);
					 }
					 if(strlen(message) <= 1) continue;

					 /* Create the alarm structure */
					 alarm = (alarm_t *)malloc(sizeof(alarm_t));
					 if(alarm == NULL){
								printf("Error allocating alarm structure. \n");
								exit(-1);
					 }

					 /* Parse the input message into tow fields: the number of seconds and the message. */
					 if(sscanf(message, "%d %64[^\n]", &alarm->seconds, alarm->message) < 2){
								printf("Bad input. \n");
								free((void *)alarm);
					 }
					 else{
								status = pthread_create(&thread_id, NULL, alarm_thread, (void *)alarm);
								if(status != 0){
										  printf("Error creating alarm thread. Exiting. \n");
										  break;
								}
					 }
		  } /* End while */
}

/* The alarm thread that sleeps for the number of seconds specified in the alarm field, prints the message and exits. */
void *alarm_thread(void *arg)
{
		  alarm_t *alarm = (alarm_t *)arg; // Typecast the input parameter to the appropriate structure 
		  int status;

		  /* Detach ourselves from the calling thread. That is, inform the pthread library that the calling thread does not 
			* need to know when this thread terminates or the termination status. */
		  status = pthread_detach(pthread_self());
		  if(status != 0){
					 printf("alarm_thread: detach error. \n");
					 exit(-1);
		  }
		  sleep(alarm->seconds);
		  printf("(%d} %s \n", alarm->seconds, alarm->message);
		  free((void *)alarm);
		  return;
}
 
