/* 
   This code illustrates how to use semaphores to solve the sleeping barber problem. Compile as 
   follows:
   gcc -o sleeping_barber sleeping_barber.c -lpthread -lm

   Usage:
   sleeping_barber <customers> <chairs>
   Example: sleeping_barber 10 5; means: there are 10 customers and 5 waiting chairs in the 
   barber shop

*/

#define _REENTRANT
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <math.h>

#define TRUE 1
#define FALSE 0
#define MIN_TIME 4
#define MAX_TIME 8

#define MAX_NUM_CUSTOMERS 50

void *customer(void *num); // Prototype for customer thread
void *barber(void *); // Prototype of barber thread 
int UD(int, int); // Random number generator


/* Definition of semaphores */

sem_t waitingRoom; // Signal that the waiting room can accommodate  customers
sem_t barberSeat; // Signal to ensure exclusive access to the barber seat
sem_t doneWithCustomer; // Signals the customer that the barber is done with him/her
sem_t barberBed; // Signal to wake up the barber

int doneWithAllCustomers = FALSE; // Flag indicating the barber can go home

int main(int argc, char **argv){

  if(argc < 3){
    printf("Usage: sleeping_barber <customers> <chairs> \n");
    exit(0);
  }

  int numCustomers = atoi(argv[1]); // Number of customers
  int numWaitingChairs = atoi(argv[2]); // Number of waiting chairs in the barber shop

  srand((long)time(NULL)); /* Initialize randomizer */
 
  if(numCustomers > MAX_NUM_CUSTOMERS){
    printf("Number of customers exceeds the maximum capacity of the barber \n");
    printf("Resetting the number of customers to %d \n", (int)MAX_NUM_CUSTOMERS);
    numCustomers = MAX_NUM_CUSTOMERS;
  }
  
  /* Initialize the semaphores */
  sem_init(&waitingRoom, 0, numWaitingChairs);
  sem_init(&barberSeat, 0, 1);
  sem_init(&doneWithCustomer, 0, 0);
  sem_init(&barberBed, 0, 0);

  pthread_t btid; // ID for the barber thread
  pthread_t tid[MAX_NUM_CUSTOMERS]; // IDs for customer threads

  /* Create barber thread */
  pthread_create(&btid, 0, barber, 0); 

  /* Create customer threads and give each an ID */
  int customerID[MAX_NUM_CUSTOMERS]; // Customer IDs
  int i;
  for(i = 0; i < numCustomers; i++){
    customerID[i] = i;
    pthread_create(&tid[i], 0, customer, &customerID[i]);
  }
  
  for(i = 0; i < numCustomers; i++)
    pthread_join(tid[i], 0);

  doneWithAllCustomers = TRUE;
  sem_post(&barberBed); // Wake up barber
  pthread_join(btid, 0);
}

void *barber(void *junk){

  while(!doneWithAllCustomers){ // Customers remain to be serviced
    printf("Barber: Sleeping \n");
    sem_wait(&barberBed);

    if(!doneWithAllCustomers){
      printf("Barber: Cutting hair \n");
      int waitTime = UD(MIN_TIME, MAX_TIME); // Simulate cutting hair
      sleep(waitTime);
      sem_post(&doneWithCustomer); // Indicate that chair is free
    }
    else{
      printf("Barber: Going home \n");
    }
  }
}

void *customer(void *customerNumber){
  int number = *(int *)customerNumber;
  printf("Customer %d: Leaving for the barber shop \n", number);
  int waitTime = UD(MIN_TIME, MAX_TIME); // Simulate going to the barber shop
  sleep(waitTime);
  printf("Customer %d: Arrived at the barber shop \n", number);

  sem_wait(&waitingRoom); // Wait to get into the barber shop
  printf("Customer %d: Entering waiting room \n", number);

  sem_wait(&barberSeat); // Wait for the barber to become free
  sem_post(&waitingRoom); // Let people waiting outside the shop know

  sem_post(&barberBed); // Wake up barber
  sem_wait(&doneWithCustomer); // Wait until hair is cut
  sem_post(&barberSeat); // Give up seat

  printf("Customer %d: Going home \n", number);
}


/* Returns a random number between min and max */
int UD(int min, int max){
  return((int)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX))));
}




