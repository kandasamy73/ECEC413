/* A program that illustrates the use of OpenMp to compute the area under a curve f(x) using the trapezoidal rule. 
 * Given a function y = f(x), and a < b, we can estimate the area between the graph of f(x) (within the vertical lines x = a and 
 * x = b) and the x-axis by dividing the interval [a, b] into n subintervals and approximating the area over each subinterval by the 
 * area of a trapezoid. 
 * If each subinterval has the same length and if we define h = (b - a)/n, x_i = a + ih, i = 0, 1, ..., n, then the approximation 
 * becomes: h[f(x_0)/2 + f(x-1) + f(x_2) + ... + f(x_{n-1}) + f(x_n/2)
 *
 * This code assumes that f(x) = (x^3 + x^2  + 1)/(x - 1)
 *
 * Author: Naga Kandasamy
 * Date: 04/15/2011
 * Compile as follows: gcc -o trap_v1 trap_v1.c -fopenmp -std=c99
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Function prototypes
void trap (double, double, int, double *);

int
main (int argc, char **argv)
{
    if (argc != 2){
        printf ("Usage: trap <num threads> \n");
        exit (0);
  }
  
    int thread_count = atoi (argv[1]);	                        /* Number of threads to be created. */
    double approximate_area = 0.0;
    double a;			                                        /* Start of the interval. */
    double b;			                                        /* End of the interval. */
    int n;			                                            /* Number of subintervals. */

    printf ("Enter a, b, and n. \n");
    printf("a and b represent the interval [a, b] and n represents the number of subintervals. \n");
    scanf ("%lf %lf %d", &a, &b, &n);
  
    if ((n % thread_count) != 0){
        printf ("n must be evenly divisible by the thread_count. \n");
        exit (0);
    }

#pragma omp parallel num_threads(thread_count)                  /* OpenMP block */
    {
        trap (a, b, n, &approximate_area);
    }

    printf("With %d trapeziods, the estimate for the integral between [%f, %f] is %f \n", n, a, b, approximate_area);

    return 0;
}

double
f (double x)                                                    /* Function of interest. */
{
    return ((x * x * x + x * x + 1) / (x - 1));
}

void
trap (double a, double b, int n, double *approximate_area)      /* Function executed by each thread in the OpenMP block. */
{
    int my_id = omp_get_thread_num ();	                        /* Obtain thread ID. */
    int thread_count = omp_get_num_threads ();

    double h = (b - a) / (float) n;	                            /* Length of the subinterval. */

    /* We assume that the number of subintervals is evenly divisible by thread_count. */
    int chunk_size = n / thread_count;	
    double start_offset = a + h * chunk_size * my_id;
    double end_offset = start_offset + h * chunk_size;

  
    /* Approximate the area under the curve in the interval [start_offset, end_offset]. */
    double my_result = 0.0;
    double x;
    my_result = (f (start_offset) + f (end_offset)) / 2.0;
    for (int i = 1; i <= (chunk_size - 1); i++){
        x = start_offset + i * h;
        my_result += f (x);
    }
    my_result = my_result * h;

#pragma omp critical
    *approximate_area += my_result;
}
