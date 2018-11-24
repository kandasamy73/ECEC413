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
 * Compile as follows: gcc -o trap_v4 trap_v4.c -fopenmp -std=c99
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double f(double); 

int 
main(int argc, char **argv)
{
    if(argc != 2){
        printf("Usage: trap_v4 <num threads> \n");
        exit(0);
    }

    int thread_count = atoi(argv[1]); 
    double approximate_area = 0.0; 
    double a;                           /* Start of the interval. */
    double b;                           /* End of the interval. */
    int n;                              /* Number of subintervals. */

    printf("Enter a, b, and n. \n");
    printf("a and b represent the interval [a, b] and n represents the number of subintervals. \n");
    scanf("%lf %lf %d", &a, &b, &n);
	
    double h = (b - a)/n;               /* Number of subintervals. */
    approximate_area = (f(a) + f(b))/2.0;

#pragma omp parallel for num_threads(thread_count) reduction(+: approximate_area)
    for(int i = 1; i <= (n - 1); i++){
        approximate_area += f(a + i*h);
    }
    
    /* Threads synchronize here. */
    approximate_area = h*approximate_area; 

    printf("With %d trapeziods, the estimate for the integral between [%f, %f] is %f \n", n, a, b, approximate_area);
	
    return 0;
}

double 
f(double x)                      /* Function of interest. */
{
    return((x*x*x + x*x + 1)/(x - 1));
}
