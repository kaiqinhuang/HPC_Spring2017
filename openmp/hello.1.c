/* An OpneMP multithreaded program that prints hello world. 
   Compile with:
	cc -fopenmp file-name.c
*/

#include <omp.h>
#include <stdio.h>

int main()
{
	printf("Hello from main.\n");
	// Parallel region with default number of threads
	#pragma omp parallel
	{
		int myID = 0;
		printf("hello(%d)", myID);
		printf("world(%d)\n", myID);
	}
}


/* Output:
 
 Hello from main.
 hello(0)world(0)
 hello(0)world(0)
 hello(0)world(0)
 hello(0)world(0)
 hello(0)world(0)
 hello(0)world(0)
 hello(0)world(0)
 hello(0)world(0)
 
 First line that it prints out is "Hello from main."
    And then prints "hello(0)world(0)" in each thread.
    The default number of threads is eight here.
*/
