/* An OpneMP  multithreaded program where each thread prints hello world. 
   Compile with an appropriate switch:
     cc -fopenmp file-name.c
*/

#include <omp.h>
#include <stdio.h>

int main()
{
	printf("Hello from main.\n");
	// Parallel region with default number of threads:	
	#pragma omp parallel
	{
		// Runtime library function to return a thread ID:
		int myID = omp_get_thread_num();
		printf("hello(%d)", myID);
		printf("world(%d)\n", myID);
	}
}


/* Output:

 Hello from main.
 hello(5)hello(6)world(6)
 world(5)
 hello(1)world(1)
 hello(2)world(2)
 hello(4)world(4)
 hello(7)world(7)
 hello(0)world(0)
 hello(3)world(3)

 It prints out "hello(myID)world(myID)" in each thread and myID is the thread number.

*/
