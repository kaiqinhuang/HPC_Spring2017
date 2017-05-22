/* An OpneMP  multithreaded program where each thread prints hello world. 
   Compile with an appropriate switch:
     cc -fopenmp file-name.c
*/

#include <omp.h>
#include <stdio.h>

int main()
{
	printf("Hello from main.\n");
	omp_set_num_threads(20); // Requests a non-default number of threads.
	// Parallel region with the requested number of threads:	
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
 hello(2)hello(6)hello(12)hello(4)world(2)
 world(12)
 hello(0)world(0)
 hello(8)world(8)
 world(6)
 hello(3)world(3)
 hello(5)world(5)
 hello(10)world(10)
 hello(11)world(11)
 hello(9)world(9)
 hello(15)world(15)
 hello(1)world(1)
 world(4)
 hello(17)world(17)
 hello(7)world(7)
 hello(18)world(18)
 hello(16)world(16)
 hello(14)world(14)
 hello(19)world(19)
 hello(13)world(13)

It prints out "hello(myID)world(myID)" in each thread and the number of threads is non-default.
There are 20 threads in this case.

*/
