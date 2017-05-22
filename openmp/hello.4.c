/* An OpneMP  multithreaded program where each thread prints hello world. 
   Compile with an appropriate switch:
     cc -fopenmp file-name.c
*/

#include <omp.h>
#include <stdio.h>

int main()
{
	printf("Hello from main.\n");
	// Parallel region with a non-default number of threads:	
	#pragma omp parallel num_threads(20)
	{
		// Runtime library function to return a thread ID:
		int myID = omp_get_thread_num();
		printf("hello(%d)", myID);
		printf("world(%d)\n", myID);
	}
}


/* Output:

 Hello from main.
 hello(1)hello(17)world(17)
 hello(6)world(6)
 hello(0)hello(10)world(10)
 hello(9)world(9)
 world(1)
 hello(11)hello(4)hello(19)world(19)
 world(4)
 hello(8)world(8)
 hello(2)world(2)
 hello(3)world(3)
 hello(16)world(16)
 hello(15)world(15)
 hello(18)world(18)
 world(11)
 hello(13)world(13)
 hello(7)world(7)
 hello(5)world(5)
 world(0)
 hello(14)world(14)
 hello(12)world(12)

It does exatly same thing as hello.3.c but defines the number of threads (20) in 
 "#pragma omp parallel num_threads(20)" instead of "omp_set_num_threads(20)".
 
*/
