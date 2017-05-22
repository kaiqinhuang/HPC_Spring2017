#include <stdio.h>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

using namespace std;

int main(int argc, char ** argv)
{
    string userInput;
    cout<< "Enter a Linux Command: " << endl;
    cin >> userInput;
    
    cout << "The user entered " << userInput << endl;
    
    while(userInput != "exit")
    {
        pid_t p_id = fork();  //create child process
    
        if(p_id < 0)  // -1
        {
            cout << "fatal error" << endl;
        }
    
        else if (p_id == 0)  //if p_id == 0, the child process can run
        {
            execlp(userInput.c_str(), userInput.c_str(), NULL);  //pass in the command user has entered
        }
        
        else
        {
            waitpid(p_id, 0, 0);  //wait to run the process with a positive p_id
        }
        
        //prompt user again
        cout<< "Enter a Linux Command: " << endl;
        cin >> userInput;
        
        cout << "The user entered " << userInput << endl;
    }
    
    exit(0);
}
