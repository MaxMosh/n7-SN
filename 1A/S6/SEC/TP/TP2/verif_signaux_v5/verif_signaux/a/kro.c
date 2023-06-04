#include <signal.h>
#include <stdio.h>
#include <unistd.h>

int main () {
    printf ("%d\n",SIGUSR1);
    printf ("%d\n",SIGUSR2);
    return 0;
}
