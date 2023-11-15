#include <stdio.h>    /* entrées/sorties */
#include <unistd.h>   /* primitives de base : fork, ...*/
#include <stdlib.h>   /* exit */
#include <signal.h>

#define MAX_PAUSES 10     /* nombre d'attentes maximum */

 


int main(int argc, char *argv[]) {


	// Traitant signal
	void handler_signal(int signal) {
		printf("\nPid du proocessus : %d \nSignal reçu : %d\n", getpid(), signal) ;
		return;
	}	
	
	
	for (int i=0 ; i < NSIG ; i ++){
		signal(i, handler_signal);
	}
	
	int nbPauses;
	nbPauses = 0;
	printf("Processus de pid %d\n", getpid());
	for (nbPauses = 0 ; nbPauses < MAX_PAUSES ; nbPauses++) {
		pause();		// Attente d'un signal
		printf("pid = %d - NbPauses = %d\n", getpid(), nbPauses);
    } ;
    return EXIT_SUCCESS;
}
