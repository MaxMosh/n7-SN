#include <stdio.h>    /* entrées/sorties */
#include <unistd.h>   /* primitives de base : fork, ...*/
#include <stdlib.h>   /* exit */
#include <signal.h>


// Traitant des signaux SIGUSR1 et SIGUSR2
void handler_sigusr(int num_SIGUS) {
	printf("\nReception %d \n", num_SIGUS) ;
	return;
}


int main(int argc, char *argv[]) {

	// On associe le traitant aux deux signaux utilisateurs
	struct sigaction action;
	action.sa_handler = handler_sigusr;
	sigemptyset(&action.sa_mask);
	action.sa_flags = 0;
	sigaction(SIGUSR1, &action, NULL);
	sigaction(SIGUSR2, &action, NULL);

	// On déclare un nouveau set
	sigset_t ensemble_signal;
	// On vide le set de signal
	sigemptyset(&ensemble_signal);
	// On ajoute les deux signaux voulus au set
	sigaddset(&ensemble_signal, SIGINT);
	sigaddset(&ensemble_signal, SIGUSR1);
	// On applique le masque aux deux signaux du set
	sigprocmask(SIG_BLOCK, &ensemble_signal, NULL);
	
	// On attend 10 secondes
	sleep(10);
	
	// Envoi de 2 SIGUSR1
	kill(getpid(),SIGUSR1);
	kill(getpid(),SIGUSR1);
	// On attend 5 secondes
	sleep(5);
	// Envoi de 2 SIGUSR2
	kill(getpid(),SIGUSR2);
	kill(getpid(),SIGUSR2);
	
	// On enlève SIGINT de notre ensemble de signaux
	sigdelset(&ensemble_signal, SIGINT);
	// On applique ensuite la procédure suivante pour retirer le signal SIGUSR1 du masque
	sigprocmask(SIG_UNBLOCK, &ensemble_signal, NULL);
	
	// On attend 10 secondes
	sleep(10);
	// On vide le set de signal
	sigemptyset(&ensemble_signal);
	// Puis on y ajoute SIGINT
	sigaddset(&ensemble_signal, SIGINT);
	// Puis on démasque enfin SIGINT
	sigprocmask(SIG_UNBLOCK, &ensemble_signal, NULL);
	
	printf("Salut\n");
	return EXIT_SUCCESS;
}
