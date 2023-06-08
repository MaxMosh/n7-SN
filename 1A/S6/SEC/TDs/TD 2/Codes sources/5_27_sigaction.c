/* attente 27s ou 5 signaux reçus (ex 2.5.3) [PM, le 08/04/13]
 * 
 * Schéma de base, interface sigaction, sans distinguer la source des SIGALRM
 *
 */

#include <signal.h>
#include <stdio.h>
#include <unistd.h>

int nbrecus = 0;
int nbalarm = 0;

void affsig (int sig) {
    printf ("Reception d'un signal %d\n", sig);
    nbrecus++;
}

void actif (int sig) {
	printf ("Reception du signal %d (SIGALRM)\n", sig);
    printf ("Toujours actif...\n");
    alarm (3);
    nbalarm++;
}

int main (void) {
    struct sigaction mon_action;
    int i, ret;
	
    mon_action.sa_handler = affsig;
    sigemptyset(&mon_action.sa_mask);
    mon_action.sa_flags = 0;

    for (i = 1; i <= NSIG; i++)
		ret= sigaction(i, &mon_action, NULL);
    
    mon_action.sa_handler = actif;
    ret= sigaction(SIGALRM, &mon_action, NULL);
    
    alarm (3);
    while ((nbrecus != 5) && (nbalarm != 9)) {
        pause ();
    }
	printf ("reçus %d, alarm %d\n", nbrecus,nbalarm);
    return 0;
}
