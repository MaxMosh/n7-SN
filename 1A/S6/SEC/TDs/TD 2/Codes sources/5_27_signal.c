/* attente 27s ou 5 signaux reçus (ex 2.5.3) [PM, le 08/04/13]
 * 
 * Schéma de base, interface signal, sans distinguer la source des SIGALRM
 *
 */

#include <signal.h>
#include <stdio.h>
#include <unistd.h>

static int nbrecus = 0;
static int nbalarm = 0;

void affsig (int sig)
{   /* signal (SIGALRM, actif) est nécessaire en System V, inutile en POSIX */
	printf ("Reception d'un signal %d\n", sig);
    nbrecus++;
}

void actif (int sig)
{   /* signal (SIGALRM, actif) est nécessaire en System V, inutile en POSIX */
	printf ("Reception du signal %d (SIGALRM)\n", sig);
	printf ("Toujours actif...\n");
    alarm (3);
    nbalarm++;
}

int main () {
    int i;
    for (i = 1; i <= NSIG; i++)
    	{signal (i, affsig);}
    signal (SIGALRM, actif);
    alarm (3);
    while ((nbrecus != 5) && (nbalarm != 9)) {
        pause ();
    }
	printf ("reçus %d, alarm %d\n", nbrecus,nbalarm);
    return 0;
}