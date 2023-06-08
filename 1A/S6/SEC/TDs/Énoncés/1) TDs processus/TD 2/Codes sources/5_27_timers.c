 /* attente 27s ou 5 signaux reçus (ex 2.5.3) [PM, le 08/04/13]
 * 
 *	Utilisation des timers pour distinguer la source des SIGALRM.
 *		on compte le temps restant depuis le dernier armement de SIGALRM : 
 *		s'il est assez proche de 3s, c'est qu'il vient d'être réarmé, 
 *		"donc" qu'il provient du timer. (bien sur, il y a une petite fenêtre 
 *		(normalement < 1 ms) où l'on peut se tromper...)
 *
 */


#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>

static int nbrecus = 0;
static int nbalarm = 0;
struct itimerval duree;


void affsig (int sig) {
    printf ("Reception d'un signal %d\n", sig);
    nbrecus++;
}

void actif (int sig) { 
    getitimer(ITIMER_REAL, &duree);
	printf ("Reception du signal %d (SIGALRM)\n", sig);
	printf ("Toujours actif, restait %ld s, %d µs...\n",duree.it_value.tv_sec,
			duree.it_value.tv_usec);
	if ((duree.it_value.tv_sec == 2)&&(duree.it_value.tv_usec >= 990000)) {
		nbalarm++;
	}else {nbrecus++;}
}

int main () {
    struct sigaction mon_action;
    int i, ret;
	
    mon_action.sa_handler = affsig;
    sigemptyset(&mon_action.sa_mask);
    mon_action.sa_flags = 0;

    for (i = 1; i <= NSIG; i++)
		ret= sigaction(i, &mon_action, NULL);
    
    mon_action.sa_handler = actif;
    ret= sigaction(SIGALRM, &mon_action, NULL);
	duree.it_value.tv_sec = 3;
	duree.it_value.tv_usec = 0;
	duree.it_interval.tv_sec = 3;
	duree.it_interval.tv_usec = 0;
	setitimer(ITIMER_REAL, &duree, NULL);  /* armement du timer temps-rÈel */
    while ((nbrecus != 5) && (nbalarm != 9)) {
        pause ();
    }
	printf ("reçus %d, alarm %d\n", nbrecus,nbalarm);
    return 0;
}