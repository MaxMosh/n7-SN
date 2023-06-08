/* attente 27s ou 5 signaux reçus (ex 2.5.3) [PM, le 08/04/13]
 
 On répond complètement à la question, en utilisant la structure siginfo_t
	la structure sigaction comprend un champ sa_flags, 
	qui permet de positionner un indicateur SA_SIGINFO, 
	qui lorsqu'il est positionné, suppose (et impose) que le handler
	référencé par cette structure sigaction ait la signature suivante :
		void handler(int sig, siginfo_t *info, ucontext_t *uap);
	le deuxième paramètre du handler est un pointeur sur une structure 
	siginfo_t  qui donne toutes les informations sur les circonstances 
	de l'envoi du signal (dont le pid de l'émetteur)
  
 siginfo_t {
	int      si_signo;    // Signal number
	int      si_errno;    /* An errno value 
	int      si_code;     /* Signal code 
	int      si_trapno;   /* Trap number that caused hardware-generated signal
	pid_t    si_pid;      /* Sending process ID 
						** renseigné pour SIGCHLD, 
						** et les signaux temps-réel : quelle chance !
	uid_t    si_uid;      /* Real user ID of sending process
	int      si_status;   /* Exit value or signal
	clock_t  si_utime;    /* User time consumed 
	clock_t  si_stime;    /* System time consumed
	sigval_t si_value;    /* Signal value
	int      si_int;      /* POSIX.1b signal
	void    *si_ptr;      /* POSIX.1b signal
	int      si_overrun;  /* Timer overrun count; POSIX.1b timers
	int      si_timerid;  /* Timer ID; POSIX.1b timers
	void    *si_addr;     /* Memory location which caused fault
	long     si_band;     /* Band event 
	int      si_fd;       /* File descriptor
	short    si_addr_lsb; /* Least significant bit of address
 }
*/


#include <signal.h>
#include <stdio.h>
#include <unistd.h>

int nbrecus = 0;
int nbalarm = 0;

void affsig (int sig)
{
    printf ("Reception d'un signal %d\n", sig);
    nbrecus++;
}

// void actif (int sig, struct __siginfo *info, void *uap) {
void actif (int sig, siginfo_t *info, void *uap) {
/*	l'un ou l'autre : siginfo_t est un alias pour struct __siginfo, 
	pas toujours défini selon les installations */
	printf ("%d : Reception du signal %d (SIGALRM), émetteur : %d\n", 
			getpid(), sig, info->si_pid);
	if (info->si_pid ==  0 ) {
	/* les SIGALRM programmés sont  envoyés par le scheduler (processus 0) */
		    alarm (3);
			nbalarm++;
	} else {nbrecus++;}
}

int main (void) {
    struct sigaction mon_action;
    struct sigaction mon_action0;
    int i, ret;
	
    mon_action0.sa_handler = affsig;
    sigemptyset(&mon_action0.sa_mask);
    mon_action0.sa_flags = 0;
    for (i = 1; i <= NSIG; i++)
		ret= sigaction(i, &mon_action0, NULL);
    
    mon_action.sa_sigaction = actif;
    sigemptyset(&mon_action.sa_mask);
    mon_action.sa_flags = SA_SIGINFO;
    ret= sigaction(SIGALRM, &mon_action, NULL);
	
    alarm (3);
    while ((nbrecus != 5) && (nbalarm != 9)) {
        pause ();
    }
	printf ("reçus %d, alarm %d\n", nbrecus,nbalarm);
    return 0;
}
