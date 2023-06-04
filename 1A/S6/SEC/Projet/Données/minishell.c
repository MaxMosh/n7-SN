#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <stdbool.h>
// #include <wait.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <string.h>
#include "readcmd.h"
#include "proc.h"

#define SUCCES 0
#define ECHEC 1

// Définition des variables globales
static int id_minishell = 0;
Liste_Processus liste_processus;

// Procédure permettant d'afficher le prompt
void afficher_prompt(){
	printf(">>>");
}


// Procédure permettant de lire la commande
void lire_commande(struct cmdline** une_ligne){
	*une_ligne = readcmd();
    while ((*une_ligne)->seq == NULL){
		afficher_prompt();
		*une_ligne = readcmd();
	}
    if ((*une_ligne)->err != NULL) {
    	printf("Erreur de lecture\n");
        exit(1);
    }
}


// Fonction renvoyant un booléen vraie si une commande est interne, faux sinon
bool est_commande_interne(struct cmdline* une_ligne){
	bool est_interne = false;
	if (strcmp(une_ligne->seq[0][0],"exit") == 0){
		est_interne = true;
	}
	else if (strcmp(une_ligne->seq[0][0],"cd") == 0){
		est_interne = true;
	}
	else if (strcmp(une_ligne->seq[0][0],"lj") == 0){
		est_interne = true;
	}
	else{
	}
	return est_interne;
}


// Procédure permettant d'exécuter une commande interne
void exec_commande_interne(struct cmdline* une_ligne, Liste_Processus* la_liste){
	char* la_commande = malloc(sizeof(char)*100);
	strcpy(la_commande, une_ligne->seq[0][0]); // Pour enregistrer la commande dans la liste des processus

	// On effectue toutes les comparaisons nécessaires pour savoir s'il s'agit d'une commande interne
	if (strcmp(une_ligne->seq[0][0],"exit") == 0){
        printf("Au revoir utilisateur !\n");
        exit(2);
    }
	else if (strcmp(une_ligne->seq[0][0],"cd") == 0){
		Processus processus_commande_interne = creer_processus(id_minishell, getpid(), ACTIF, la_commande);
		ajouter_processus(processus_commande_interne, la_liste);
		id_minishell++;
		if (une_ligne->seq[0][1] != NULL){
			// Récupération du chemin passé en argument
			chdir(une_ligne->seq[0][1]);
			int suppression_ok = supprimer_processus(processus_commande_interne.id, la_liste);

			// Pour le débuggage : on ajoute le processus avec un état terminé (vérification de la commande lj et des structures)
			/*
			Processus processus_commande_interne_termine = creer_processus(id_minishell-1, getpid(), TERMINE, la_commande); // Il faudra ajouter le chemin demandé
			ajouter_processus(processus_commande_interne_termine, la_liste);
			*/
		}
		else{
			// Récupération du chemin par défaut (chemin vers lequel la commande "cd" sans argument nous amène)
			char *chemin_home = getenv("HOME"); 
			chdir(chemin_home);
			int suppression_ok = supprimer_processus(processus_commande_interne.id, la_liste);

			// Pour le débuggage : on ajoute le processus avec un état terminé (vérification de la commande lj et des structures)
			/*
			Processus processus_commande_interne_termine = creer_processus(id_minishell-1, getpid(), TERMINE, la_commande); // Il faudra ajouter le chemin demandé
			ajouter_processus(processus_commande_interne_termine, la_liste);
			*/
		}
    }
	else if (strcmp(une_ligne->seq[0][0],"lj") == 0){

		Processus processus_commande_interne = creer_processus(id_minishell, getpid(), ACTIF, la_commande);
		ajouter_processus(processus_commande_interne, la_liste);
		id_minishell++;
		afficher_processus(*la_liste);
		int suppression_ok = supprimer_processus(processus_commande_interne.id, la_liste);

		// Pour le débuggage : on ajoute le processus avec un état terminé (vérification de la commande lj et des structures)
		/*
		Processus processus_commande_interne_termine = creer_processus(id_minishell-1, getpid(), TERMINE, la_commande); // Il faudra ajouter le chemin demandé
		ajouter_processus(processus_commande_interne_termine, la_liste);
		*/
    }
}


// Procédure permettant d'exécuter une commande externe
void exec_commande_externe(struct cmdline* une_ligne, Liste_Processus* la_liste){
	int idFils, codeTerm;

	// On copie la commande pour pouvoir l'afficher lors de l'affichage des processus
	char* la_commande = malloc(sizeof(char)*100);
	strcpy(la_commande, une_ligne->seq[0][0]);

	// On clône le processus
	int pidFils = fork();

	// Cas d'erreur
	if (pidFils == -1) {
		printf ("Erreur fork\n");
		exit(1);
	}

	// Exécution de la ligne de commande
	if (pidFils == 0){
		// Le cas des commandes internes a été traitée précédemment, on n'essaie donc pas de l'exécuter avec "execvp"
		if (est_commande_interne(une_ligne)){
			exit(8);
		}else{
			if (une_ligne->in != NULL){
				// On redirige l'entrée standard
				int fd_i = open(une_ligne->in, O_RDONLY, NULL);
				close(0);
				dup2(fd_i, 0);
			}
			if (une_ligne->out != NULL){
				// On redirige la sortie standard
				int fd_o = open(une_ligne->out, O_WRONLY | O_CREAT | O_TRUNC, 0644);
				close(1);
				dup2(fd_o, 1);
			}

			/* // On définit le handler par défaut sur sig_ign (pour ignorer tous les signaux dans un premier temps)
			struct sigaction sig_ign;
			sig_ign.sa_handler = SIG_IGN;
			sigaction(SIGCHLD, &sig_ign, NULL);
			*/

			// Exécution de la commande
			execvp(une_ligne->seq[0][0], une_ligne->seq[0]);
			exit(9);
		}
	} else {
		// On ajoute le processus exécuté par le fils à la liste des processus
		Processus processus_commande_externe = creer_processus(id_minishell, pidFils, ACTIF, (char*)la_commande);
		ajouter_processus(processus_commande_externe, la_liste);
		id_minishell++;

		// Le père "attends" le retour du fils dans le cas d'un processus qui n'est pas demandé en arrière plan, déclenchant ainsi sa mort	
		if (une_ligne->backgrounded == NULL) {
			la_liste->pid_processus_courant = la_liste->liste[la_liste->nb_processus-1].pid;

			while (la_liste->pid_processus_courant >= 0){
				pause();
			}
		}
	}
}


// Handler du signal SIGCHLD (j'ai laissé le corps des instructions permettant de tester si les suppressions de processus se faisaient bien)
// Il s'agit des commentaires encadrées par /*[ ]*/
void sigchld_handler(int signum) {
	pid_t pidchld;
	int status;
	pidchld = waitpid(-1, &status, WNOHANG|WUNTRACED|WCONTINUED);
	if (WIFEXITED(status)) {
		// Pour le débuggage : on garde le processus retiré de la liste des processus ...
		/*[
		Processus processus_commande_externe_termine;
		int id_proc_term = pid_proc_to_id_proc(pidchld, liste_processus);
		int processus_a_termine_present = trouver_processus(id_proc_term, liste_processus, &processus_commande_externe_termine);
		]*/
		int id_proc_term = pid_proc_to_id_proc(pidchld, liste_processus); // REMETTRE SI ON ENLEVE LE DEBUGGAGE
		if (supprimer_processus(id_proc_term, &liste_processus) == -1){
			printf("Erreur de suppression de processus dans le handler de SIGCHLD");
		}
		
		// AJOUTER UN IF POUR CHANGER L'ID DU PROCESSUS EN AVANT PLAN SI C'EST CELUI-CI QUI EST TERMINE
		//if (pid_proc_to_id_proc(pidchld, liste_processus) == id_proc_term) {
		if (pidchld == liste_processus.pid_processus_courant) {
			liste_processus.pid_processus_courant = -1;
			kill(getpid(), SIGCONT);
		}

		// Pour le débuggage : ... puis on le rajoute avec un état TERMINE dans la liste des processus
		/*[
		if (processus_a_termine_present == 0) {
			processus_commande_externe_termine.etat = TERMINE;
			int ajouter_ok_fini = ajouter_processus(processus_commande_externe_termine, &liste_processus);
		} else {
			printf("Processus externe terminé : non trouvé\n");
		}
		]*/
	} else if (WIFSIGNALED(status)) {
		// Pour le débuggage : on garde le processus retiré de la liste des processus ...
		/*[
		Processus processus_commande_externe_termine;
		int id_proc_term = pid_proc_to_id_proc(pidchld, liste_processus);
		int processus_a_termine_present = trouver_processus(id_proc_term, liste_processus, &processus_commande_externe_termine);
		]*/

		//printf("Le fils %d est mort avec le signal %d\n", pid, WTERMSIG(status));
		int id_proc_term = pid_proc_to_id_proc(pidchld, liste_processus); // REMETTRE SI ON ENLEVE LE DEBUGGAGE
		if (supprimer_processus(id_proc_term, &liste_processus) == -1){
			printf("Erreur de suppression de processus dans le handler de SIGCHLD");
		}

		// On met une valeur négative de pid car cela permet de signifier qu'il n'y a plus aucun processus en avant plan
		if (pidchld == liste_processus.pid_processus_courant) {
			liste_processus.id_processus_courant = -1;
		}

		// Pour le débuggage : ... puis on le rajoute avec un état TERMINE dans la liste des processus
		/*[
		if (processus_a_termine_present == 0) {
			processus_commande_externe_termine.etat = TERMINE;
			ajouter_processus(processus_commande_externe_termine, &liste_processus);
		} else {
			printf("Processus externe terminé : non trouvé\n");
		}
		]*/

	} else if (WIFSTOPPED(status)) {
		// CHANGEMENT D'ETAT : PAUSE/CONTINUE

		Processus processus_commande_externe_stoppe;
		int id_proc_a_stoppe = pid_proc_to_id_proc(pidchld, liste_processus);
		int emplacement_processus = trouver_indice_liste(id_proc_a_stoppe, liste_processus);

		liste_processus.liste[emplacement_processus].etat = SUSPENDU;
	}
}








int main() {

	// On assigne le bon handler à SIGCHLD
	struct sigaction action_chld;
	action_chld.sa_handler = sigchld_handler;

	sigaction(SIGCHLD, &action_chld, NULL);	

	// Définition du handler de SIGINT (non fonctionnel)
	/*
	struct sigaction action_int;
	action_int.sa_handler = sigint_handler;
	sigaction(SIGINT, &action_int, NULL);
	*/

	// On initialise notre liste de processus
	int init_list_ok = init_liste_processus(&liste_processus);


	// Pour vérifier que lj fonctionne : on rajoute un processus pour le minishell lui même
	/*Processus processus_temp;
	processus_temp = creer_processus(id_minishell, getpid(), ACTIF, "minishell");
	int ajout_ok = ajouter_processus(processus_temp, &liste_processus);
	id_minishell++;
	*/

	while (true) { // Il serait préférable de mettre un do { ... } while (true), mais la structure initiale de mon programme ne le permettait pas
    	struct cmdline* line;
	
		afficher_prompt();
		
		// Lecture de la commande
    	lire_commande(&line);

		// On exécute la commande selon si elle est interne ou externe
		if (est_commande_interne(line)){
			exec_commande_interne(line, &liste_processus);
		}
		else{
			exec_commande_externe(line, &liste_processus);
		}
	}
	printf("\nAu revoir utilisateur !\n");
    return 0;
}
