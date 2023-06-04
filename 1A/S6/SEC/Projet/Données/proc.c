#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <wait.h>
#include <string.h>
#include <signal.h>
#include "proc.h"


Processus creer_processus(int id, pid_t pid, Etat etat, char* commande){
    Processus processus;
    processus.id = id;
    processus.pid = pid;
    processus.etat = etat;
    processus.commande = commande;
    return processus;
}

int init_liste_processus(Liste_Processus* liste_processus){
    liste_processus->nb_processus = 0;
    return 0;
}

void ajouter_processus(Processus processus, Liste_Processus* liste_processus){
    if (liste_processus->nb_processus < MAX_LISTE_PROC){
        liste_processus->liste[liste_processus->nb_processus] = processus;
        liste_processus->nb_processus++;
    }
    else {
        printf("Nombre maximum de processus atteint\n");
    }
}

int pid_proc_to_id_proc(pid_t pid_proc, Liste_Processus liste_processus){
    for (int i = 0; i < liste_processus.nb_processus; i++){
        if (liste_processus.liste[i].pid == pid_proc){
            return liste_processus.liste[i].id;
        }
    }
    return -1;
}

int id_proc_to_pid_proc(int id_proc, Liste_Processus liste_processus){
    for (int i = 0; i < liste_processus.nb_processus; i++){
        if (liste_processus.liste[i].id == id_proc){
            return liste_processus.liste[i].pid;
        }
    }
    return -1;
}

int supprimer_processus(int id_proc, Liste_Processus* liste_processus){
    int i = 0;
    while ((liste_processus->liste[i].id != id_proc) && (i < liste_processus->nb_processus)){
        i++;
    }
    if (i == liste_processus->nb_processus){
        printf("Processus non trouvé\n");
        return -1;
    }
    else if (i == liste_processus->nb_processus-1){
        liste_processus->nb_processus--;
        return 0;
    }
    else {
        for (int j = i+1; j < liste_processus->nb_processus; j++){
            liste_processus->liste[j-1] = liste_processus->liste[j];
        }
        liste_processus->nb_processus--;
        return 0;
    }
}

int supprimer_processus_pid(pid_t pid_proc, Liste_Processus* liste_processus){
    int i = 0;
    while ((liste_processus->liste[i].pid != pid_proc) && (i < liste_processus->nb_processus)){
        i++;
    }
    if (i == liste_processus->nb_processus){
        printf("Processus non trouvé\n");
        return -1;
    }
    else if (i == liste_processus->nb_processus-1){
        liste_processus->nb_processus--;
        return 0;
    }
    else {
        for (int j = i+1; j < liste_processus->nb_processus; j++){
            liste_processus->liste[j-1] = liste_processus->liste[j];
        }
        liste_processus->nb_processus--;
        return 0;
    }
}

void afficher_processus(Liste_Processus liste_processus){
    printf("xxxxxxxxxxxXXXXXX LES PROCESSUS LANCES PAR LE MINISHELL XXXXXXXxxxxxxxxxxx\n");
    for (int i = 0; i < liste_processus.nb_processus; i++){
        printf("Processus %d de pid %d dans l'état %d exécuté avec la commande \"%s\"\n", liste_processus.liste[i].id, liste_processus.liste[i].pid, liste_processus.liste[i].etat, (char*)liste_processus.liste[i].commande);
    }
    printf("xxxxxxxxxxxXXXXXX---------------------------------------XXXXXXXxxxxxxxxxxx\n");
}

int trouver_processus(int id_proc, Liste_Processus liste_processus, Processus* processus_trouve){

    for (int i = 0; i < liste_processus.nb_processus; i++){
        if (liste_processus.liste[i].id == id_proc){
            *processus_trouve = liste_processus.liste[i];
            return 0;
        }
    }
    return 1;
}

int trouver_indice_liste(int id_proc, Liste_Processus liste_processus){
    for (int i = 0; i < liste_processus.nb_processus; i++){
        if (liste_processus.liste[i].id == id_proc){
            return i;
        }
    }
    return -1;
}


int stopper_processus(Liste_Processus* liste_processus, Processus processus){
    Processus processus_a_stopper;
    int i = 0;
    while ((liste_processus->liste[i].id != processus.id) && (i < liste_processus->nb_processus)){
        i++;
    }
    if (i == liste_processus->nb_processus){
        printf("Processus non trouvé\n");
        return 1;
    }
    else {
        processus_a_stopper = liste_processus->liste[i];
        kill(processus_a_stopper.pid, SIGSTOP);
        supprimer_processus(processus_a_stopper.id, liste_processus);
        return 0;
    }
}