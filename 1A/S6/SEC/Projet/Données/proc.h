#ifndef PROC_MANAG_LISTE_H
#define PROC_MANAG_LISTE_H


#define MAX_LISTE_PROC 100

// Définition des structures

enum Etat {
    ACTIF,
    SUSPENDU,
    TERMINE // Pour le débuggage
};

typedef enum Etat Etat;

struct Processus {
    int id;
    pid_t pid;
    Etat etat;
    char* commande;
};

typedef struct Processus Processus;

struct Liste_Processus {
    Processus liste[MAX_LISTE_PROC];
    int nb_processus;
    int id_processus_courant;
    pid_t pid_processus_courant;
};

typedef struct Liste_Processus Liste_Processus;



// Définition des fonctions et procédures

// Initialiser la liste de processus
int init_liste_processus(Liste_Processus* liste_processus);

// Créer un processus à partir de ses attributs
Processus creer_processus(int id, pid_t pid, Etat etat, char* commande);

// Ajouter un processus à la liste de processus
void ajouter_processus(Processus processus, Liste_Processus* liste_processus);

// Obtenir l'id minishell d'un processus à partir de son pid (ATTENTION : ON SUPPOSE ALORS QU'IL N'EXISTE QU'UN SEUL PROCESSUS AVEC CE PID)
int pid_proc_to_id_proc(pid_t pid_proc, Liste_Processus liste_processus);

// Obtenir le pid d'un processus à partir de son id minishell
int id_proc_to_pid_proc(int id_proc, Liste_Processus liste_processus);

// Supprimer un processus de la liste de processus
int supprimer_processus(int id_proc, Liste_Processus* liste_processus);

// Afficher la liste de processus
void afficher_processus(Liste_Processus liste_processus);

// Trouver un processus à partir de son id minishell (stocké dans processus_trouve)
int trouver_processus(int id_proc, Liste_Processus liste_processus, Processus* processus_trouve);

// Trouver l'indice d'un processus à partir de son id minishell
int trouver_indice_liste(int id_proc, Liste_Processus liste_processus);

// Stopper un processus (NON IMPLANTE)
int stopper_processus(Liste_Processus* liste_processus, Processus processus);


#endif