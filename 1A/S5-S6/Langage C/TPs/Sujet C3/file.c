/**
 *  \author Xavier Crégut <nom@n7.fr>
 *  \file file.c
 *
 *  Objectif :
 *	Implantation des opérations de la file
*/

// BinÙme : Maxime Moshfeghi & Samy Afker

#include <malloc.h>
#include <assert.h>

#include "file.h"


void initialiser(File *f)
{
    f->tete = NULL;         // On initialise la tÍte
    f->queue = NULL;        // Et la queue ‡ NULL
    assert(est_vide(*f));
}


void detruire(File *f)
{
    free(f->tete);      // On libËre la mÈmoire allouÈe dynamiquement ‡ f->tete
    f->tete = NULL;     // On n'oublie pas de remettre le pointeur ‡ NULL
}


char tete(File f)
{
    assert(! est_vide(f));
    return f.tete->valeur;  // AprËs s'Ítre assurÈ que la file n'est pas vide, on renvoie la valeur de la cellule en tÍte de file
}


bool est_vide(File f)
{
    return (f.tete == NULL && f.queue == NULL);
}

/**
 * Obtenir une nouvelle cellule allouée dynamiquement
 * initialisée avec la valeur et la cellule suivante précisé en paramètre.
 */
static Cellule * cellule(char valeur, Cellule *suivante)
{
    Cellule *cell = malloc(sizeof(Cellule));    // On alloue dynamiquement la mÈmoire pour une cellule
    cell->valeur = valeur;                      // On lui attribue ensuite la valeur voulue
    cell->suivante = suivante;                  // ainsi que la cellule suivante souhaitÈe
    return cell;
}


void inserer(File *f, char v)
{
    assert(f != NULL);
    Cellule * new_queue = cellule(v, NULL);
    // On disjoint les cas o˘ la file est nulle et celui o˘ la file ne l'est pas
    if (f->tete == NULL){
        f->queue = new_queue;
        f->tete = new_queue;    // Dans ce cas, la file a mÍme tÍte et queue
    }
    else{
        f->queue->suivante = new_queue;
        f->queue = f->queue->suivante;
    }
}

void extraire(File *f, char *v)
{
    assert(f != NULL);
    assert(! est_vide(*f));
    *v = f->tete->valeur;           // On stocke la valeur en tÍte de file dans *v
    f->tete = f->tete->suivante;    // Et on redÈfinit la tÍte ‡ la cellule suivante
}


int longueur(File f)
{
    if (f.tete == NULL){    // Condition d'arrÍt de la fonction
        return 0;
    } else {                // Appel rÈcursif de la fonction longueur
        File file_temp;
        file_temp.tete = f.tete->suivante;
        return 1 + longueur(file_temp);
    }
}
