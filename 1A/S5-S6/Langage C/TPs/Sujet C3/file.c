/**
 *  \author Xavier Cr�gut <nom@n7.fr>
 *  \file file.c
 *
 *  Objectif :
 *	Implantation des op�rations de la file
*/

// Bin�me : Maxime Moshfeghi & Samy Afker

#include <malloc.h>
#include <assert.h>

#include "file.h"


void initialiser(File *f)
{
    f->tete = NULL;         // On initialise la t�te
    f->queue = NULL;        // Et la queue � NULL
    assert(est_vide(*f));
}


void detruire(File *f)
{
    free(f->tete);      // On lib�re la m�moire allou�e dynamiquement � f->tete
    f->tete = NULL;     // On n'oublie pas de remettre le pointeur � NULL
}


char tete(File f)
{
    assert(! est_vide(f));
    return f.tete->valeur;  // Apr�s s'�tre assur� que la file n'est pas vide, on renvoie la valeur de la cellule en t�te de file
}


bool est_vide(File f)
{
    return (f.tete == NULL && f.queue == NULL);
}

/**
 * Obtenir une nouvelle cellule allou�e dynamiquement
 * initialis�e avec la valeur et la cellule suivante pr�cis� en param�tre.
 */
static Cellule * cellule(char valeur, Cellule *suivante)
{
    Cellule *cell = malloc(sizeof(Cellule));    // On alloue dynamiquement la m�moire pour une cellule
    cell->valeur = valeur;                      // On lui attribue ensuite la valeur voulue
    cell->suivante = suivante;                  // ainsi que la cellule suivante souhait�e
    return cell;
}


void inserer(File *f, char v)
{
    assert(f != NULL);
    Cellule * new_queue = cellule(v, NULL);
    // On disjoint les cas o� la file est nulle et celui o� la file ne l'est pas
    if (f->tete == NULL){
        f->queue = new_queue;
        f->tete = new_queue;    // Dans ce cas, la file a m�me t�te et queue
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
    *v = f->tete->valeur;           // On stocke la valeur en t�te de file dans *v
    f->tete = f->tete->suivante;    // Et on red�finit la t�te � la cellule suivante
}


int longueur(File f)
{
    if (f.tete == NULL){    // Condition d'arr�t de la fonction
        return 0;
    } else {                // Appel r�cursif de la fonction longueur
        File file_temp;
        file_temp.tete = f.tete->suivante;
        return 1 + longueur(file_temp);
    }
}
