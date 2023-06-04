#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#define TAILLE_PORTE_MONNAIE 5

// Definition du type monnaie
// TODO
struct Monnaie{
  int quantite;
  char devise;
};

typedef struct Monnaie Monnaie;
/**
 * \brief Initialiser une monnaie
 * \param[]
 * \pre
 * // TODO
 */
void initialiser(Monnaie* M_init, int q, char d){
    assert(q > 0);
    (*M_init).quantite = q;
    (*M_init).devise = d;
}


/**
 * \brief Ajouter une monnaie m2 à une monnaie m1
 * \param[]
 * // TODO
 */
bool ajouter(Monnaie* m1, Monnaie m2){
    if ((*m1).devise == m2.devise){
        (*m1).quantite = (*m1).quantite + m2.quantite;
        printf("L'opération s'est passée correctement\n");
        return true;
    }
    else{
        printf("Erreur, l'opération n'a pas pu se dérouler\n");
        return false;
    }
}


/**
 * \brief Tester Initialiser
 * \param[]
 * // TODO
 */
void tester_initialiser(){
    Monnaie M_test;
    int q = 7;
    char d;
    d = '$';
    initialiser(&M_test,q,d);
}

/**
 * \brief Tester Ajouter
 * \param[]
 * // TODO
 */
void tester_ajouter(){
    Monnaie m1;
    Monnaie m2;
    int q;
    char d;
    q = 7;
    d = '$';
    initialiser(&m1,q,d);
    q = 2;
    initialiser(&m2,q,d);
    ajouter(&m1,m2);
    assert(m1.quantite == 9);
}



int main(void){
    // Un tableau de 5 monnaies
    Monnaie porte_monnaie[TAILLE_PORTE_MONNAIE];

    //Initialiser les monnaies
    int q;
    char d;
    for (int i = 0; i < TAILLE_PORTE_MONNAIE; i++){
        printf("Entrer la quantité :\n");
        scanf(" %d", &q);
        porte_monnaie[i].quantite = q;
        printf("Entrer la devise :\n");
        scanf(" %c", &d);
        porte_monnaie[i].devise = d;
    };

    // Afficher la somme des toutes les monnaies qui sont dans une devise entrée par l'utilisateur.
    Monnaie S;
    S.quantite = 0;
    S.devise = porte_monnaie[0].devise;

    for (int i = 0; i < TAILLE_PORTE_MONNAIE; i++){
        ajouter(&S,porte_monnaie[i]);
    }
    printf("La somme des espèces du porte-monnaie est : %d\n", S.quantite);

    return EXIT_SUCCESS;
}
