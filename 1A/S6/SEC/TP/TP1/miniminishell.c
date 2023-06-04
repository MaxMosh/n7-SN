#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <wait.h>

#define SUCCES 0
#define ECHEC 1

int main() {
    char buf[30];
    int ret;

    int idFils, codeTerm;

    printf(">>>");
    ret = scanf("%30s", buf);
    while (ret != EOF) {
        // Evite un buffer overflow
        buf[29] = '\0';


        // On clône le processus
        int pidFils = fork();

        // Cas d'erreur
        if (pidFils == -1) {
            printf ("Erreur fork\n");
            exit(1);
        }
        // Exécution de la ligne de commande
        if (pidFils == 0){
            execlp(buf, buf, NULL);
            exit(9);
        } else {
            // Le père "attends" le retour du fils, déclenchant ainsi sa mort
            idFils = wait(&codeTerm);
            if (idFils == -1) {
                perror ("wait");
                exit(2);
            }
            if (WEXITSTATUS(codeTerm) == 9) {
                printf("ECHEC\n");
            } else {
                printf("SUCCES\n");
            }
        }
        printf(">>>");
        ret = scanf("%30s", buf);
    }
    printf("\nSalut Max\n");
    return ret;
}
