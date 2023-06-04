Avertissement
-------------
Le script de vérification `verif_signaux.sh` doit être considéré comme un simple outil mis
 à votre disposition, pour vous fournir une indication quant à la viabilité de vos réponses,
  et non  comme une application de validation automatique de votre travail. Simplement, 
  si vous passez la vérification, vous pouvez avoir bon espoir quant à l'évaluation 
  effective. Et inversement.

En particulier :

  - il est inutile de modifier le script pour qu'il donne une réponse `OK` : la validation
  se fera sur nos propres outils.
  - le script n'est pas protégé contre les erreurs résultant de (mauvaises) actions liées
  à l'exécution de vos programmes. Par exemple si votre programme détruit des fichiers
  de manière intempestive, le script de vérification peut devenir invalide.
  Il est donc prudent de prévoir une sauvegarde de l'original, si vous voulez être prémunis
   contre ce genre d'accidents.
  - en revanche, le script de vérification fonctionne bien avec des réponses correctes.
    Par conséquent, si une erreur est signalée sur une ligne du script, vous pouvez être
    quasi-certains que cela ne découle pas d'une erreur dans le script de test.

Conventions de nommage
----------------------

Pour que le script de vérification `verif_signaux.sh` puisse être appliqué :

  - le fichier source du programme à vérifier doit être **exactement** nommé `etu.c` et 
    rangé dans le répertoire `etu`, situé au même niveau que `verif_signaux.sh`
  - le répertoire `etu` contient par ailleurs un fichier texte `réponses` destiné à recueillir
    vos réponses aux questions posées.
  - le répertoire contenant `verif_signaux.sh` ne devra pas être modifié, en dehors de l'ajout du
    fichier source `etu.c`.
  

Appel et résultats du script de vérification
--------------------------------------------

Le script `verif_signaux.sh` doit être lancé depuis un terminal, le répertoire courant 
étant le répertoire contenant `verif_signaux.sh`.

* Lorsqu'il est lancé sans option, `verif_signaux.sh` effectue un diagnostic sur le programme
`etu.c`.  
Si la vérification échoue le script affiche `KO`, sinon il affiche `OK`. 
Notez que la mention `OK` est une condition nécessaire pour que la réponse soit juste,
mais que ce n'est pas une condition suffisante.    
En particulier, vous êtes laissés juges de leur pertinence, mais a priori la vérification
ne devrait afficher aucun warning suite à la compilation.   
Lorsque le script `verif_signaux.sh` se termine, il affiche un message `OK` ou `KO`.   
 Il est possible que la réponse fournie provoque le blocage du script. Dans ce cas, il faut
  tuer le processus exécutant le script.
* Lorsqu'il est lancé avec l'option `-s` (pour soumettre), le script prépare l'archive qui
pourra être déposée sur Moodle. L'archive créée par l'appel de `verif_signaux.sh -s` se 
trouve au même niveau que `verif_signaux.sh`

