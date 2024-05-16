(* Pour tester les fonctions, on les copie dans le terminal *)

(* Exercice 1 *)

(* fonction cardpart *)
(* renvoie le nombre d'élément de l'ensemble des parties d'un ensemble *)
(* ens : l'ensemble (modélisé par une liste) *)
(* précondition : la liste ens a une structure d'ensemble (unicité des éléments) *)

let rec cardpart ens = match ens with
    | []    -> 1
    | _::q  -> 2*cardpart q;;
    

(* Exercice 2 *)

(* Question 1 *)

(* fonction ajout *)
(* une liste contenant les ensembles en entrée et les ensembles en entrée auquel on a ajouté un élément *)
(* e : élément à ajouter *)
(* listeens : liste d'ensembles *)
(* précondition : les ensembles présents dans la liste listeens ont une structure d'ensemble (unicité des éléments) et l'élément e n'est présent
   dans aucun des ensembles de listeens *)

(* DEBUT MAXIME *)

(* let ajout e listeens =
    let ajoutunens el ens =
        el::ens
    in
    match (List.map (ajoutunens e) listeens) with
    | t::[]::[] -> []::[]:: *)

(* CORRECTION PROF *)

let ajout listeens e =
    listeens@List.map(fun ens -> e::ens) listeens;;


(* Question 2 *)

(* fonction parties *)
(* une liste contenant l'ensemble des parties d'un ensemble *)
(* ens : ensemble dont on cherche l'ensemble des parties *)
(* précondition : la liste ens a une structure d'ensemble (unicité des éléments) *)

let rec parties ens = match ens with
    | []    -> [[]]
    | t::q  -> ajout (parties q) t;; (* D'APRES CORRECTION *)


(* Exercice 3 *)

(* VERSION À LA MAIN DE L'EXERCICE 4 *)


(* Exercice 4 *)

let rec insertion e liste = match liste with
    | []    -> [[e]]
    | t::q  -> (e::liste)::(List.map (fun uneliste -> t::uneliste) (insertion e q));;


let rec permutations liste = match liste with
    | [] -> [[]]
    | t::q -> List.flatten(List.map (insertion t) (permutations q));;