(* Pour tester les fonctions, on les copie dans le terminal *)

(* Exercice 1 *)

(* Question 1 *)

(* fonction deuxieme *)
(* renvoie le deuxième élément d'une liste *)
(* l : la liste *)
(* précondition : taille l >= 2 *)

let deuxieme l = match l with
    | []    -> failwith "La liste est vide."
    | _::[] -> failwith "La liste ne contient qu'un élément."
    |_::element2::l -> element2;;


(* Autre version *)

let deuxieme2 l = 
    match l with
        |[]     -> failwith "La liste est vide."
        |t::q   -> match q with
                        |[] -> "La liste ne contient qu'un élément."
                        |element2::_ -> element2;;


(* Question 2 *)

(* fonction n_a_zero *)
(* renvoie la liste des n premier entier naturel dans l'ordre décroissant *)
(* parametre n : entier à partir duquel on part *)
(* précondition : n >= 0 *)

let rec n_a_zero n =
    if n = 0 then n::[]
    else n:: n_a_zero (n - 1);;


(* Tests *)
n_a_zero 0;;
n_a_zero 1;;
n_a_zero 16;;


let zero_a_n n =
    let rec zero_a_n_term p liste_p_plus_1_a_n =
        if p < 0 then liste_p_plus_1_a_n
        else zero_a_n_term (p - 1) (p::liste_p_plus_1_a_n)
    in zero_a_n_term n [];;


(* Tests *)
zero_a_n 0;;
zero_a_n 1;;
zero_a_n 16;;


(* Question 3 *)

(* fonction positions *)
(* renvoie la liste des positions d'un élément e dans une liste *)
(* parametre l : liste dans laquelle on cherche l'élément e *)
(* parametre e : élément à chercher dans la liste l *)
(* précondition : liste l non vide *)


(* Ma version *)
let positions l e =
    let rec positions_rec liste element k =
        match liste with
            |[]     ->  []
            |t::q   ->  if t = element then k::positions_rec q element (k + 1)
                        else positions_rec q element (k + 1)
    in
    positions_rec l e 0;;


(* Tests *)
positions [1;2;2;2;1;2] 1;;



(* Exercice 2 *)

(* Question 1 *)

(* fonction map_perso *)
(* applique une fonction f à tous les éléments d'une liste (renvoie la liste) *)
(* paramètre f : fonction à appliquer *)
(* l : la liste *)
(* précondition : la fonction f est bien définie vis-à-vis du type des éléments de la liste l *)


(* Ma version : je n'ai pas utilisé fold_right et fold_left alors que j'aurais dû *)
let rec map_perso f l = match l with
    | []    -> []
    | t::q -> (f t)::(map_perso f q);;


(* Tests *)
let x_carre x = x*x;;
map_perso x_carre [1;2;3];;


(* Version prof *)
let rec map_prof f l =
    List.fold_right (fun t map_queue -> (f t)::map_queue) l [];;

(* Tests *)
let x_carre x = x*x;;
map_prof x_carre [1;2;3];;


(* Question 2 *)

let flatten l =
    List.fold_right (fun l1 l2 -> l1@l2) l []


(* Question 3 *)

let fsts = List.map fst


(* Question 4 *)

let split liste_couple = 
    List.fold_right (fun (a,b) (la,lb) -> (a::la,b::lb)) liste_couple ([],[])