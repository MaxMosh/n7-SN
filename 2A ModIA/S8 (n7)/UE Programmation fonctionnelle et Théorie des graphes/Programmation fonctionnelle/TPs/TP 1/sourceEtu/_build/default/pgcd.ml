(* Exercice à rendre **)
(* auteur : Maxime Moshfeghi *)
(* contrat *)
(* pgcd : int -> int -> int *)
(* calcule le PGCD de l'entier x et de l'entier y *)
(* paramètre x : int, premier entier *)
(* paramètre y : int, deuxième entier *)
(* résultat : int, le PGCD de l'entier x et de l'entier y *)
(* précondition : x et y sont des entiers *)

let pgcd a b = 
  let abs r =
    if r >= 0 then r else -r
  in
  let rec euclide x y =
    (* ALTERNATIVE 1 *)
    match x, y with
    | (0, 0) -> failwith "Il n'existe pas de PGCD à 0 et 0"
    | (_, 0) -> x
    | (0, _) -> y
    | (_, _) -> euclide y (x mod y)
    (* ALTERNATIVE 2 (moins bien) : 
    if x = 0 && y = 0
      then failwith "Il n'existe pas de PGCD à 0 et 0"
    else if y = 0
      then x
    else euclide y (x mod y) *)
  in euclide (abs a) (abs b)

(* tests unitaires *)
let%test _ = pgcd 3 4 = 1
let%test _ = pgcd 0 3 = 3
let%test _ = pgcd 9 6 = 3
let%test _ = pgcd 6 6 = 6