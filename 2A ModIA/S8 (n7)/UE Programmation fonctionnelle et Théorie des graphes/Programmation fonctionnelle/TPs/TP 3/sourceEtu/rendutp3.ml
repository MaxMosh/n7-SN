(* Maxime Moshfeghi : partie à rendre du TP3 *)

(****** Algorithmes combinatoires et listes ********)


(*** Code binaires de Gray ***)

(*CONTRAT
Fonction qui génère un code de Gray
Paramètre n : la taille du code
Resultat : le code sous forme de int list list
*)


let rec gray_code n = match n with
  | 0 -> [[]]
  | _ -> (List.map (fun liste -> 0::liste) (gray_code (n-1)))@(List.map (fun liste -> 1::liste) (List.rev(gray_code (n-1))))
  
(* failwith "TO DO" *)


(* TESTS *)
let%test _ = gray_code 0 = [[]]
let%test _ = gray_code 1 = [[0]; [1]]
let%test _ = gray_code 2 = [[0; 0]; [0; 1]; [1; 1]; [1; 0]]
let%test _ = gray_code 3 = [[0; 0; 0]; [0; 0; 1]; [0; 1; 1]; [0; 1; 0]; [1; 1; 0]; [1; 1; 1]; [1; 0; 1]; [1; 0; 0]]
let%test _ = gray_code 4 = [[0; 0; 0; 0]; [0; 0; 0; 1]; [0; 0; 1; 1]; [0; 0; 1; 0]; [0; 1; 1; 0];
                            [0; 1; 1; 1]; [0; 1; 0; 1]; [0; 1; 0; 0]; [1; 1; 0; 0]; [1; 1; 0; 1];
                            [1; 1; 1; 1]; [1; 1; 1; 0]; [1; 0; 1; 0]; [1; 0; 1; 1]; [1; 0; 0; 1];
                            [1; 0; 0; 0]]


(*** Combinaisons d'une liste ***)

(* CONTRAT 
TO DO : FAIT
Fonction qui donne l'ensemble des combinaisons possibles pour une liste
Paramètre k : taille des combinaisons
Paramètre l : liste pour laquelle on va cherche les combinaisons à k éléments
Resultat : liste des combinaisons à k éléments possibles pour la liste l (sous forme de 'a list list)
*)
let rec combinaison k l = match k,l with
  | _,[]  -> []
  | 0,_   -> [[]]
  (* | (List.length l),_ -> [l] *)
  | _,t::q -> if (List.length l = k) then [l]
              else (List.map (fun x -> t::x) (combinaison (k - 1) q))@(combinaison k q)

(* failwith "TO DO" *)


(* TESTS *)
(* TO DO : FAIT *)
let%test _= combinaison 5 [1;2;3;4] = []
let%test _= combinaison 4 [1;2;3;4] = [[1;2;3;4]]
let%test _= combinaison 3 [1;2;3;4] = [[1; 2; 3]; [1; 2; 4]; [1; 3; 4]; [2; 3; 4]]
let%test _= combinaison 2 [1;2;3;4] = [[1; 2]; [1; 3]; [1; 4]; [2; 3]; [2; 4]; [3; 4]]

let%test _= combinaison 0 [1;2;3;4] = [[]]

let%test _= combinaison 3 [] = []