(* Exercice 1 *)

type 'a arbre = Noeud of bool * ('a branche list)
and 'a branche = 'a * 'a arbre;;


(* Exercice 2 *)

(* Signature de la fonction appartient : 'a arbre -> 'a list -> bool *)

(* Fonction auxiliaire recherche *)
(* Signature de la fonction recherche : 'a -> 'a branche list -> 'a branche option *)
let rec recherche c lb = match lb with
  | []            -> None
  | (tc,ta)::qlb  ->  if c < tc then None
                      else (if c = tc then Some ta
                            else recherche c qlb);;

let rec appartient lc (Noeud(b,lb)) = match lc with
  | []      -> b
  | c::qlc  -> match (recherche c lb) with
                  | None    -> false
                  | Some a  -> appartient qlc a;;


(* Exercice 3 *)

(* Signature de la fonction ajout : 'a list -> 'a arbre -> 'a arbre *)

(* Fonction auxiliaire maj *)
(* Signature fonction maj : 'a -> 'b -> ('a * 'b) list -> ('a * 'b) list *)

let rec maj c new_b lb = match lb with
  | []            ->  [(c,new_b)]
  | (tc,ta)::qlb  ->  if c < tc then (c,new_b)::lb
                      else (if c = tc then (c,new_b)::qlb
                            else (tc,ta)::(maj c new_b qlb));;


let rec ajout lc (Noeud(b,lb)) = match lc with
  | []      ->  Noeud(true,lb)
  | c::qlc  ->  let arbre_c = match (recherche c lb) with
                | None    -> Noeud(false,[])
                | Some a  -> a
                in Noeud(c, maj c (ajout qlc arbre_c) lb);;


(* Exercice 4 *)

type ('a, 'b) trie = Trie of 'b arbre * ('a -> 'b list) * ('b list -> 'a)


(* TODO : PARTIE SUR LES TRIE A FINIR *)
let appartient_trie ...
  let lc = decomp mot in appartient lc a 
