(* Slide 6/28 *)

let hd liste = match liste with
    | []            -> failwith "La liste est vide"
    | latete::_     -> latete
    

let tl liste = match liste with
    | []             -> failwith "La liste est vide"
    | _::laqueue     -> laqueue


let rec taille liste =
    match liste with
        | [] -> 0
        | _::queue -> 1 + taille queue


(* VERSION MOI : FAUX, ON NE PEUT PAS APPEND PAR LA DROITE *)
let rec append liste1 liste2 = 
    match liste2 with
        | [] -> liste1
        | tete::queue -> liste1::tete::(append liste1::tete queue)


(* VERSION AYOUB : VERSION FONCTIONNELLE *)
let rec append liste1 liste2 = 
    match liste1 with
        | [] -> liste2
        | tete::queue -> tete::(append queue liste2)


(* Slide 8/28 *)

(* Type de l'itérateur map *)
(* ('a->'b)->'a list->'b list *)
(* La valeur entre parenthèse correpond au type de départ de d'arrivée de la fonction f. *)

let rec map f liste =
    | [] -> []
    | tete::queue -> (f tete)::(map f queue)


(* Slide 11/28 *)

(* A FINIR *)
(* Type de l'itérateur fold_right *)
(* ('a->'b->'b)->'a list->'b->'b *)
let rec fold_right f liste e =
    match liste with
        | [] -> e
        | tete::queue ->


(* Type de l'itérateur fold_left *)
(* ('a->'b->'a)->'a->'b list->'a *)
let rec fold_left f e liste =
    match liste with
        | [] -> e
        | tete::queue -> f e queue
