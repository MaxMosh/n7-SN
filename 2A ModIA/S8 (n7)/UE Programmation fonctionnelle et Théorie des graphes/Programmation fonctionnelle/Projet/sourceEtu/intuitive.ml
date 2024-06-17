open Encodage
open Chaines

(* Saisie des mots en mode T9 *)

(* fonction encoder_lettre *)
(* renvoie la touche à taper *)
(* param c : lettre à encoder *)
(* précondition : la lettre c est entre a et z *)
let rec encoder_lettre liste_asso c =
  match liste_asso with
    |[]   ->  failwith("La touche associée au caractère n'a pas été trouvée")
    |t::q ->  let (num,liste_char) = t
              in 
              if List.mem c liste_char then num
              else encoder_lettre q c

(* TODO : à finir (types string et liste) *)
let encoder_mot liste_asso mot =
  List.map encoder_lettre mot