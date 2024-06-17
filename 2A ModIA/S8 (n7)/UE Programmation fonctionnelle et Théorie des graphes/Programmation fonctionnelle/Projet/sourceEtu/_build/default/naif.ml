open Encodage
open Chaines

(* Saisie des mots sans le mode T9 *)
(* Pour marquer le passage d'un caractère à un autre, on utilise la touche 0 *)

let rec index c liste =
  match liste with
    |[]   -> failwith("La touche associée au caractère n'a pas été trouvée")
    |t::q ->  if t = c then 1 
              else 1 + index c q

(* fonction encoder_lettre *)
(* renvoie la touche à taper ainsi que le nombre de fois où il faut taper dessus *)
(* param c : lettre à encoder *)
(* précondition : la lettre c est entre a et z *)
let rec encoder_lettre liste_asso c =
  match liste_asso with
    |[]   ->  failwith("La touche associée au caractère n'a pas été trouvée")
    |t::q ->  let (num,liste_char) = t
              in 
              if List.mem c liste_char then (num, (index c liste_char))
              else encoder_lettre q c


let%test _ = encoder_lettre t9_map 'a' = (2,1)
let%test _ = encoder_lettre t9_map 'b' = (2,2)
let%test _ = encoder_lettre t9_map 'w' = (9,1)



let rec dupliquer (a,b) =
  match b with
    |0 -> [0]
    |_ -> a::(dupliquer (a,b-1))

(* let rec encoder_mot liste_asso mot =
  match mot with
    |[]   -> []
    |t::q -> (dupliquer (encoder_lettre liste_asso t))@(encoder_mot liste_asso q) *)

let rec encoder_mot liste_asso mot =
  match mot with
    |""   -> []
    |_-> (dupliquer (encoder_lettre liste_asso (String.get mot 0)))@(encoder_mot liste_asso (String.sub mot 1 (String.length mot - 1)))


let%test _ = encoder_mot t9_map "bonjour" = [2;2;0;6;6;6;0;6;6;0;5;0;6;6;6;0;8;8;0;7;7;7;0]



let rec decoder_lettre liste_asso (a,b) =
  match liste_asso with
    |[]     ->  failwith("Encodage non existant")
    |t::q   ->  let (num, liste_car) = t
                in
                if num = a then List.nth liste_car (b-1)
                else decoder_lettre q (a,b)


let%test _ = decoder_lettre t9_map (2,2) = 'b'
let%test _ = decoder_lettre t9_map (8,3) = 'v'


(* let rec compte liste_num_finie_par_zero =
  match liste_num_finie_par_zero with
    |0::[]  -> 0
    |t::q -> 1 + compte q *)





    
let split_zero liste =
    (* On définit la fonction f qui nous permet de "splitter" en liste de liste au niveau des zéros (courant 
    stocke le rassemblement en cours) *)
    let f (courant,liste) x =
      match x with
       |0 ->  ([],liste@[courant])
       |_ ->  (x::courant,liste)
    in
    let (final_liste,liste_tot) = List.fold_left f ([],[[]]) liste 
    in
    List.tl liste_tot


(* let%test _ = encoder_mot t9_map "bonjour" = [2;2;0;6;6;6;0;6;6;0;5;0;6;6;6;0;8;8;0;7;7;7;0] *)

let decoder_mot liste_asso liste_num =
  let liste_vers_tuple l = (List.hd l, List.length l)
  in let liste_car = List.map (decoder_lettre liste_asso) (List.map liste_vers_tuple (split_zero liste_num))
  in
  String.of_seq (List.to_seq liste_car)