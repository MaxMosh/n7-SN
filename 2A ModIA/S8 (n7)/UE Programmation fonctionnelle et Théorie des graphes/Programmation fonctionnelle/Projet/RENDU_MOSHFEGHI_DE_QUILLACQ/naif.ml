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
(* param type_encodage : type de l'encodage choisi *)
(* param c : lettre à encoder *)
(* précondition : le caractère c est entre a et z *)
let rec encoder_lettre type_encodage c =
  match type_encodage with
    |[]   ->  failwith("La touche associée au caractère n'a pas été trouvée")
    |t::q ->  let (num,liste_char) = t
              in 
              if List.mem c liste_char then (num, (index c liste_char))
              else encoder_lettre q c


let%test _ = encoder_lettre t9_map 'a' = (2,1)
let%test _ = encoder_lettre t9_map 'b' = (2,2)
let%test _ = encoder_lettre t9_map 'w' = (9,1)



(* fonction dupliquer *)
(* renvoie une liste contenant un certain nombre de fois un chiffre, abrégé par un 0 en fin de liste *)
(* param a : chiffre à duppliquer *)
(* param b : nombre de fois que l'on souhaite dupliquer le chiffre a *)
(* précondition : l'entier a est entre 0 et 9, et b est entre 1 et 3 *)
let rec dupliquer (a,b) =
  match b with
    (* lorsque b est égal à 0, on renvoie la liste [0] pour passer à la lettre suivante *)
    |0 -> [0]
    (* appel récursif : on concatène la lettre a avec la duplication de celle-ci b-1 fois *)
    |_ -> a::(dupliquer (a,b-1))


let%test _ = dupliquer (1,3) = [1;1;1;0]



(* fonction encoder_mot *)
(* renvoie un mot encodé sous forme d'une liste, ou chaque lettre encodée est "séparée" par un 0 *)
(* param type_encodage : type de l'encodage choisi *)
(* param mot : mot à encoder *)
let rec encoder_mot type_encodage mot =
  match mot with
    (* lorsque le mot est vide, on renvoie une liste vide *)
    |""   -> []
    (* appel récursif : sinon, on concatène la duplication de l'encodage du premier mot avec l'appel de la fonction sur la sous-chaîne de caractère sans le premier caractère *)
    |_-> (dupliquer (encoder_lettre type_encodage (String.get mot 0)))@(encoder_mot type_encodage (String.sub mot 1 (String.length mot - 1)))


let%test _ = encoder_mot t9_map "bonjour" = [2;2;0;6;6;6;0;6;6;0;5;0;6;6;6;0;8;8;0;7;7;7;0]



(* fonction decoder_lettre *)
(* renvoie le décodage d'une lettre *)
(* param type_encodage : type de l'encodage choisi *)
(* param a : touche tapée *)
(* param b : nombre de fois où la touche a est tapée *)
let rec decoder_lettre type_encodage (a,b) =
  match type_encodage with
    |[]     ->  failwith("Encodage non existant")
    |t::q   ->  let (num, liste_car) = t
                in
                if num = a then List.nth liste_car (b-1)
                else decoder_lettre q (a,b)


let%test _ = decoder_lettre t9_map (2,2) = 'b'
let%test _ = decoder_lettre t9_map (8,3) = 'v'



(* fonction split_zero *)
(* renvoie une liste de liste d'entiers, chaque liste dans la liste correspond à une suite d'entiers non séparés par des zéros dans la liste d'entrée *)
(* param liste : liste d'entiers en entrée *)
let split_zero liste =
    (* on définit la fonction f qui nous permet de "splitter" en liste de liste au niveau des zéros (courant stocke le rassemblement en cours) *)
    let f (courant,liste) x =
      match x with
       |0 ->  ([],liste@[courant])
       |_ ->  (x::courant,liste)
    in
    let (final_liste,liste_tot) = List.fold_left f ([],[[]]) liste 
    in
    List.tl liste_tot


let%test _ = split_zero [2;2;0;4;4;4;0;3;0] = [[2;2];[4;4;4];[3]]



(* fonction decoder_mot *)
(* renvoie le décodage d'un mot *)
(* param type_encodage : type de l'encodage choisi *)
(* param liste_num : liste de touches tapées avec multiplicité, le changement de lettre est marqué par un 0 *)
let decoder_mot type_encodage liste_num =
  (* définition d'une fonction locale associant à une liste un couple contenant la tête et la longueur de la liste *)
  let liste_vers_tuple l = (List.hd l, List.length l)
  (* le List.map de droite sépare la liste de numéro selon le découpage avec split_zero, puis utilise la fonction locale définie ci-dessus pour associer à chacun des découpages un tuple *)
  (* le List.map à gauche permet ensuite d'encoder selon le type d'encodage chaque tuple contenu dans la liste alors obtenue *)
  in let liste_car = List.map (decoder_lettre type_encodage) (List.map liste_vers_tuple (split_zero liste_num))
  in
  (* on convertit la liste de caractères en une liste de chaîne de caractères de taille 1 (avec List.to_seq) ; puis en une unique chaîne de caractères après concaténation (avec List.of_seq) *)
  String.of_seq (List.to_seq liste_car)


let%test _ = decoder_mot t9_map [2;2;0;6;6;6;0;6;6;0] = "bon"