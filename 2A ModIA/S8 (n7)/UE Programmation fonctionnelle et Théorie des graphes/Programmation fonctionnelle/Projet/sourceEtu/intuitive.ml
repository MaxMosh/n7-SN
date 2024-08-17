open Encodage
open Chaines

(* Saisie des mots en mode T9 *)

(* fonction encoder_lettre *)
(* renvoie la touche à taper *)
(* param type_encodage : type de l'encodage choisi *)
(* param c : lettre à encoder *)
(* précondition : le caractère c est entre a et z *)
let rec encoder_lettre type_encodage c =
  match type_encodage with
    |[]   ->  failwith("La touche associée au caractère n'a pas été trouvée")
    |t::q ->  let (num,liste_char) = t
              in 
              (* on a trouvé le bon numéro, on le renvoie *)
              if List.mem c liste_char then num
              (* on n'a pas trouvé le bon numéro, on appelle la fonction récursivement sur la queue *)
              else encoder_lettre q c


let%test _ = encoder_lettre t9_map 'b' = 2
let%test _ = encoder_lettre t9_map 'c' = 2
let%test _ = encoder_lettre t9_map 'd' = 3


(* fonction encoder_mot *)
(* renvoie la liste de touches correspondant à un mot en mode intuitif *)
(* param type_encodage : type de l'encodage choisi *)
(* param mot : mot à encoder *)
let encoder_mot type_encodage mot =
    (* on applique la fonction encoder_lettre à chacun des éléments de la liste *)
    List.map (encoder_lettre type_encodage) (List.of_seq (String.to_seq mot))


let%test _ = encoder_mot t9_map "bcd" = [2;2;3]
let%test _ = encoder_mot t9_map "bad" = [2;2;3]
let%test _ = encoder_mot t9_map "bon" = [2;6;6]



(* définition du type dico *)
type dico = Noeud of (string list * (int * dico) list)



(* définition du dictionnaire vide *)
let empty =
  Noeud ([], [])



(* fonction ajouter *)
(* ajoute un mot à un dictionnaire selon un encodage *)
(* param type_encodage : type de l'encodage choisi *)
(* param dictio : dictionnaire où l'on veut ajouter le mot *)
(* param mot : mot à ajouter (chaîne de caractères) *)
let ajouter type_encodage dictio mot=
  (* on définit une fonction nous permettant de remplir récursivement notre dictionnaire *)
  let rec ajouter_aux diction encodage_courant mot_tot=
    match (diction,encodage_courant) with
      (* dans le cas d'un noeud vide, on ajoute la tête de l'encodage courant dans le noeud courant et on fait l'appel récursif pour le noeud suivant avec la queue *)
      |Noeud([],[]),x::q->Noeud([],[(x,ajouter_aux empty q mot)])
      (* si on est au bout du mot, on peut l'ajouter au dictionnaire dans la liste de chaîne de caractères du noeud courant *)
      |Noeud(a,b),[]->Noeud(mot::a,b)
      (* autrement, on doit faire l'appel récursif sur l'ensemble des noeuds, on définit pour cela une seconde fonction auxiliaire f *)
      |Noeud(a,b),x::q->
        let rec f x2 q2 b2 =
          match b2 with
            |[]->[(x2,ajouter_aux empty q2 mot)]
            |z::e->let (numero,dictionnaire)=z in 
                if numero=x then (x2,ajouter_aux dictionnaire q2 mot)::e else z::(f x2 q2 e)
        in
        Noeud(a,f x q b)
      in
      ajouter_aux dictio (encoder_mot type_encodage mot) mot


let%test _ = ajouter t9_map empty "bonjour" = 
                    Noeud
                    ([],
                    [(2,
                      Noeud
                        ([],
                        [(6,
                          Noeud
                            ([],
                            [(6,
                              Noeud
                                ([],
                                [(5,
                                  Noeud
                                    ([],
                                    [(6,
                                      Noeud
                                        ([],
                                        [(8,
                                          Noeud
                                            ([], [(7, Noeud (["bonjour"], []))]))]))]))]))]))]))])



(* À PARTIR D'ICI, NOUS NE FORMULERONS PLUS LES TESTS QUI IMPLIQUENT DES DICTIONNAIRES CAR ILS RENDENT LE CODE TROP ILLISIBLE *)
(* un fichier "dictio_petit.txt" a été créé pour les tests visuels dans le terminal de chacune des fonctions (avec dune utop) *)

(* les deux fonctions suivantes servent à créer un dicitionnaire selon un fichier .txt *)
let read_lines filename =
  let ic = open_in filename in
  let rec read_lines_aux acc =
    try
      let line = input_line ic in
      read_lines_aux (line :: acc)
    with End_of_file ->
      close_in ic;
      List.rev acc
  in
  read_lines_aux []


let creer_dico type_encodage chemin =
  let liste_mots = read_lines chemin
  in
  List.fold_left (ajouter type_encodage) empty liste_mots



(* fonction suppr_string *)
(* supprime un mot d'un dictionnaire, en retirant simplement la chaîne de caractères correspondante de sa liste de mots pour le noeud concerné *)
(* param dictionnaire : dictionnaire d'où l'on veut retirer le mot *)
(* param mot : mot à retirer (chaîne de caractères) *)
(* param mot_encode : mot encodé (correspondant à l'encodage du paramètre mot, permet de le garder dans les appels récursifs) *)
let rec suppr_string dictionnaire mot mot_encode =
  match (dictionnaire,mot_encode) with
  (* si le noeud est vide, il n'y a rien à supprimer, on renvoie le dictionnaire *)
  |Noeud([],[]),x::q->dictionnaire
  (* si on est au bout du mot, on observe si dans le noeud courant est contenu notre mot souhaité, que l'on supprile si on le trouve *)
  |Noeud(a,b),[]->
    let rec g mot a =
      match a with
        |[]->[]
        |z::e->if (String.equal z mot) then e else z::(g mot e)
    in
    Noeud(g mot a,b)  
  (* autrement, on doit faire l'appel récursif de manière similaire à la fonction ajouter (nouvelle fonction auxiliaire...) *)
  |Noeud(a,b),x::q->
    let rec f x2 q2 b2 =
      match b2 with
        |[]->[]
        |z::e->let (numero,dictionnaire)=z in 
            if numero=x2 then (x2,suppr_string dictionnaire mot q2)::e else z::(f x2 q2 e)
    in
    Noeud(a,f x q b)



(* fonction elague_noeud *)
(* retire les noeuds inutiles d'un dictionnaire (dans le cas où le mot a été supprimé avec suppr_string et qu'il était seul dans son noeud) *)
(* param dictio : dictionnaire où l'on veut nettoyer les noeuds inutiles *)
let rec elague_noeud dictio = match dictio with
  | Noeud (mots, branches) ->
      (* élaguer récursivement chaque branche *)
      let f (num,sous_noeud) =
        let sous_noeud_epure = elague_noeud sous_noeud in
        match sous_noeud_epure with
        (* on supprime la branche inutile *)
        | Noeud ([], []) -> None
        | _ -> Some (num, sous_noeud_epure)
      in 
      let branches_epurees = List.filter_map f branches in
      Noeud (mots, branches_epurees)



(* fonction supprimer *)
(* retire un mot et élague ensuite les noeuds inutiles *)
(* param type_encodage : type de l'encodage choisi *)
(* param dictionnaire : dictionnaire d'où l'on veut supprimer le mot *)
(* param mot : mot que l'on souhaite supprimer *)
let supprimer type_encodage dictionnaire mot =
  let dictio1 = suppr_string dictionnaire mot (encoder_mot type_encodage mot)
  in elague_noeud dictio1



(* fonction appartient *)
(* vérifie si un mot appartient à un dictionnaire *)
(* param type_encodage : type de l'encodage choisi *)
(* param dictio : dictionnaire dans lequel on recherche le mot *)
(* param mot : mot que l'on cherche *)
let rec appartient type_encodage dictio mot =
  match dictio with 
    |Noeud([],[])-> false
    |Noeud(a,b)-> let g (num,liste)=(appartient type_encodage liste mot)
    in
    (List.mem mot a) || List.fold_left (fun x acc -> acc || x) false (List.map g b)



(* fonction occurence_mot *)
(* compte le nombre de fois où un mot est dans un dictionnaire *)
(* param type_encodage : type de l'encodage choisi *)
(* param dictio : dictionnaire dans lequel on compte le nombre d'occurences du mot *)
(* param mot : mot que l'on compte *)
let rec occurence_mot type_encodage dictio mot =
  match dictio with 
    |Noeud([],[])-> 0
    |Noeud(a,b)-> let g (num,liste)=(occurence_mot type_encodage liste mot) in
    List.fold_left (fun x acc -> acc+ x) 0 (List.map (fun x -> if String.equal x mot then 1 else 0) a) + List.fold_left (fun x acc -> acc + x) 0 (List.map g b)



(* fonction mot_coherent *)
(* vérifie qu'un mot est bien associé à un encodage *)
(* param type_encodage : type de l'encodage choisi *)
(* param mot : mot que l'on souhaite vérifier *)
(* param chemin : encodage que l'on souhaite tester *)
let mot_coherent type_encodage mot chemin =
  let encodage_mot = encoder_mot type_encodage mot in
  encodage_mot = chemin


let%test _ = mot_coherent t9_map "bonjour" (encoder_mot t9_map "bonjour") = true
let%test _ = mot_coherent t9_map "bonjour" (encoder_mot t9_map "bon") = false



(* fonction coherent_aux *)
(* vérifie la cohérence d'un noeud et de ses sous-noeuds *)
(* param type_encodage : type de l'encodage choisi *)
(* param dictionnaire : dictionnaire que l'on vérifie *)
(* param chemin : chemin que l'on vérifie *)
let rec coherent_aux type_encodage dictionnaire chemin =
  match dictionnaire with
  | Noeud (mots, sous_noeuds) ->
    (* vérifie la cohérence des mots dans le noeud actuel *)
    let mots_coherents = List.for_all (fun mot -> mot_coherent type_encodage mot chemin) mots
    in
    (* vérifie la cohérence des sous-noeuds *)
    let sous_noeuds_coherents = List.for_all (fun (num, sous_dico) ->
      coherent_aux type_encodage sous_dico (chemin @ [num])) sous_noeuds 
    in
    mots_coherents && sous_noeuds_coherents

(* fonction coherent *)
(* vérifie la cohérence d'un dictionnaire *)
(* param type_encodage : type de l'encodage choisi *)
(* param dico : dictionnaire que l'on vérifie *)
let coherent type_encodage dico =
  coherent_aux type_encodage dico []



(* fonction decoder_mot *)
(* renvoie la liste de mots correspondant à un code *)
(* param dictio : dictionnaire dans lequel on cherche la liste de mots *)
(* param mot_encode : code de la liste de mots que l'on cherche *)
let rec decoder_mot dictio mot_encode =
  match dictio, mot_encode with
  (* si on est au bout de l'encodage, on peut renvoyer la liste de chaîne de caractères du noeud courant *)
  | Noeud(a, _), [] -> a
  (* sinon, on fait l'appel récursif *)
  | Noeud(_, b), t::q ->
    (* on cherche le noeud qui correspond à l'encodage actuel *)
    let rec f noeuds =
      match noeuds with
      | [] -> []
      | (num, sous_dico)::reste ->
        if num = t then decoder_mot sous_dico q else f reste
    in f b



(* fonction prefixe *)
(* renvoie la liste de mots présentant un certain préfixe encodé *)
(* param dictio : dictionnaire dans lequel on cherche la liste de mots *)
(* param prefixe_encode : préfixe encodé de la liste de mots que l'on cherche *)
let rec prefixe dictio prefixe_encode =
  match dictio, prefixe_encode with
  (* si on est au bout du prefixe, on retounre tous les mots à partir du noeud actuel *)
  | Noeud(a, b), [] -> 
    let rec f1 dictio =
      match dictio with 
      | Noeud(a, b) -> 
        let g (num, sous_dico) = f1 sous_dico in
        a @ List.fold_left (fun acc x -> acc @ x) [] (List.map g b)
    in f1 dictio
    (* sinon on cherche la branche qui correspond à l'encodage du préfixe courant *)
  | Noeud(_, b), t::q ->
    let rec f branches =
      match branches with
      | [] -> []
      | (num, sous_dico)::reste ->
        if num = t then prefixe sous_dico q else f reste
    in f b



(* fonction max_mots_code_identique *)
(* renvoie la longueur de la liste de mots associé à un noeud la plus grande dans un dictionnaire *)
(* param dictio : dictionnaire dans lequel on cherche la longueur de la liste de mots la plus grande *)
let rec max_mots_code_identique dictio =
  match dictio with 
    |Noeud([],[])-> 0
    |Noeud(a,b)-> let g (num,liste)=(max_mots_code_identique liste) in
      max (List.length a) (List.fold_left (fun x acc ->max acc x) 0 (List.map g b))



(* fonction lister *)
(* renvoie sous forme d'une liste l'ensemble des mots d'un dictionnaire *)
(* param dictio : dictionnaire que l'on veut "aplatir" *)
let rec lister dictio =
  match dictio with 
    |Noeud([],[])-> []
    (*  *)
    |Noeud(a,b)-> let g (num,liste)=(lister liste) in
       a @ (List.fold_left (fun x acc -> acc @ x) [] (List.map g b))
(* remarque : on aurait aussi pu définir cette fonction comme un appel à la fonction prefixe avec la liste [] *)