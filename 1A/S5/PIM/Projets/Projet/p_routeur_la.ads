with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with adresse_ip;	use adresse_ip;


package p_routeur_la is

   type T_arbre is private;

	-- Initialiser un arbre.  L'arbre est vide.
	procedure Initialiser(arbre: out T_arbre) with
		Post => Est_Vide (arbre);

	-- Est-ce l'arbre est vide ?
	function Est_Vide (arbre : T_arbre) return Boolean;

	-- Obtenir le nombre d'éléments d'un arbre.
	function Taille (arbre : in T_arbre) return Integer ;

 -- On supprime une adresse dans l'arbre selon la politique LFU (LAST FREQUENTLY USED) : on supprime l'adresse la moins utilise du cache
    procedure Supprimer_LFU (arbre : in out T_arbre ) ;

 -- On supprime une adresse dans l'arbre selon la politique LFU (LAST RECENTLY USED) : on supprime l'adresse la moins recemment utilise du cache.
    procedure Supprimer_LRU (arbre : in out T_arbre ) ;

	-- Supprimer tous les éléments de l'arbre.
    procedure Vider (arbre : in out T_arbre);

    --on regarde si une adresse est presente dans l'arbre ou pas
   function adresse_presente(arbre : in T_arbre ; route : in T_adresse_IP ;Date:in out Integer ) return boolean;

    --on regarde si le noeud racine de l'arbre considere possede une adresse.
    function contient_adresse (arbre: in T_arbre) return boolean;

    --on affiche toute les adresse de l'arbre (accompagne de leur masque et interface).
    procedure Afficher(Cache:T_arbre);

    --on parcourt l'arbre pour voir si l'adresse en entree possede une route qui lui convient , on renvoit la route si il y en a une compatible.
   procedure parcourir (arbre : in T_arbre ; adresse : in T_Adresse_IP ; possede_cache :in out Boolean ; route : in out T_Adresse_IP;Interf:in out Unbounded_String; Date : in out Integer) ;

    --on traite l'ajout d'une route dans l'arbre
   procedure Ajouter_IP (arbre : in out T_arbre; Precision_Cache : in Integer; Taille_Max : in Integer; Adresse : in T_Adresse_IP; Interf : in Unbounded_String ; Date : in out Integer;Politique:in Unbounded_String;Defaut_Cache:in out Integer);

    --on enregistre une adresse dans l'arbre
   procedure Enregistrer(arbre : in out T_arbre ; Adresse : in T_Adresse_IP ; Masque : in T_Adresse_IP ; Interf : in Unbounded_String ; Frequence : in Integer := 0;Date : in out Integer);

private
    type T_Cellule;

   type T_arbre is access T_Cellule;

   type T_Cellule is
      record
         Adresse : T_Adresse_IP;
         Masque : T_Adresse_IP;      --masque de l'adresse
         Interf : Unbounded_String;  --interface de l'adresse
         Frequence : Integer;        --frequence d'utilisation de l'adresse dans l'arbre
         Date:Integer;               --derniere date d'utilisation de l'adresse dans l'arbre
         Suivant_g : T_arbre;        --noeuds gauche fils de l'arbre
         Suivant_d: T_arbre;         --noeuds droit fils de l'arbre
        end record;

end p_routeur_la;

