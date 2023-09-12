with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with adresse_ip;	use adresse_ip;

-- Définition de structures de données associatives sous forme d'une liste chaînée associative (LCA).


package P_Routeur_LL is
    
    
	type T_LCA is limited private;

	-- Initialiser Liste.  Liste est vide.
	procedure Initialiser(Liste: out T_LCA) with
		Post => Est_Vide (Liste);

    -- Est-ce que Liste est vide ?
	function Est_Vide (Liste : T_LCA) return Boolean;

	-- Obtenir le nombre d'éléments de Liste. 
	function Taille (Liste : in T_LCA) return Integer with
		Post => Taille'Result >= 0
			and (Taille'Result = 0) = Est_Vide (Liste);

	-- Enregistrer les données associées à une Adresse dans Liste.
	-- Si l'adresse est déjà présente dans Liste, se données sont changées.
	procedure Enregistrer (Liste : in out T_LCA ; Adresse : in T_Adresse_IP ; Masque : in T_Adresse_IP ; Interf : in Unbounded_String ; Frequence : in Integer := 0 ; Date : in Integer := 0 ) with
		Post => Adresse_Presente (Liste, Adresse); 

	-- Supprimer une adresse dans Liste selon Politique.
	procedure Supprimer (Liste : in out T_LCA ; Politique : in Unbounded_String) with
		Post =>  Taille (Liste) = Taille (Liste)'Old - 1; -- un élément de moins
			
	-- Supprimer une adresse dans Liste selon la politique FIFO.
	procedure Supprimer_FIFO (Liste : in out T_LCA) with
		Post =>  Taille (Liste) = Taille (Liste)'Old - 1; -- un élément de moins
			
	-- Supprimer une adresse dans Liste selon la politique LRU.
	procedure Supprimer_LRU (Liste : in out T_LCA) with
		Post =>  Taille (Liste) = Taille (Liste)'Old - 1; -- un élément de moins
	
	-- Supprimer une adresse dans Liste selon la politique LFU.
	procedure Supprimer_LFU (Liste : in out T_LCA) with
		Post =>  Taille (Liste) = Taille (Liste)'Old - 1; -- un élément de moins

    -- Savoir si Adresse est présente dans Liste.
	function Adresse_Presente (Liste : in T_LCA ; Adresse : in T_Adresse_IP) return Boolean;

    -- Cherche Adresse dans Liste, si Adresse est retrouvée la fonction renvoit l'adresse de la cellule contenant Adresse sinon elle renvoit Null. [Le but de cette procédure est d'éviter d'appeler la fonction Adresse_Presente à chaque fois]
	function Recherche_Adresse (Liste : in T_LCA; Adresse : in T_Adresse_IP) return T_LCA;

	-- Supprimer tous les éléments de Liste.
	procedure Vider (Liste : in out T_LCA) with
		Post => Est_Vide (Liste);

	-- Appliquer un traitement (Traiter) pour chaque cellule de Liste.
	generic
		with procedure Traiter (Adresse : in T_Adresse_IP; Masque : in T_Adresse_IP ; Interf : in Unbounded_String ; Frequence : in Integer := 0 ; Date : in Integer := 0 );
	procedure Pour_Chaque (Liste : in T_LCA);


	-- Cherche la route qui convient à IP_A_Router dans Liste.
	procedure Cherche_Route (Liste : in T_LCA ; IP_A_Router : in T_Adresse_IP ; Route_Presente : out Boolean ; Route : in out T_Adresse_IP ; Interf : out Unbounded_String ; Politique : in Unbounded_String ; Est_Cache : in Boolean ; Date : in out Integer) with
		Post => (Not Route_Presente) or (Route_Presente and ( (Creer_Masque (Route) and IP_A_Router) = Route ) );

	procedure Creer_Table (Nom_Entree : in Unbounded_String ; Table : in out T_LCA);
	
	procedure Ajouter_IP (Cache : in out T_LCA; Politique : in Unbounded_String; Precision_Cache : in Integer; Taille : in out Integer; Taille_Max : in Integer; Adresse : in T_Adresse_IP; Interf : in Unbounded_String ; Defaut : in out Integer ; Date : in out Integer);
	
	function Precision (Table : in T_LCA) return Integer with
	    Post => (Precision'Result>=0 and Precision'Result<=4);

private

	type T_Cellule;
	type T_LCA is access T_Cellule;
	type T_Cellule is 
	    record
            Adresse : T_Adresse_IP;
            Masque : T_Adresse_IP;
            Interf : Unbounded_String;
            Frequence : Integer;
            Date : Integer;
            Suivant : T_LCA;
        end record;
        
end P_Routeur_LL;
