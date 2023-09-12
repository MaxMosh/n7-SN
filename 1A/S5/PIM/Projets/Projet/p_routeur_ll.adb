with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Unchecked_Deallocation;
with Ada.Text_IO.Unbounded_IO;  use Ada.Text_IO.Unbounded_IO;
with Ada.Strings;               use Ada.Strings;	-- pour Both utilisé par Trim


package body P_Routeur_LL is

	procedure Free is
		new Ada.Unchecked_Deallocation (Object => T_Cellule, Name => T_LCA);

	-- Cherche Adresse dans Liste, si Adresse est retrouvée la fonction renvoit l'adresse de la cellule contenant Adresse sinon elle renvoit Null. [Le but de cette procédure est d'éviter d'appeler la fonction Adresse_Presente à chaque fois]
	function Recherche_Adresse (Liste : in T_LCA; Adresse : in T_Adresse_IP) return T_LCA is
	
	Adresse_Presente : Boolean;
	Temp : T_LCA;
	begin
		Temp := Liste;
		if Temp = Null then
			Adresse_Presente := False;
			return Null;
		else
			Adresse_Presente := (Temp.All.Adresse = Adresse);
			while (not Adresse_Presente) and (Temp /= Null) loop
				Temp := Temp.All.Suivant;
				if (Temp /= Null) then
					Adresse_Presente := (Temp.All.Adresse = Adresse);
				else
					Null;
				end if;
			end loop;
			if Adresse_Presente then
				return Temp;
			else
				return Null;
			end if;
		end if;
	end Recherche_Adresse;

	procedure Initialiser(Liste: out T_LCA) is
	begin
		Liste := Null;
	end Initialiser;

    -- Est-ce que Liste est vide ?
	function Est_Vide (Liste : T_LCA) return Boolean is
	begin
		return (Liste = Null);
	end;

	-- Fonction récursive qui calcule la taille d'une LCA
	function Taille (Liste : in T_LCA) return Integer is
	
		Suivant : T_LCA;
	
	begin
		if (Liste = Null) then
			return 0;
		else
			Suivant := Liste.All.Suivant;
			return 1 + Taille(Suivant);
		end if;
	end Taille;

	procedure Enregistrer (Liste : in out T_LCA ; Adresse : in T_Adresse_IP ; Masque : in T_Adresse_IP ; Interf : in Unbounded_String ; Frequence : in Integer := 0; Date : in Integer := 0) is
	
		Temp : T_LCA;
		Cellule_Adresse : T_LCA;
	
	begin
		Cellule_Adresse := Recherche_Adresse (Liste, Adresse);
		-- Dans ce cas une nouvelle cellule est créée puis ajoutée au début de la LCA
		if Cellule_Adresse = Null then
			Temp := new T_Cellule'(Adresse, Masque, Interf, Frequence, Date, Liste);
			Liste := Temp;
		-- Sinon on modifie la donnée
		else
			Cellule_Adresse.All.Masque := Masque;
			Cellule_Adresse.All.Interf := Interf;
			Cellule_Adresse.All.Frequence := Frequence;
			Cellule_Adresse.All.Date := Date;
		end if;
	end Enregistrer;


    -- Fonction récursive qui vérifie la présence d'une clé dans une LCA
	function Adresse_Presente (Liste : in T_LCA ; Adresse : in T_Adresse_IP) return Boolean is
	
		Suivant : T_LCA;
	
	begin
		if Est_Vide(Liste) then
			return False;
		else
			Suivant := Liste.All.Suivant;
			return (Liste.All.Adresse = Adresse or Adresse_Presente(Suivant, Adresse));
		end if;
	end;

	-- Supprime une adresse du cache selon la politique connée
	procedure Supprimer (Liste : in out T_LCA ; Politique : in Unbounded_String) is
	
	begin
		if Politique = To_Unbounded_String("FIFO") then
			Supprimer_FIFO(Liste);
		elsif Politique = To_Unbounded_String("LRU") then
			Supprimer_LRU(Liste);
		elsif Politique = To_Unbounded_String("LFU") then
			Supprimer_LFU(Liste);
		else
			Null;
		end if;
	end Supprimer;
	
	-- Supprimer une adresse dans Liste selon la politique FIFO.
	procedure Supprimer_FIFO (Liste : in out T_LCA) is
    
    Temp : T_LCA;
	begin
	    Temp := Liste;
	    if Temp.All.Suivant = Null then
	        Liste := Null;
	    else
	        while Temp.All.Suivant.All.Suivant /= Null loop
		        Temp := Temp.All.Suivant;
		    end loop;
		    Temp.All.Suivant := Null;
		 end if;
	end Supprimer_FIFO;
			
	-- Supprimer une adresse dans Liste selon la politique LRU.
	procedure Supprimer_LRU (Liste : in out T_LCA) is
	
	Date_Min : Integer;
	
	Temp_Parcours : T_LCA;
	Temp_Parcours_Avant : T_LCA;
	Cellule_Min : T_LCA;
	Cellule_Min_Avant : T_LCA;
	
	Begin
		Temp_Parcours := Liste;
		Temp_Parcours_Avant := Temp_Parcours;
		Cellule_Min := Liste;
		Cellule_Min_Avant := Cellule_Min;
		Date_Min := Temp_Parcours.All.Date;
		while Temp_Parcours /= Null loop
			if Temp_Parcours.All.Date < Date_Min then
				Date_Min := Temp_Parcours.All.Date;
				Cellule_Min := Temp_Parcours;
				Cellule_Min_Avant := Temp_Parcours_Avant;
			else
				Null;
			end if;
			Temp_Parcours_Avant := Temp_Parcours;
			Temp_Parcours := Temp_Parcours.All.Suivant;
		end loop;

		if Cellule_Min = Liste then
			Liste := Liste.All.Suivant;
		else
			Cellule_Min_Avant.All.Suivant := Cellule_Min.All.Suivant;
			Cellule_Min.All.Suivant := Null;
		end if;
	end Supprimer_LRU;
	
	-- Supprimer une adresse dans Liste selon la politique LFU.
	procedure Supprimer_LFU (Liste : in out T_LCA) is
	
	Freq_Min : Integer;
	
	Temp_Parcours : T_LCA;
	Temp_Parcours_Avant : T_LCA;
	Cellule_Min : T_LCA;
	Cellule_Min_Avant : T_LCA;
	
	Begin
		Temp_Parcours := Liste;
		Temp_Parcours_Avant := Temp_Parcours;
		Cellule_Min := Liste;
		Cellule_Min_Avant := Cellule_Min;
		Freq_Min := Temp_Parcours.All.Frequence;
		while Temp_Parcours /= Null loop
			if Temp_Parcours.All.Frequence <= Freq_Min then
				Freq_Min := Temp_Parcours.All.Frequence;
				Cellule_Min := Temp_Parcours;
				Cellule_Min_Avant := Temp_Parcours_Avant;
			else
				Null;
			end if;
			Temp_Parcours_Avant := Temp_Parcours;
			Temp_Parcours := Temp_Parcours.All.Suivant;
		end loop;
		
		if Cellule_Min = Liste then
			Liste := Liste.All.Suivant;
		else
			Cellule_Min_Avant.All.Suivant := Cellule_Min.All.Suivant;
			Cellule_Min.All.Suivant := Null;
		end if;

	end Supprimer_LFU;

	-- Permet de libérer la mémoire
	procedure Vider (Liste : in out T_LCA) is
	begin
		Free(Liste);
	end Vider;

	procedure Pour_Chaque (Liste : in T_LCA) is
	
	p : T_LCA;
	begin
		p := Liste;
		while p /= Null loop
			begin
				Traiter(p.All.Adresse, p.All.Masque, p.All.Interf, p.All.Frequence, p.All.Date);
				p := p.All.Suivant;
			exception
      				when others => Put("Le traitement n'a pas pu être effectué");
      					New_Line;
      					p := p.All.Suivant;
			end;
		end loop;
	end Pour_Chaque;

	-- Cherche la route qui convient à IP_A_Router dans Liste.

	procedure Cherche_Route (Liste : in T_LCA ; IP_A_Router : in T_Adresse_IP ; Route_Presente : out Boolean ; Route : in out T_Adresse_IP ; Interf : out Unbounded_String ; Politique : in Unbounded_String ; Est_Cache : in Boolean ; Date : in out Integer) is
		
		Masque_Max : T_Adresse_IP;
		Cellule_Temp : T_LCA;
		Cellule_Route : T_LCA;
		
		begin
			Cellule_Temp := Liste;
			Cellule_Route := Liste;
			Route_Presente := False;
			Masque_Max := 0;
			
			-- Recherche de la route convenable avec le masque le plus grand
			while Cellule_Temp /= Null loop
				if ( (IP_A_Router and Cellule_Temp.All.Masque) = Cellule_Temp.All.Adresse ) and ( Cellule_Temp.All.Masque >= Masque_Max ) then
					Route_Presente := True;
					Route := Cellule_Temp.Adresse;
					Interf := Cellule_Temp.Interf;
					Masque_Max := Cellule_Temp.All.Masque;
					Cellule_Route := Cellule_Temp;
					Cellule_Temp := Cellule_Temp.Suivant;
				else
					Cellule_Temp := Cellule_Temp.Suivant;
				end if;
			end loop;
			
			-- Si une route est retrouvée et qu'il s'agit du cache, on met à jour la fréquence ou la date de dernière utilisation de la route (suivant la politique)
			if Route_Presente and Est_Cache then
				if Politique = "LRU" then
					Cellule_Route.All.Date := Date + 1;
					Date := Date + 1;
				elsif Politique = "LFU" then
					Cellule_Route.All.Frequence := Cellule_Route.All.Frequence + 1;
				else
					Null;
				end if;
			else
				Null;
			end if;
			
		end Cherche_Route;

	-- Procédure pour transformer une table de routage au format texte en LCA
	
	procedure Creer_Table (Nom_Entree : in Unbounded_String ; Table : in out T_LCA) is

		Mot : Unbounded_String;
		Ligne : Unbounded_String;
		Index_Mot : Integer;
		Entree : File_Type;	-- Le descripteur du ficher d'entrée
		type T_Tab is array(1..3) of Unbounded_String;
		Tab : T_Tab; -- Le tableau qui contiendra les mots
		begin
			Open (Entree, In_File, To_String (Nom_Entree));
			begin
			Initialiser(Table);
				loop
				-- Si on attend de lire le texte, on sera déjà sur la ligne suivante.
				
					Ligne := Get_Line (Entree); -- Lire le reste de la ligne depuis le fichier Entree
					Trim (Ligne, Both);	        -- supprimer les blancs du début et de la fin de Texte
					Index_Mot := 1;
					Mot := To_Unbounded_String(" ");
					for I in 1..Length(Ligne) loop
						-- Si on rencontre des espaces, ajouter le mot au tableau
						if (Element(Ligne, I) = ' ') and (not (Element(Mot, Length(Mot)) = ' ')) then
							Trim (Mot, Both);
							Tab(Index_Mot) := Mot;
							Mot := To_Unbounded_String("");
							Index_Mot := Index_Mot + 1;
						else
							Null;
						end if;
						Mot := Mot & Element(Ligne, I);
					end loop;
					
					Trim (Mot, Both);
					Tab(Index_Mot) := Mot;
					
					-- AJOUT DE L'ADRESSE, MASQUE, INTERFACE À LA TABLE DE ROUTAGE
					Enregistrer(Table, Creer_Adresse(Tab(1)), Creer_Adresse(Tab(2)), Tab(3));
					exit when End_Of_File (Entree);	
				end loop;
			exception
				when End_Error =>
					Put ("Blancs en surplus à la fin du fichier.");
					null;
			end;
			Close (Entree);
	end Creer_Table;
	
	procedure Ajouter_IP (Cache : in out T_LCA; Politique : in Unbounded_String; Precision_Cache : in Integer; Taille : in out Integer; Taille_Max : in Integer; Adresse : in T_Adresse_IP; Interf : in Unbounded_String ; Defaut : in out Integer ; Date : in out Integer) is
		
		Temp_Route : T_Adresse_IP := 0;
		UN_OCTET : constant T_Adresse_IP := 2 ** 8; 
		
		begin
			-- Supprime une adresse dans le cache si ce dernier est plein
			if Taille >= Taille_Max then
				Supprimer (Cache, Politique);
				Taille := Taille - 1;
				Defaut := Defaut + 1;
			else
				Null;
			end if;
			
			-- la route complétée à Precision_Cache près)
			Temp_Route := Adresse - (Adresse MOD UN_OCTET ** Precision_Cache);
			
			Enregistrer (Cache, Temp_Route, Creer_Masque(Temp_Route), Interf, 1, Date);
			Date := Date + 1;
			Taille := Taille + 1;
		end Ajouter_IP;
		
	function Precision (Table : in T_LCA) return Integer is -- Cette fonction calcule la précision du cache (voir rapport pour plus d'informations)
	
	Cellule_Temp : T_LCA := Table;
	Masque_Max : T_Adresse_IP := Cellule_Temp.All.Masque;	
	Precision_Cache : Integer := 0;
		
	begin
		while Cellule_Temp /= Null loop
			if Cellule_Temp.All.Masque > Masque_Max then
				Masque_Max := Cellule_Temp.All.Masque;
			else
				null;
			end if;
			Cellule_Temp := Cellule_Temp.All.Suivant;
		end loop;
		
		while (Masque_Max / UN_OCTET ** Precision_Cache) mod UN_OCTET = 0 loop
			Precision_Cache := Precision_Cache + 1;
		end loop;
		
		return Precision_Cache;
	end Precision;
	
end P_Routeur_LL;
