with Ada.Text_IO;               use Ada.Text_IO;
with Ada.Strings.Unbounded;     use Ada.Strings.Unbounded;
with p_routeur_ll;		use p_routeur_ll;
with p_routeur_la; use p_routeur_la;
with adresse_ip;		use adresse_ip;
with Ada.Command_Line;          use Ada.Command_Line;
with Ada.Exceptions;            use Ada.Exceptions;	-- pour Exception_Message
with Ada.Text_IO.Unbounded_IO;  use Ada.Text_IO.Unbounded_IO;
with Ada.Integer_Text_IO;       use Ada.Integer_Text_IO;
with Ada.Strings;		use Ada.Strings;
with Ada.Float_Text_IO;		use Ada.Float_Text_IO;


-- Illustration de la lecture et de l'écriture de fichiers.
-- Ce programme attend un fichier sur la ligne de commande (par exemple,
-- exemple1.txt ou exemple2.txt) qui contient sur chaque ligne un nombre et un
-- texte.
procedure routeur_la is

   -- Lecture/Écriture sur des fichier .txt
   Nom_Table : Unbounded_String := To_Unbounded_String("table.txt");
   Nom_Paquets : Unbounded_String := To_Unbounded_String("paquets.txt");
   Nom_Resultats : Unbounded_String := To_Unbounded_String("resultats.txt");
   Sortie : File_Type;	-- Le descripteur du ficher de sortie
   Paquets : File_Type;
   Ligne : Unbounded_String;
   Numero_Ligne : Integer;
   Fini : Boolean := False;

   -- Variables du programme
   Table_Routage : T_LCA;
   Cache : T_arbre;
   Adresse_Paquet : T_Adresse_IP;
   Route_Presente : Boolean;
   Route : T_Adresse_IP := 0;
   Interf : Unbounded_String;
   Politique : Unbounded_String := To_Unbounded_String("LRU");
   Taille_Max : Integer := 10;
   Afficher_Stat : Boolean := True;
   Defaut_Cache : Integer := 0;
   Nbr_Paquets : Integer := 0;
   Precision_Cache : Integer := 0;
   Taux_Defaut : float;
   Date : Integer := 0;


   procedure Afficher_LCA(Adresse : in T_Adresse_IP; Masque : in T_Adresse_IP ; Interf : in Unbounded_String ; Frequence : in Integer := 0 ; Date : in Integer := 0) is

	begin
		Afficher_IP(Adresse); Put("    ");
		Afficher_IP(Masque); Put("    ");
		Put(Interf);
		New_Line;
	end Afficher_LCA;

	Procedure Afficher_LCA is new Pour_Chaque(Afficher_LCA);

	procedure Aidez_Moi is
		begin
			Put_Line("exemple d'usage : " & Command_Name & " -t <table> -p <paquets> -r <resultats>"); New_Line;
			Put_Line("Voici les options du routeur :"); New_Line;
			Put_Line("-c <taille>");
			Put_Line("Définir la taille du cache. <taille> est la taille du cache. La valeur 0 indique qu’il n y a pas de cache. La valeur par défaut est 10."); New_Line;
			Put_Line("-P FIFO|LRU|LFU");
			Put_Line("Définir la politique utilisée pour le cache (par défaut FIFO) "); New_Line;
			Put_Line("-s");
			Put_Line("Afficher les statistiques (nombre de défauts de cache, nombre de demandes de route, taux de défaut de cache). C’est l’option activée par défaut."); New_Line;
			Put_Line("-S");
			Put_Line("Ne pas afficher les statistiques."); New_Line;
			Put_Line("-t <fichier>");
			Put_Line("Définir le nom du fichier contenant les routes de la table de routage. Par défaut, on utilise le fichier table.txt."); New_Line;
			Put_Line("-p <fichier>");
			Put_Line("Définir le nom du fichier contenant les paquets à router. Par défaut, on utilise le fichier paquets.txt."); New_Line;
			Put_Line("-r <fichier>");
			Put_Line("Définir le nom du fichier contenant les résultats (adresse IP destination du paquet et inter-face utilisée). Par défaut, on utilise le fichier resultats.txt.");
			New_Line;
			Put_Line("Vous pouvez également consulter le manuel d'utilisation pour plus d'informations.");
		end Aidez_Moi;

begin
		-- Récupérer les arguments
		for i in 1..Argument_Count loop
			if To_Unbounded_String(Argument(i)) = "-t" then
				Nom_Table := To_Unbounded_String(Argument(i+1));
			elsif To_Unbounded_String(Argument(i)) = "-p" then
				Nom_Paquets := To_Unbounded_String(Argument(i+1));
			elsif To_Unbounded_String(Argument(i)) = "-P" then
				Politique := To_Unbounded_String(Argument(i+1));
			elsif To_Unbounded_String(Argument(i)) = "-r" then
				Nom_Resultats := To_Unbounded_String(Argument(i+1));
			elsif To_Unbounded_String(Argument(i)) = "-s" then
				Afficher_Stat := True;
			elsif To_Unbounded_String(Argument(i)) = "-S" then
				Afficher_Stat := False;
			elsif To_Unbounded_String(Argument(i)) = "-c" then
				Taille_Max := Integer'Value(Argument(i+1));
			elsif To_Unbounded_String(Argument(i)) = "-help" then
				Aidez_Moi;
				Fini := True;
			else
			    Null;
			end if;
		end loop;

		Creer_Table(Nom_Table, Table_Routage); -- Crée la liste chaînée qui contient la table de routage

		Precision_Cache := Precision (Table_Routage);

		Initialiser (Cache);

		Create (Sortie, Out_File, To_String (Nom_Resultats)); -- Crée le fichier des résultats
		Open (Paquets, In_File, To_String (Nom_Paquets)); -- Ouvre le fichier contenant les paquets à router

		begin
			while not Fini loop
				-- Si on attend de lire le texte, on sera déjà sur la ligne suivante.
				Numero_Ligne := Integer (Line (Paquets));

				Ligne := Get_Line (Paquets); -- Lire le reste de la ligne depuis le fichier Entree
				Trim (Ligne, Both);	        -- supprimer les blancs du début et de la fin de Texte
				Fini := End_Of_File (Paquets);
				-- AJOUT POUR TRAITEMENT COMMANDES
				-- Comme le case ne fonctionne pas pour des Unbounded_String, on a utilisé une structure de contrôle "Si"
				if Ligne = To_Unbounded_String("table") then
					Put(Ligne & " (ligne"); Put(Integer'Image(Numero_Ligne)); Put(")");
					New_Line;
					Afficher_LCA(Table_Routage);
					New_Line;

				elsif Ligne = To_Unbounded_String("cache") then
					Put(Ligne & " (ligne"); Put(Integer'Image(Numero_Ligne)); Put(")");
					New_Line;
					if Taille_Max /= 0 then
						Afficher (Cache);
					else
						Put_Line("Il s'agit du routeur simple");
					end if;
					New_Line;

				elsif Ligne = To_Unbounded_String("stat") then
					Put(Ligne & " (ligne"); Put(Integer'Image(Numero_Ligne)); Put(")");
					New_Line;
					if Taille_Max /= 0 then
						Taux_Defaut := Float(Defaut_Cache)/Float(Nbr_Paquets)*100.0;
						Put("Nombre de demandes de route : "); Put(Nbr_Paquets); New_Line;
						Put("Nombre de défaut de cache : "); Put(Defaut_Cache); New_Line;
						Put("Taux de défaut de cache : "); Put(Taux_Defaut, 2, 2, 0); Put_Line("%");
					else
						Put_Line("Il s'agit du routeur simple");
					end if;
					New_Line;
					New_Line;

				elsif Ligne = To_Unbounded_String("fin") then
					Put(Ligne & " (ligne"); Put(Integer'Image(Numero_Ligne)); Put(")");
					Fini := True;
				else
					Nbr_Paquets := Nbr_Paquets + 1;
					Put (Sortie, Ligne);
					Adresse_Paquet := Creer_Adresse(Ligne);
					if Taille_Max /= 0 then
						parcourir(Cache, Adresse_Paquet, Route_Presente, Route, Interf, Date);
						if Route_Presente then
							Put (Sortie, "    " & Interf);
						else
							Cherche_Route(Table_Routage, Adresse_Paquet, Route_Presente, Route, Interf, Politique, False, Date);
							if not Route_Presente then
                                                                    Put (Sortie, "    " & "Aucune route trouvée");
                                                                    Ajouter_IP (Cache, Precision_Cache, Taille_Max, Adresse_Paquet, Interf, Date,Politique ,Defaut_Cache );
							else
								Put (Sortie, "    " & Interf);
								Ajouter_IP (Cache, Precision_Cache, Taille_Max, Adresse_Paquet, Interf, Date,Politique ,Defaut_Cache );
							end if;
						end if;
					else
					    Cherche_Route(Table_Routage, Adresse_Paquet, Route_Presente, Route, Interf, Politique, False, Date);
					    if not Route_Presente then
								Put (Sortie, "    " & "Aucune route trouvée");
						else
								Put (Sortie, "    " & Interf);
						end if;
					end if;
					New_Line (Sortie);
				end if;
				-- FIN AJOUT
			end loop;
			if Afficher_Stat and Taille_Max /= 0 then
			    New_Line; New_Line;
				Taux_Defaut := Float(Defaut_Cache)/Float(Nbr_Paquets)*100.0;
				Put("Nombre de demandes de route : "); Put(Nbr_Paquets); New_Line;
				Put("Nombre de défaut de cache : "); Put(Defaut_Cache); New_Line;
				Put("Taux de défaut de cache : "); Put(Taux_Defaut, 2, 2, 0); Put_Line("%");
			else
				Null;
			end if;
		exception
			when End_Error =>
				Put ("Blancs en surplus à la fin du fichier.");
				null;
		end;
		Close (Paquets);
		Close (Sortie);
		Vider (Cache); Vider (Table_Routage);
exception
	when E : others =>
		Put_Line (Exception_Message (E)); New_Line;
		Put_Line("commande erronée, ./routeur -help pour plus d'informations concernant l'usage");
end routeur_la;
