package body TH is
	-- Dans les garndes lignes, le principe est de réimplémenter chaque fonction faite pour les LCA avec les TH,
	-- en itérant sur les cases des TH.
	procedure Initialiser(Sda: out T_TH) is
	begin
		for i in 1..(Taille_TH) loop
			Initialiser(Sda(i));
		end loop;
	end Initialiser;


	function Est_Vide (Sda : T_TH) return Boolean is
		B_Vide : Boolean;
	begin
		B_Vide := True;
		for i in 1..Taille_TH loop
			if not Est_Vide(Sda(i)) then
				B_Vide := False;
			else
				Null;
			end if;
		end loop;
		return B_Vide;
	end Est_Vide;


	function Taille (Sda : in T_TH) return Integer is
		Somme : Integer;
	begin
		Somme := 0;
		for i in 1..Taille_TH loop
			Somme := Somme + Taille(Sda(i));
		end loop;
		return Somme;
	end Taille;


	procedure Enregistrer (Sda : in out T_TH ; Cle : in T_Cle ; Donnee : in T_Donnee) is
		Entier_H : constant Integer := Fonction_Hachage(Cle);
	begin

		Enregistrer(Sda(Entier_H), Cle, Donnee);
	end;


	function Cle_Presente (Sda : in T_TH ; Cle : in T_Cle) return Boolean is
	Entier_H : constant Integer := Fonction_Hachage(Cle);
	begin
		return Cle_Presente(Sda(Entier_H), Cle);
	end;


	function La_Donnee (Sda : in T_TH ; Cle : in T_Cle) return T_Donnee is
		Entier_H : constant Integer := Fonction_Hachage(Cle);
    begin
		return La_Donnee(Sda(Entier_H), Cle);
	end La_Donnee;


	procedure Supprimer (Sda : in out T_TH ; Cle : in T_Cle) is
		Entier_H : constant Integer := Fonction_Hachage(Cle);
	begin
		Supprimer(Sda(Entier_H), Cle);
	end Supprimer;


	procedure Vider (Sda : in out T_TH) is
	begin
		for i in 1..Taille_TH loop
			Vider(Sda(i));
		end loop;
	end Vider;


	procedure Pour_Chaque (Sda : in T_TH) is
		procedure Pour_Chaque is
			new LCA_Case.Pour_Chaque(Traiter => Traiter);
	begin
		for i in 1..Taille_TH loop
			Pour_Chaque(Sda(i));
		end loop;
	end Pour_Chaque;


end TH;
