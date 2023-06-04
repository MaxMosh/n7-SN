with SDA_Exceptions;         use SDA_Exceptions;
with Ada.Unchecked_Deallocation;

package body LCA is

	procedure Free is
			new Ada.Unchecked_Deallocation (Object => T_Cellule, Name => T_LCA);


	procedure Initialiser(Sda: out T_LCA) is
	begin
		Sda := Null;
	end Initialiser;


	function Est_Vide (Sda : T_LCA) return Boolean is
	begin
		return Sda = Null;
	end Est_Vide;


    function Taille (Sda : in T_LCA) return Integer is
    begin
        if Sda = Null then
            return 0;
        else
            return Taille(Sda.All.Suivant) + 1;
        end if;
	end Taille;


    procedure Enregistrer (Sda : in out T_LCA ; Cle : in T_Cle ; Donnee : in T_Donnee) is
        Nouvelle_Cellule : T_LCA;
    begin
        Nouvelle_Cellule := new T_Cellule;
        Nouvelle_Cellule.all.Cle := Cle;
        Nouvelle_Cellule.all.Donnee := Donnee;
        Nouvelle_Cellule.all.Suivant := Null;
        if Sda = Null then
            Sda := Nouvelle_Cellule;
        elsif Sda.All.Cle = Cle then
            Sda.All.Donnee := Donnee;
        elsif Sda.All.Suivant = Null then
            Sda.All.Suivant := Nouvelle_Cellule;
        else
            Enregistrer (Sda.All.Suivant, Cle, Donnee);
        end if;
	end;


	function Cle_Presente (Sda : in T_LCA ; Cle : in T_Cle) return Boolean is
    begin
        if Sda = Null then
            return False;
        elsif Sda.All.Cle = Cle then
            return True;
        else
            return Cle_Presente(Sda.All.Suivant, Cle);
        end if;
	end;


    function La_Donnee (Sda : in T_LCA ; Cle : in T_Cle) return T_Donnee is
    begin
        if Sda = Null then
            raise Cle_Absente_Exception;
        elsif Sda.all.Cle = Cle then
            return Sda.all.Donnee;
        else
            return La_Donnee(Sda.all.Suivant, Cle);
        end if;
	end La_Donnee;


	procedure Supprimer (Sda : in out T_LCA ; Cle : in T_Cle) is
		Sda_temp : T_LCA;								--on définit une variable de type LCA qui servira à isoler la cellule avec la clé que l'on veut supprimer
	begin
		if Sda = Null then
			raise Cle_Absente_Exception;
		elsif Sda.all.Cle = Cle then
			Sda_temp := Sda;
			Sda := Sda.all.Suivant;
			Sda_temp.all.Suivant := Null;
			Free(Sda_temp);
		elsif Sda.all.Suivant = Null then
			raise Cle_Absente_Exception;
		elsif Sda.all.Suivant.all.Cle = Cle then
			Sda_temp := Sda.all.Suivant;
			Sda.all.Suivant := Sda_temp.all.Suivant;
			Free(Sda_temp);
		else
			Supprimer(Sda.all.Suivant, Cle);
		end if;
	end Supprimer;


	procedure Vider (Sda : in out T_LCA) is
	begin
		if Sda /= Null then
			Vider (Sda.all.Suivant);
			Free (Sda);
		else
			Null;
		end if;
	end Vider;


	procedure Pour_Chaque (Sda : in T_LCA) is
		temp : T_LCA;
	begin
		temp := Sda;
		while temp /= Null loop
			Traiter(temp.all.Cle, temp.all.Donnee);
			temp := temp.all.Suivant;
		end loop;
	exception
		when others => Pour_Chaque(temp.All.Suivant);
	end Pour_Chaque;


end LCA;
