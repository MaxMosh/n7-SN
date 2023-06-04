with lca;
with th;
with Ada.Integer_Text_IO;   use Ada.Integer_Text_IO;
with Ada.Text_IO; 				use Ada.Text_IO;
with Ada.Strings.Unbounded;		use Ada.Strings.Unbounded;

-- Programme de test du module th.
procedure th_sujet is
	TAILLE : constant Integer := 11;
	function fonction_HTest(Cle : Unbounded_String) return Integer is
	begin
		if length(Cle) <= TAILLE then
			return length(Cle);
		else
			return length(Cle) mod TAILLE + 1;
		end if;
	end fonction_HTest;

	package SDA_Entiers is
			new TH (T_Cle => Unbounded_String, T_Donnee => Integer, Taille_TH => TAILLE, Fonction_Hachage => fonction_HTest);
	use SDA_Entiers;


	procedure Ecrire_cellule(Cle : in Unbounded_String ; Donnee : in Integer) is
	begin
		Put("{ " & To_String(Cle) & " | " & Integer'image(Donnee) & " }");
		New_Line;
	end Ecrire_cellule;

	procedure Ecrire_TH is
			new Pour_Chaque(Ecrire_Cellule);

	-- Déclaration de la variable de type TH sur lesquels on opère quelques instructions, ainsi que de la
	-- taille souhaitée pour celle-ci
	Sda_Test : T_TH;
begin

	--On initialise une TH
	Initialiser(Sda_Test);

	--On remplit la TH
	Enregistrer(Sda_Test,  To_Unbounded_String("un"), 1);
	Enregistrer(Sda_Test,  To_Unbounded_String("deux"), 2);
	Enregistrer(Sda_Test,  To_Unbounded_String("trois"), 3);
	Enregistrer(Sda_Test,  To_Unbounded_String("quatre"), 4);
	Enregistrer(Sda_Test,  To_Unbounded_String("cinq"), 5);
	Enregistrer(Sda_Test,  To_Unbounded_String("quatre-vingt-dix-neuf"), 99);
	Enregistrer(Sda_Test,  To_Unbounded_String("vingt-et-un"), 21);

	--On affiche la TH
	Ecrire_TH(Sda_Test);
	Vider(Sda_Test);
end th_sujet;
