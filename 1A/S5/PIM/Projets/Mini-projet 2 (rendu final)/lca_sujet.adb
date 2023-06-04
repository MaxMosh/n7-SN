with lca;
with Ada.Text_IO; 				use Ada.Text_IO;
with Ada.Strings.Unbounded;		use Ada.Strings.Unbounded;

-- Programme de test du module lca.
procedure lca_sujet is

	package SDA_Entiers is
			new LCA (T_Cle => Unbounded_String, T_Donnee => Integer);
	use SDA_Entiers;

	procedure Ecrire_cellule(Cle : in Unbounded_String ; Donnee : in Integer) is
	begin
		Put("{ " & To_String(Cle) & " | " & Integer'image(Donnee) & " }");
		New_Line;
	end Ecrire_cellule;

	procedure Ecrire_LCA is
			new Pour_Chaque(Ecrire_Cellule);

	--Déclaration de la variable de type LCA sur lesquels on opère quelques instructions
	Sda_Test : T_LCA;
begin

	--On initialise une LCA
	Initialiser(Sda_Test);

	--On remplit la LCA
	Enregistrer(Sda_Test,  To_Unbounded_String("un"), 1);
	Enregistrer(Sda_Test,  To_Unbounded_String("deux"), 2);

	--On affiche la LCA
	Ecrire_LCA(Sda_Test);

end lca_sujet;
