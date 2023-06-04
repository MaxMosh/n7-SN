with Ada.Text_IO;               use Ada.Text_IO;
with Ada.Integer_Text_IO;       use Ada.Integer_Text_IO;
with Ada.Strings.Unbounded;     use Ada.Strings.Unbounded;
with  Ada.Text_IO.Unbounded_IO; use  Ada.Text_IO.Unbounded_IO;

-- Exemple d'utilisation des Unbonded_String
procedure Exemple_Unbounded_String is
	Chaine: Unbounded_String;
begin
	-- Intialiser une Unbounded_String � partir d'une String
	Chaine := To_Unbounded_String ("Exemple");

	-- Afficher une Unbounded_String
	Put_Line ("La chaine est : "  & Chaine);	--! & concat�nation des cha�nes

	-- Longueur d'une Unbounded_String
	Put ("Sa longueur est : ");
	Put (Length (Chaine), 1);
	New_Line;

	-- L'initiale de la cha�ne
	Put_Line ("Son initiale est " & To_String (Chaine) (1));

	-- Ajouter une cha�ne � la fin d'une Unbounded_String
	Append (Chaine, " final");
	Put_Line ("La chaine apr�s concat�nation est : "  & Chaine);

	-- Lire au clavier une Unbounded_String
	Put ("Donner une cha�ne : ");
	Chaine := Get_Line;
	Put_Line ("La chaine lue est  : "  & Chaine);

end Exemple_Unbounded_String;
