with Ada.Text_IO;               use Ada.Text_IO;
with adresse_ip;		use adresse_ip;
with Ada.Integer_Text_IO;       use Ada.Integer_Text_IO;


package body adresse_ip is


	-- Affiche une adresse ip
	procedure Afficher_IP (Adresse : in T_Adresse_IP) is
		begin
			Put (Natural ((Adresse / UN_OCTET ** 3) mod UN_OCTET), 1); Put (".");
			Put (Natural ((Adresse / UN_OCTET ** 2) mod UN_OCTET), 1); Put (".");
			Put (Natural ((Adresse / UN_OCTET ** 1) mod UN_OCTET), 1); Put (".");
			Put (Natural  (Adresse mod UN_OCTET), 1);
		end Afficher_IP;
		
		
	-- Créer l'adresse ip à partir d'une chaîne de caractères.
	function Creer_Adresse (Adresse : Unbounded_String) return T_Adresse_IP is
	
	Adresse_IP : T_Adresse_IP := 0;
	Octet_Temp : Unbounded_String;
	i : Integer := 3;
	c : Character;
	begin
		for j in 1..Length(Adresse) loop
			c := Element(Adresse, j);
			if c /= '.' then
				Octet_Temp := Octet_Temp & c;
			else
				Adresse_IP := Adresse_IP + T_Adresse_IP'Base(Integer'Value(To_String(Octet_Temp))) * 2**(8*i);
				Octet_Temp := To_Unbounded_String("");
				i := i-1;
			end if;
		end loop;
		Adresse_IP := Adresse_IP + T_Adresse_IP'Base(Integer'Value(To_String(Octet_Temp))) * 2**(8*i);
		return Adresse_IP;
	end Creer_Adresse;	
	
	
	-- Créer le masque correspondant à une adresse.
	function Creer_Masque (Adresse : T_Adresse_IP) return T_Adresse_IP is
		
		Masque : T_Adresse_IP;
		
		begin
			Masque := -1;
			for i in 0..3 loop
				if (Adresse / UN_OCTET ** i MOD UN_OCTET) = 0 then
					Masque := Masque - 255 * UN_OCTET ** i;
				else
					Null;
				end if;
			end loop;
			return Masque;
		end Creer_Masque;	

	
end adresse_ip;
