with Ada.Strings.Unbounded;     use Ada.Strings.Unbounded;

package adresse_ip is

    	type T_Adresse_IP is mod 2 ** 32;

	UN_OCTET: constant T_Adresse_IP := 2 ** 8;       -- 256
	POIDS_FORT : constant T_Adresse_IP  := 2 ** 31;	 -- 10000000.00000000.00000000.00000000

	-- Affiche une adresse ip
	procedure Afficher_IP (Adresse : in T_Adresse_IP);
	
	
	-- Créer l'adresse ip à partir d'une chaîne de caractères.
	function Creer_Adresse (Adresse : Unbounded_String) return T_Adresse_IP;
	
	
	-- Créer le masque correspondant à une adresse.
	function Creer_Masque (Adresse : T_Adresse_IP) return T_Adresse_IP;
	
	
end adresse_ip;
