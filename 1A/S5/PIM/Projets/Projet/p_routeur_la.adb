with Ada.Unchecked_Deallocation;
with ada.Text_IO; use ada.Text_IO;
with Ada.Text_IO.Unbounded_IO;  use Ada.Text_IO.Unbounded_IO;

package body p_routeur_la is

   UN_OCTET: constant T_Adresse_IP := 2 ** 8;       -- 256
   POIDS_FORT : constant T_Adresse_IP  := 2 ** 31;	 -- 10000000.00000000.00000000.00000000²
   Exception_adresse:Exception;

    --on affiche toute les adresse de l'arbre (accompagne de leur masque et interface).
    procedure Afficher(Cache:T_arbre) is

    begin
        if Est_Vide(Cache) then
            null;
        elsif contient_adresse(Cache) then               -- si le noeud possede une adresse on l'affiche avec son masque et interface
            Afficher_IP(cache.all.Adresse); Put("    ");
            Afficher_IP(cache.all.Masque); Put("    ");
         Put(Cache.all.Interf);
         New_Line;
        else                                             --sinon on parcours le reste de l'arbre
            Afficher(Cache.all.Suivant_g);
            Afficher(Cache.all.Suivant_d);
        end if;
    end Afficher;

   procedure Free is
            new Ada.Unchecked_Deallocation(Object => T_Cellule,Name=>T_arbre);

    -- Initialiser un arbre.  L'arbre est vide.
    procedure Initialiser(arbre: out T_arbre) is
    begin
        arbre:=null;
   end Initialiser;

    --la fonction renvoit le minimum entre deux entier.
    function min(a,b:integer) return integer is
    begin
        if a>b then
            return b;
        else
            return a;
        end if ;
    end min;

    -- On supprime une adresse dans l'arbre selon la politique LFU (LAST FREQUENTLY USED) : on supprime l'adresse la moins utilise du cache.
   Procedure Supprimer_LFU (arbre : in out T_arbre) is

        --on renvoit la plus petite frequence d'une adresse dans l'arbre
      function freq_min(arbre:T_arbre) return integer is
      begin
         if Est_Vide(arbre) then   --on veut pas que les feuilles influe sur la fonction donc il faut prendre un nombre tres grand pour eviter que le minimum soit modifie
            return 100000000;
         elsif contient_adresse(arbre) then
            return arbre.all.Frequence;
         else
            return min(freq_min(arbre.all.suivant_g),freq_min(arbre.all.suivant_d));   --on fait le minimum de tout les adresse de l'arbre
         end if;
      end freq_min;

       --on renvoit une adresse qui possede la frequence en entier
      procedure adresse_freq(arbre:T_arbre;freq:Integer;rep:in out T_adresse_IP) is
      begin
         if Est_Vide(arbre) then
            null;
         elsif contient_adresse(arbre) then
            if arbre.all.Frequence=freq then
               rep:=arbre.all.adresse;          --dans le cas ou plusieurs adresse ont la meme frequence ce sera la derniere adresse parcouru par la fonction qui sera renvoye
            else
                  null;
            end if;
         else
            adresse_freq(arbre.all.suivant_g,freq,rep);
            adresse_freq(arbre.all.suivant_d,freq,rep);
            end if;
      end adresse_freq;


        -- cette fonction supprime l'adresse en entree
        --il n'y a pas besoin de faire une exception dans le cas ou l'adresse n'est pas trouve
        --en effet ne servant que dans les fonction supprime , elle aura toujours en entree une fonction presente dans l'arbre
        procedure sup_adresse(arbre:in out T_arbre;adresse:T_adresse_IP) is
        begin
         if Est_Vide(arbre) then
            null;
         elsif contient_adresse(arbre) then
            if arbre.all.Adresse=adresse then
               Vider(arbre);                   -- si on trouve le noeuds contenant l'adresse on peut le vider car ses noeuds fils sont vide
            else
               null;
            end if;
         else
            sup_adresse(arbre.all.suivant_g,adresse);
            sup_adresse(arbre.all.suivant_d,adresse);
         end if;
      end sup_adresse;

        rep:T_adresse_IP;  --rep correponds a l'adresse la moins frequente.
        freq:Integer;      --elle corresponds a la plus petite frequence.
    begin
      if Est_Vide(arbre) then
         null;
      else
         rep:=0;
         Freq:=freq_min(arbre);
         adresse_freq(arbre,freq,rep);
         sup_adresse(arbre,rep);       -- comme dit precedemment rep est la sortie de adresse_freq et donc est necessairement presente dans l'arbre
      end if;
end supprimer_LFU;

   --On vide l'arbre
   procedure Vider(arbre:in out T_arbre) is
   begin
      if Est_Vide(arbre) then
         null;
      elsif (contient_adresse(arbre)) then
         null;
      else
         vider(arbre.all.suivant_g);
         vider(arbre.all.suivant_d);
      end if;
      free(arbre);
   end Vider;

    -- On supprime une adresse dans l'arbre selon la politique LFU (LAST RECENTLY USED) : on supprime l'adresse la moins recemment utilise du cache.
   Procedure Supprimer_LRU (arbre : in out T_arbre) is

      --on essaie de trouver la date la plus ancienne
      function date_min(arbre:T_arbre) return integer is
      begin
         if Est_Vide(arbre) then
            return 1000000000;               --meme commentaire que la fonction freq_min
         elsif contient_adresse(arbre) then
            return arbre.all.Date;
         else
            return min(date_min(arbre.all.suivant_g),date_min(arbre.all.suivant_d));
         end if;
      end date_min;

        -- on veut supprimer l'adresse contenant la date en entree
        --pas besoin de mettre d'exception la fonction renverra necessairement une date contenue dans l'arbre
      procedure adresse_date(arbre:T_arbre;date:Integer;rep:in out T_adresse_IP) is
      begin
         if Est_Vide(arbre) then
            null;
         elsif contient_adresse(arbre) then
            if arbre.all.date=date then
               rep:=arbre.all.adresse;
            else
                  null;
            end if;
         else
            adresse_date(arbre.all.suivant_g,date,rep);
            adresse_date(arbre.all.suivant_d,date,rep);
            end if;
      end adresse_date;

        procedure sup_adresse(arbre:in out T_arbre;adresse:T_adresse_IP) is
        begin
         if Est_Vide(arbre) then
            null;
         elsif contient_adresse(arbre) then
            if arbre.all.Adresse=adresse then
               Vider(arbre);
            else
               null;
            end if;
         else
            sup_adresse(arbre.all.suivant_g,adresse);
            sup_adresse(arbre.all.suivant_d,adresse);
         end if;
      end sup_adresse;

        rep:T_adresse_IP;
        date:Integer;
    begin
      if Est_Vide(arbre) then
         null;
      else
         rep:=0;
         Date:=date_min(arbre);
         adresse_date(arbre,date,rep);
         sup_adresse(arbre,rep);
      end if;
end supprimer_lru;

     --on parcourt l'arbre pour voir si l'adresse en entree possede une route qui lui convient , on renvoit la route si il y en a une compatible.
    procedure parcourir (arbre : in T_arbre ; adresse : in T_Adresse_IP ; possede_cache :in out Boolean ; route : in out T_Adresse_IP;Interf:in out Unbounded_String; Date : in out Integer) is
        curent:T_Adresse_IP;
        --on creer une fonction auxiliaire pour ajouter la variable curent qui sera initialisé a adresse.
        procedure parcourir_aux (arbre :in  T_arbre ; adresse:in T_adresse_IP ;possede_cache:in out Boolean;route: in out T_Adresse_IP;curent:in out T_Adresse_IP;interf:in out Unbounded_String) is
            Bit_A_1 : Boolean;
        begin
            Bit_A_1 := (curent and POIDS_FORT) /= 0;  --a chaque iteration on recupere le 1er bit de l'adresse
            if Est_Vide(arbre) then                   --si on tombe sur l'arbre vide notre parcours est termine l'adresse n'a pas de route compatible
                possede_cache:=false;
            elsif contient_adresse(arbre) then        --si on tombe sur une route on verifie si elle est bien compatible
                if (adresse and arbre.all.Masque)=arbre.all.Adresse then  --Pour savoir si elle est compatible on regarde la partie masquable de la route et la meme que celle de l'adresse
                    route:=arbre.adresse;
                    Interf:=arbre.Interf;
                    arbre.Date := Date;
                    possede_cache:=True;
                else
                    possede_cache:=False;
                end if;
            else
                if Bit_A_1 then                      --si le premier bit est 1 on va a droite
                    curent:=curent*2;                --on decale l'adresse vers la droite pour que a l'appelle recursive la procedure regarde le bit d'apres
                     parcourir_aux(arbre.all.Suivant_d,adresse,possede_cache,route,curent,interf);
                else                                 --si le premier bit est 0 on va a gauche
                    curent:=curent*2;
                     parcourir_aux(arbre.all.Suivant_g,adresse,possede_cache,route,curent,interf);
                end if;
            end if;
        end parcourir_aux;

   begin
      if Est_Vide(arbre) then
         possede_cache:=False;
        else
            curent:=adresse;
            parcourir_aux(arbre,adresse,possede_cache,route,curent,interf);
      end if;
    end parcourir;

    -- Supprimer tous les éléments de l'arbre.
    function Est_Vide(arbre: in T_arbre) return boolean is
        begin
            return arbre=null;
        end Est_Vide;


    -- Obtenir le nombre d'éléments d'un arbre.
   function Taille (arbre : in T_arbre) return Integer is
      begin
        if Est_Vide(arbre) then
         return 0;
      elsif contient_adresse(arbre)then  --on ajoute seulement lorsque un noeud possede une adresse
         return 1;
      else
         return Taille(arbre.all.suivant_g)+Taille(arbre.all.suivant_d);
      end if;
    end Taille;

     --on regarde si une adresse est presente dans l'arbre ou pas
   function adresse_presente (arbre : in T_arbre ; route : in T_adresse_IP ;Date:in out Integer) return boolean is

        --meme principe que pour la fonction parcourir
      function adresse_presente_aux(arbre : in  T_arbre ; route : in T_adresse_IP ;curent : in T_Adresse_IP;Date:in out Integer) return boolean is
         Bit_A_1:Boolean;  --premier bit de la route

      begin
         Bit_A_1 := (curent and POIDS_FORT) /= 0;
         if Est_Vide(arbre) then        -- si notre parcours meme a du vide c'est que l'adresse n'existe pas
            return false;
         elsif contient_adresse(arbre) then  -- si on tombe sur une adresse alors on compare
            if arbre.all.adresse=route then
               arbre.all.Frequence:=arbre.all.Frequence+1;  -- si ladresse est presente on ajoute 1 a sa frequence d'utilisation
               arbre.all.Date:=Date;                        --si l'adresse est presente on modifie alors sa date de derniere utilisation
               return True;
            else
               return false;
            end if;
         else                                               --sinon on prcourt la fonction comme parcourir
            if Bit_A_1 then
               return adresse_presente_aux(arbre.all.Suivant_d,route, (curent*2),Date);
            else
               return adresse_presente_aux(arbre.all.Suivant_g,route,  (curent*2),Date);
            end if;
         end if;
      end adresse_presente_aux;
   begin
      return adresse_presente_aux(arbre,route,route,Date);
   end adresse_presente;

    --on regarde si le noeud racine de l'arbre considere possede une adresse
   function contient_adresse (arbre: in T_arbre) return boolean is
   begin
      if Est_Vide(arbre) then
         return false;
      else
         return Est_Vide(arbre.all.Suivant_g)and Est_Vide(arbre.all.Suivant_d);  -- l'arbre possede une adresse dans sa racine si ses 2 fils sont vides.
      end if;
   end;

     --on enregistre une adresse dans l'arbre
    procedure Enregistrer(arbre : in out T_arbre ; Adresse : in T_Adresse_IP ; Masque : in T_Adresse_IP ; Interf : in Unbounded_String ; Frequence : in Integer := 0 ;Date:in out integer) is

        curent:T_Adresse_IP;
        profondeur:Integer;   --profondeur de l'arbre dans laquelle on se situe

        --meme raison que pour les fonction parcourir et adresse presente
      procedure auxiliaire(arbre : in out T_arbre ; Adresse : in T_Adresse_IP ; Masque : in T_Adresse_IP ; Interf : in Unbounded_String ; Frequence : in Integer := 0;Date:in Integer;curent:in out T_Adresse_IP;profondeur:in out Integer) is
         Bit_A_1:Boolean;
         Bit_A_1_route:Boolean;    --cette fois si on va devoir comparer le i eme bit de l'adresse avec celui de la route qu'il recontrera lors de son parcours
        begin
            if profondeur>31 then
                raise Exception_adresse;   --si on arrive a une profondeur de plus de 32 cest que les 2 adresse sont identique ce qui risque faire boucler a linfini la fonction recursive
            else                           --en regle generale dans la fonction ajoute ip ce cas ne peut arriver mais etant donner que cette fonction et accessible par l'utilisateur il est necessaire de mettre en place une exception
                if Est_Vide(arbre) then    --si larbre est vide on rajoute l'adresse ici
                    arbre:=new T_Cellule'(Adresse,Masque,Interf,Frequence,date,null,null);
                elsif contient_adresse(arbre) then
                    Bit_A_1 := (curent and POIDS_FORT) /= 0;
                    Bit_A_1_route:=((arbre.all.Adresse*(2**profondeur)) and POIDS_FORT) /= 0;
                    if Bit_A_1 then
                        if Bit_A_1_route then   --si les 2 bit compare sont des 1 alors on copie la route (avec masque et interface) dans le fils droit initiallement vide et on compare les bit dapres
                            arbre.all.Suivant_d:=new T_Cellule'(arbre.all.Adresse,arbre.all.Masque,arbre.all.Interf,arbre.all.Frequence,arbre.all.Date,null,null);
                            profondeur:=profondeur+1;
                            curent:=curent*2;
                            auxiliaire(arbre.all.Suivant_d,Adresse,Masque,Interf,Frequence,Date,curent,profondeur);
                        else                   --sinon on met ladresse dans le fils droit et la route dans le gauche
                            arbre.all.Suivant_d:=new T_Cellule'(Adresse,Masque,Interf,Frequence,Date,null,null);
                            arbre.all.Suivant_g:=new T_Cellule'(arbre.all.Adresse,arbre.all.Masque,arbre.all.Interf,arbre.all.Frequence,arbre.all.Date,null,null);
               end if;
                    else
               if not(Bit_A_1_route) then       --si les 2 bit compare sont des 0 alors on copie la route (avec masque et interface) dans le fils gauche initiallement vide et on compare les bit dapres
                            arbre.all.Suivant_g:=new T_Cellule'(arbre.all.Adresse,arbre.all.Masque,arbre.all.Interf,arbre.all.Frequence,arbre.all.Date,null,null);
                            profondeur:=profondeur+1;
                            curent:=curent*2;
                            auxiliaire(arbre.all.Suivant_g,Adresse,Masque,Interf,Frequence,Date,curent,profondeur);
                        else                   --sinon on met ladresse dans le fils gauche et la route dans le droit
                            arbre.all.Suivant_d:=new T_Cellule'(arbre.all.Adresse,arbre.all.Masque,arbre.all.Interf,arbre.all.Frequence,arbre.all.Date,null,null);
                            arbre.all.Suivant_g:=new T_Cellule'(Adresse,Masque,Interf,Frequence,Date,null,null);
                        end if;
                    end if;
                else                           --sinon on parcourt l'arbre comme avec les procedures/fonctions vue precdemment
                    Bit_A_1 := (curent and POIDS_FORT) /= 0;
                    Bit_A_1_route:=((arbre.all.Adresse*(2**profondeur)) and POIDS_FORT) /= 0;
                    if Bit_A_1 then
                        curent:=curent*2;
                        profondeur:=profondeur+1;
                        auxiliaire(arbre.all.Suivant_d,Adresse,Masque,Interf,Frequence,Date, curent,profondeur);
                    else
                        profondeur:=profondeur+1;
                        curent:=curent*2;
                        auxiliaire(arbre.all.Suivant_g,Adresse,Masque,Interf,Frequence,Date, curent,profondeur);
                    end if;
                end if;
            end if;
            end auxiliaire;
   --malgres que lors de la procedure certain noeuds possede encore la route copier celle ci ne sont plus considere comme possedant la route car leur fils ne sont plus des arbres vides
   begin
      if Est_Vide(arbre) then
         curent:=0;
         profondeur:=0;
         auxiliaire(arbre,Adresse,Masque,Interf,Frequence,Date,curent,profondeur);
      else
         profondeur:=0;
         curent:=Adresse;
         auxiliaire(arbre,Adresse,Masque,Interf,Frequence,Date,curent,profondeur);
        end if;
    exception
            when Exception_adresse=>Put("erreur l'adresse est deja presente");
   end Enregistrer;

    procedure Ajouter_IP (arbre : in out T_arbre; Precision_Cache : in Integer; Taille_Max : in Integer; Adresse : in T_Adresse_IP; Interf : in Unbounded_String ;Date:in out Integer;Politique:in Unbounded_String;Defaut_cache:in out Integer) is
        Masque:T_Adresse_IP;
        Temp_Route : T_Adresse_IP := 0;
    begin
      -- Supprime l'adresse dans le cache si ce dernier est plein
      Date:=Date+1;
      Temp_Route := Adresse - (Adresse MOD UN_OCTET ** Precision_Cache);
      if not(adresse_presente(arbre ,Temp_Route,Date)) then
         if Taille(arbre) >= Taille_Max then
            if Politique=To_Unbounded_String("LRU") then
               Supprimer_LRU (arbre);
            elsif Politique=To_Unbounded_String("LFU") then
               Supprimer_LFU (arbre);
            end if;
            defaut_cache:=Defaut_cache+1;
            Masque:=Creer_Masque(Temp_Route);
            Enregistrer (arbre,Temp_Route,Masque,Interf,1,Date );
         else
            Masque:=Creer_Masque(Temp_Route);
            Enregistrer (arbre,Temp_Route,Masque,Interf,1,Date );
         end if;
      else   --si l'adresse est deja presente on enregistre pas sinon on leverait une exception dans enregistrer
         null;
      end if;

   end Ajouter_IP;


end p_routeur_la;
