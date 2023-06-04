
% TP3 de Statistiques : fonctions a completer et rendre sur Moodle
% Nom : Moshfeghi
% Prenom : Maxime
% Groupe : 1SN-J

function varargout = fonctions_TP3_stat(nom_fonction,varargin)

    switch nom_fonction
        case 'estimation_F'
            [varargout{1},varargout{2},varargout{3}] = estimation_F(varargin{:});
        case 'choix_indices_points'
            [varargout{1}] = choix_indices_points(varargin{:});
        case 'RANSAC_2'
            [varargout{1},varargout{2}] = RANSAC_2(varargin{:});
        case 'G_et_R_moyen'
            [varargout{1},varargout{2},varargout{3}] = G_et_R_moyen(varargin{:});
        case 'estimation_C_et_R'
            [varargout{1},varargout{2},varargout{3}] = estimation_C_et_R(varargin{:});
        case 'RANSAC_3'
            [varargout{1},varargout{2}] = RANSAC_3(varargin{:});
    end

end

% Fonction estimation_F (exercice_1.m) ------------------------------------
function [rho_F,theta_F,ecart_moyen] = estimation_F(rho,theta)
A = [cos(theta) sin(theta)];
X = A\rho;
rho_F = sqrt(X(1)^2 + X(2)^2);
theta_F = atan2(X(2),X(1));
ec_1 = rho - rho_F*(cos(theta - theta_F));
ec_2 = abs(ec_1);
ecart_moyen = (1/size(rho,1))*sum(ec_2,1);
end

% Fonction choix_indice_elements (exercice_2.m) ---------------------------
function tableau_indices_points_choisis = choix_indices_points(k_max,n,n_indices)
tableau_indices_points_choisis = zeros(k_max,n_indices);
for i = 1:k_max
    tableau_indices_points_choisis(i,:) = randperm(n,n_indices);
end
end

% Fonction RANSAC_2 (exercice_2.m) ----------------------------------------
function [rho_F_estime,theta_F_estime] = RANSAC_2(rho,theta,parametres,tableau_indices_2droites_choisies)
EM_etoile = Inf;
for k = 1:parametres(3)
    ind_tires = tableau_indices_2droites_choisies(k,:);
    rho_2_droites = rho(ind_tires);
    theta_2droites = theta(ind_tires);
    [rho_F_temp , theta_F_temp , ~] = estimation_F(rho_2_droites,theta_2droites);
    vect_distances = abs(rho - rho_F_temp*cos(theta - theta_F_temp));
    vect_bool = vect_distances < parametres(1);
    vect_selection = vect_distances(vect_bool);
    if (size(vect_selection,1)/size(rho,1)) > parametres(2)
        rho_proches = rho(vect_bool);
        theta_proches = theta(vect_bool);
        [rho_temp2 , theta_temp2 , EM_temp] = estimation_F(rho_proches,theta_proches);
        if EM_temp < EM_etoile
            EM_etoile = EM_temp;
            rho_F_estime = rho_temp2;
            theta_F_estime = theta_temp2;
        end
    end
end
end

% Fonction G_et_R_moyen (exercice_3.m, bonus, fonction du TP1) ------------
function [G, R_moyen, distances] = ...
         G_et_R_moyen(x_donnees_bruitees,y_donnees_bruitees)



end

% Fonction tirages_aleatoires (exercice_3.m, bonus, fonction du TP1) ----------------
function [tirages_C,tirages_R] = tirages_aleatoires_uniformes(n_tirages,G,R_moyen)
    


end

% Fonction estimation_C_et_R (exercice_3.m, bonus, fonction du TP1) -------
function [C_estime, R_estime, critere] = ...
         estimation_C_et_R(x_donnees_bruitees,y_donnees_bruitees,tirages_C,tirages_R)



end

% Fonction RANSAC_3 (exercice_3, bonus) -----------------------------------
function [C_estime,R_estime] = ...
         RANSAC_3(x_donnees_bruitees,y_donnees_bruitees,parametres,tableau_indices_3points_choisis)
     


end
