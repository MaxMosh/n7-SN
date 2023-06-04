
% TP2 de Statistiques : fonctions a completer et rendre sur Moodle
% Nom : Moshfeghi
% Prénom : Maxime
% Groupe : 1SN-J

function varargout = fonctions_TP2_stat(nom_fonction,varargin)

    switch nom_fonction
        case 'tirages_aleatoires_uniformes'
            [varargout{1},varargout{2}] = tirages_aleatoires_uniformes(varargin{:});
        case 'estimation_Dyx_MV'
            [varargout{1},varargout{2}] = estimation_Dyx_MV(varargin{:});
        case 'estimation_Dyx_MC'
            [varargout{1},varargout{2}] = estimation_Dyx_MC(varargin{:});
        case 'estimation_Dyx_MV_2droites'
            [varargout{1},varargout{2},varargout{3},varargout{4}] = estimation_Dyx_MV_2droites(varargin{:});
        case 'probabilites_classe'
            [varargout{1},varargout{2}] = probabilites_classe(varargin{:});
        case 'classification_points'
            [varargout{1},varargout{2},varargout{3},varargout{4}] = classification_points(varargin{:});
        case 'estimation_Dyx_MCP'
            [varargout{1},varargout{2}] = estimation_Dyx_MCP(varargin{:});
        case 'iteration_estimation_Dyx_EM'
            [varargout{1},varargout{2},varargout{3},varargout{4},varargout{5},varargout{6},varargout{7},varargout{8}] = ...
            iteration_estimation_Dyx_EM(varargin{:});
    end

end

% Fonction centrage_des_donnees (exercice_1.m) ----------------------------
function [x_G, y_G, x_donnees_bruitees_centrees, y_donnees_bruitees_centrees] = ...
                centrage_des_donnees(x_donnees_bruitees,y_donnees_bruitees)
    x_G = mean(x_donnees_bruitees);
    y_G = mean(y_donnees_bruitees);
    x_donnees_bruitees_centrees = x_donnees_bruitees - x_G;
    y_donnees_bruitees_centrees = y_donnees_bruitees - y_G;
end

% Fonction tirages_aleatoires_uniformes (exercice_1.m) ------------------------
function [tirages_angles,tirages_G] = tirages_aleatoires_uniformes(n_tirages,taille)


    tirages_angles = pi*(rand(n_tirages,1))-pi/2;
    % Tirages aleatoires de points pour se trouver sur la droite (sur [-20,20])
    G = ones(n_tirages,2);
    tirages_G(:,1) = 2*taille*(rand(n_tirages,1)) - taille; % A MODIFIER DANS L'EXERCICE 2
    tirages_G(:,2) = 2*taille*(rand(n_tirages,1)) - taille;
end

% Fonction estimation_Dyx_MV (exercice_1.m) -------------------------------
function [a_Dyx,b_Dyx] = ...
           estimation_Dyx_MV(x_donnees_bruitees,y_donnees_bruitees,tirages_psi)
    [x_G, y_G, x_donnees_bruitees_centrees, y_donnees_bruitees_centrees] = centrage_des_donnees(x_donnees_bruitees,y_donnees_bruitees);
    M = (ones(size(tirages_psi,1),1)*y_donnees_bruitees_centrees - tan(tirages_psi)*x_donnees_bruitees_centrees).^2;
    S = sum(M,2);
    [~,ind_min_S] = min(S);
    a_Dyx = tan(tirages_psi(ind_min_S));
    b_Dyx = y_G - a_Dyx*x_G;
end

% Fonction estimation_Dyx_MC (exercice_1.m) -------------------------------
function [a_Dyx,b_Dyx] = ...
                   estimation_Dyx_MC(x_donnees_bruitees,y_donnees_bruitees)
    A = ones(size(x_donnees_bruitees,2),1);
    A = [x_donnees_bruitees' A];
    B = y_donnees_bruitees';
    % On pourrait écrire :
    % Aplus = inv(A'*A)*A';
    % Puis :
    % X_sol = Aplus*B;
    % On préférera écrire :
    X_sol = A\B;
    a_Dyx = X_sol(1);
    b_Dyx = X_sol(2);
end

% Fonction estimation_Dyx_MV_2droites (exercice_2.m) -----------------------------------
function [a_Dyx_1,b_Dyx_1,a_Dyx_2,b_Dyx_2] = ... 
         estimation_Dyx_MV_2droites(x_donnees_bruitees,y_donnees_bruitees,sigma, ...
                                    tirages_G_1,tirages_psi_1,tirages_G_2,tirages_psi_2)    
    M1 = ones(size(tirages_G_1,1),1)*y_donnees_bruitees - tirages_G_1(:,2)*ones(1,size(y_donnees_bruitees,2)) - tan(tirages_psi_1)*x_donnees_bruitees + (tan(tirages_psi_1).*tirages_G_1(:,1))*ones(1,size(x_donnees_bruitees,2));
    normale_M1 = (1/(sqrt(2*pi)))*exp((-1/(2*sigma^2))*M1.^2);
    M2 = ones(size(tirages_G_2,1),1)*y_donnees_bruitees - tirages_G_2(:,2)*ones(1,size(y_donnees_bruitees,2)) - tan(tirages_psi_2)*x_donnees_bruitees + (tan(tirages_psi_2).*tirages_G_2(:,1))*ones(1,size(x_donnees_bruitees,2));
    normale_M2 = (1/(sqrt(2*pi)))*exp((-1/(2*sigma^2))*M2.^2);
    S = sum(log(normale_M1 + normale_M2),2);
    [~,ind_max] = max(S);
    a_Dyx_1 = tan(tirages_psi_1(ind_max,1));
    b_Dyx_1 = tirages_G_1(ind_max,2) - a_Dyx_1*tirages_G_1(ind_max,1);
    a_Dyx_2 = tan(tirages_psi_2(ind_max,1));
    b_Dyx_2 = tirages_G_2(ind_max,2) - a_Dyx_2*tirages_G_2(ind_max,1);
end

% Fonction probabilites_classe (exercice_3.m) ------------------------------------------
function [probas_classe_1,probas_classe_2] = probabilites_classe(x_donnees_bruitees,y_donnees_bruitees,sigma,...
                                                                 a_1,b_1,proportion_1,a_2,b_2,proportion_2)
    probas_classe_1 = proportion_1*exp(-(y_donnees_bruitees - a_1*x_donnees_bruitees - b_1).^2/2*sigma^2);
    probas_classe_2 = proportion_2*exp(-(y_donnees_bruitees - a_2*x_donnees_bruitees - b_2).^2/2*sigma^2);
end

% Fonction classification_points (exercice_3.m) ----------------------------
function [x_classe_1,y_classe_1,x_classe_2,y_classe_2] = classification_points ...
              (x_donnees_bruitees,y_donnees_bruitees,probas_classe_1,probas_classe_2)
    x_classe_1 = x_donnees_bruitees(probas_classe_1 > probas_classe_2);
    x_classe_2 = x_donnees_bruitees(probas_classe_2 >= probas_classe_1);
    y_classe_1 = y_donnees_bruitees(probas_classe_1 > probas_classe_2);
    y_classe_2 = y_donnees_bruitees(probas_classe_2 >= probas_classe_1);
end

% Fonction estimation_Dyx_MCP (exercice_4.m) -------------------------------
function [a_Dyx,b_Dyx] = estimation_Dyx_MCP(x_donnees_bruitees,y_donnees_bruitees,probas_classe)
    A = [(probas_classe.*x_donnees_bruitees)' , probas_classe'];
    B = (probas_classe.*y_donnees_bruitees)';
    X = A\B;
    a_Dyx = X(1);
    b_Dyx = X(2);
end

% Fonction iteration_estimation_Dyx_EM (exercice_4.m) ---------------------
function [probas_classe_1,proportion_1,a_1,b_1,probas_classe_2,proportion_2,a_2,b_2] =...
         iteration_estimation_Dyx_EM(x_donnees_bruitees,y_donnees_bruitees,sigma,...
         proportion_1,a_1,b_1,proportion_2,a_2,b_2)
    [probas_classe_1,probas_classe_2] = probabilites_classe(x_donnees_bruitees,y_donnees_bruitees,sigma,...
                                                                 a_1,b_1,proportion_1,a_2,b_2,proportion_2);
    probas_classe_1 = probas_classe_1./(probas_classe_1+probas_classe_2);
    probas_classe_2 = probas_classe_2./(probas_classe_1+probas_classe_2);
    proportion_1 = (1/size(probas_classe_1,1))*sum(probas_classe_1);
    proportion_2 = (1/size(probas_classe_2,1))*sum(probas_classe_2);
    [a_1,b_1] = estimation_Dyx_MCP(x_donnees_bruitees,y_donnees_bruitees,probas_classe_1);
    [a_2,b_2] = estimation_Dyx_MCP(x_donnees_bruitees,y_donnees_bruitees,probas_classe_2);
end
