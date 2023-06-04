
% TP1 de Statistiques : fonctions a completer et rendre sur Moodle
% Nom : Moshfeghi
% Prénom : Maxime
% Groupe : 1SN-J

function varargout = fonctions_TP1_stat(nom_fonction,varargin)

    switch nom_fonction
        case 'tirages_aleatoires_uniformes'
            varargout{1} = tirages_aleatoires_uniformes(varargin{:});
        case 'estimation_Dyx_MV'
            [varargout{1},varargout{2}] = estimation_Dyx_MV(varargin{:});
        case 'estimation_Dyx_MC'
            [varargout{1},varargout{2}] = estimation_Dyx_MC(varargin{:});
        case 'estimation_Dorth_MV'
            [varargout{1},varargout{2}] = estimation_Dorth_MV(varargin{:});
        case 'estimation_Dorth_MC'
            [varargout{1},varargout{2}] = estimation_Dorth_MC(varargin{:});
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

% Fonction tirages_aleatoires (exercice_1.m) ------------------------------
function tirages_angles = tirages_aleatoires_uniformes(n_tirages)
    tirages_angles = pi*(rand(n_tirages,1))-pi/2;
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

% Fonction estimation_Dyx_MC (exercice_2.m) -------------------------------
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

% Fonction estimation_Dorth_MV (exercice_3.m) -----------------------------
function [theta_Dorth,rho_Dorth] = ...
         estimation_Dorth_MV(x_donnees_bruitees,y_donnees_bruitees,tirages_theta)
    [x_G, y_G, x_donnees_bruitees_centrees, y_donnees_bruitees_centrees] = centrage_des_donnees(x_donnees_bruitees,y_donnees_bruitees);
    M = (cos(tirages_theta)*x_donnees_bruitees_centrees + sin(tirages_theta)*y_donnees_bruitees_centrees).^2;
    S = sum(M,2);
    [~,ind_min_S] = min(S);
    theta_Dorth = tirages_theta(ind_min_S);
    rho_Dorth = x_G*cos(theta_Dorth) + y_G*sin(theta_Dorth);
end

% Fonction estimation_Dorth_MC (exercice_4.m) -----------------------------
function [theta_Dorth,rho_Dorth] = ...
                 estimation_Dorth_MC(x_donnees_bruitees,y_donnees_bruitees)
    [x_G, y_G, x_donnees_bruitees_centrees, y_donnees_bruitees_centrees] = centrage_des_donnees(x_donnees_bruitees,y_donnees_bruitees);
    C = [x_donnees_bruitees_centrees ; y_donnees_bruitees_centrees]';
    [vect_prop,lambda] = eig(C'*C);
    [~,ind_min_lambda]=min(sum(lambda));
    Y = vect_prop(:,ind_min_lambda);
    theta_Dorth = atan(Y(2)/Y(1));
    rho_Dorth = x_G*cos(theta_Dorth) + y_G*sin(theta_Dorth);
end
