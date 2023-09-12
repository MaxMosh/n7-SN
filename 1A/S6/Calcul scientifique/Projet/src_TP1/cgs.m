%--------------------------------------------------------------------------
% ENSEEIHT - 1SN - Calcul scientifique
% TP1 - Orthogonalisation de Gram-Schmidt
% cgs.m
%--------------------------------------------------------------------------

function Q = cgs(A)

    % Recuperation du nombre de colonnes de A
    [~, m] = size(A);
    
    % Initialisation de la matrice Q avec la matrice A
    Q = A;
    
    %------------------------------------------------
    % A remplir
    % Algorithme de Gram-Schmidt classique
    %------------------------------------------------

    Q(:,1) = A(:,1)/norm(A(:,1));
    for k=2:m
        tmp = 0;
        for i=1:k-1
            terme = sum(A(:,k).*Q(:,i))/(norm(Q(:,i))^2)*Q(:,i);
            tmp = tmp + terme;
        end
        Q(:,k) = A(:,k)-tmp;
        Q(:,k) = Q(:,k)/norm(Q(:,k));
    end

end