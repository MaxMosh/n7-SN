%--------------------------------------------------------------------------
% ENSEEIHT - 1SN - Calcul scientifique
% TP1 - Orthogonalisation de Gram-Schmidt
% mgs.m
%--------------------------------------------------------------------------

function Q = mgs(A)

    % Recuperation du nombre de colonnes de A
    [~, m] = size(A);
    
    % Initialisation de la matrice Q avec la matrice A
    Q = A;
    
    %------------------------------------------------
    % A remplir
    % Algorithme de Gram-Schmidt modifie
    %------------------------------------------------
    Q(:,1) = A(:,1)/norm(A(:,1));
    for k=2:m
        for i=1:k-1
            Q(:,k) = Q(:,k) - sum(Q(:,k).*Q(:,i))/(norm(Q(:,i))^2)*Q(:,i);
        end
        Q(:,k) = Q(:,k)/norm(Q(:,k));
    end

end