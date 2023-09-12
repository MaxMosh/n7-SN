%%  Application de la SVD : compression d'images

clear all
close all

% Lecture de l'image
I = imread('BD_Asterix_0.png');
I = rgb2gray(I);
I = double(I);

[q, p] = size(I)

% Décomposition par SVD
fprintf('subspace_iter_v1\n')

l = min(p,q);

% Plugger les différentes méthodes : eig, puissance itérée et les 4 versions de la "subspace iteration method" 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% QUELQUES VALEURS PAR DÉFAUT DE PARAMÈTRES, 
% VALEURS QUE VOUS POUVEZ/DEVEZ FAIRE ÉVOLUER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% tolérance
eps = 1e-8;
% nombre d'itérations max pour atteindre la convergence
maxit = 10000;

% taille de l'espace de recherche (m)
search_space = 400;

% pourcentage que l'on se fixe
percentage = 0.995;

% p pour les versions 2 et 3 (attention p déjà utilisé comme taille)
puiss = 1;


%% subspace_iter_v1
M = I*I';
tic
[ W, D, n_ev, it, itv, flag ] = subspace_iter_v1(M, search_space, percentage, eps, maxit);
toc

% calcul des valeurs singulières

Sing = sqrt(diag(abs(D)));






%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% On choisit de ne considérer que 100 vecteurs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% vecteur pour stocker la différence entre l'image et l'image reconstuite
inter = 1:20:(100+20);
inter(end) = 100;
difference = zeros(size(inter,2), 1);

% images reconstruites en utilisant de 1 à 200 vecteurs (avec un pas de 40)
ti = 0;
td = 0;
for k = inter
    

    % calcul de l'autre ensemble de vecteurs

    for i=1:k
        X(:,i) = 1/Sing(i) * I' * W(:,i); % Vi = X et Ui = W
    end


    % calcul des meilleures approximations de rang faible
    
    Im_k = zeros(size(I));
    for j = 1:k-1
        Im_k = Im_k + Sing(j,1) * W(:, j) * X(:, j)';
    end


    % Affichage de l'image reconstruite
    ti = ti+1;
    figure(ti)
    colormap('gray')
    imagesc(Im_k)
    
    % Calcul de la différence entre les 2 images
    td = td + 1;
    difference(td) = sqrt(sum(sum((I-Im_k).^2)));
    pause
end

% Figure des différences entre image réelle et image reconstruite
ti = ti+1;
figure(ti)
hold on 
plot(inter, difference, 'rx')
ylabel('RMSE')
xlabel('rank k')
pause





