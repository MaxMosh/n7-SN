%%  Application de la SVD : compression d'images

clear all;
close all;

% Lecture de l'image
I = imread('BD_Asterix_0.png');
I = rgb2gray(I);
I = double(I);

[q, p] = size(I)

% Décomposition par SVD
fprintf('Décomposition en valeurs singulières\n')
tic
[U, S, V] = svd(I);
toc

l = min(p,q);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% On choisit de ne considérer que 200 vecteurs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% vecteur pour stocker la différence entre l'image et l'image reconstuite
inter = 1:40:(200+40);
inter(end) = 200;
differenceSVD = zeros(size(inter,2), 1);

% images reconstruites en utilisant de 1 à 200 vecteurs (avec un pas de 40)
ti = 0;
td = 0;
for k = inter

    % Calcul de l'image de rang k
    Im_k = U(:, 1:k)*S(1:k, 1:k)*V(:, 1:k)';


    % Affichage de l'image reconstruite
    ti = ti+1;
    figure(ti)
    colormap('gray')
    imagesc(Im_k)
    
    % Calcul de la différence entre les 2 images
    td = td + 1;
    differenceSVD(td) = sqrt(sum(sum((I-Im_k).^2)));
    pause
end

% Figure des différences entre image réelle et image reconstruite
ti = ti+1;
figure(ti)
hold on 
plot(inter, differenceSVD, "DisplayName", "SVD")
ylabel('RMSE')
xlabel('rank k')
legend
differenceSVD
pause



%% Plugger les différentes méthodes : eig, puissance itérée et les 4 versions de la "subspace iteration method" 

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

%%%%%%%%%%%%%
% À COMPLÉTER
%%%%%%%%%%%%%

M = I*I';

%% eig

fprintf('eig\n')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% On choisit de ne considérer que 200 vecteurs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% calcul des couples propres

tic
[W,D] = eig(M);
toc
W = fliplr(W);

% calcul des valeurs singulières

Sing = sqrt(flip(diag(abs(D))));

% vecteur pour stocker la différence entre l'image et l'image reconstuite
inter = 1:40:(200+40);
inter(end) = 200;
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
    for h = 1:k
        Im_k = Im_k + Sing(h,1) * W(:, h) * X(:, h)';
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
figure;
hold on 
plot(inter, difference, "DisplayName", "eig")
ylabel('RMSE')
xlabel('rank k')
legend
difference
pause


%% power_v11 
% Les méthodes power_v11 et power_v12 ne convergent pas en un temps
% raisonnable
%{
fprintf('power_v11\n')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% On choisit de ne considérer que 200 vecteurs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% calcul des couples propres

tic
[ V, D, n_ev, itv, flag ] = power_v11( M, search_space, percentage, eps, maxit );
toc

% calcul des valeurs singulières

Sing = sqrt(diag(abs(D)));

% vecteur pour stocker la différence entre l'image et l'image reconstuite
inter = 1:40:(200+40);
inter(end) = 200;
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
    for h = 1:k
        Im_k = Im_k + Sing(h,1) * W(:, h) * X(:, h)';
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
%hold on 
plot(inter, difference, 'rx')
ylabel('RMSE')
xlabel('rank k')
pause


%% power_v12

fprintf('power_v12\n')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% On choisit de ne considérer que 200 vecteurs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% calcul des couples propres

tic
[ V, D, n_ev, itv, flag ] = power_v12( M, search_space, percentage, eps, maxit );
toc

% calcul des valeurs singulières

Sing = sqrt(diag(abs(D)));

% vecteur pour stocker la différence entre l'image et l'image reconstuite
inter = 1:40:(min(n_ev, 200)+40);
inter(end) = min(n_ev, 200);
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
    for h = 1:k
        Im_k = Im_k + Sing(h,1) * W(:, h) * X(:, h)';
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
%hold on 
plot(inter, difference, 'rx')
ylabel('RMSE')
xlabel('rank k')
pause


%% subspace_iter_v0

fprintf('subspace_iter_v0\n')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% On choisit de ne considérer que 200 vecteurs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
[ W, D, it, flag ] = subspace_iter_v0(M, search_space, eps, maxit);
toc


% calcul des valeurs singulières

Sing = sqrt(diag(abs(D)));

% vecteur pour stocker la différence entre l'image et l'image reconstuite
inter = 1:40:(200+40);
inter(end) = 200;
difference = zeros(size(inter,2), 1);

% images reconstruites en utilisant de 1 à n_ev vecteurs (avec un pas de 40)
ti = 0;
td = 0;
for k = inter


    % calcul de l'autre ensemble de vecteurs
    
    for i=1:k
        X(:,i) = 1/Sing(i) * I' * W(:,i); % V = X et U = W
    end
    
    
    % calcul des meilleures approximations de rang faible
    
    Im_k = zeros(size(I));
    for j = 1:k
        Im_k = Im_k + Sing(j) * W(:, j) * X(:, j)';
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
figure;
hold on 
plot(inter, difference, "DisplayName", "subspace\_v0")
ylabel('RMSE')
xlabel('rank k')
pause
legend
%}

%% subspace_iter_v1

fprintf('subspace_iter_v1\n')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% On choisit de ne considérer que n_ev vecteurs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
[ W, D, n_ev, it, itv, flag ] = subspace_iter_v1(M, search_space, percentage, eps, maxit);
toc


% calcul des valeurs singulières

Sing = sqrt(diag(abs(D)));

% vecteur pour stocker la différence entre l'image et l'image reconstuite
inter = 1:40:(min(n_ev,200)+40);
inter(end) = min(n_ev,200);
difference = zeros(size(inter,2), 1);

% images reconstruites en utilisant de 1 à n_ev vecteurs (avec un pas de 40)
ti = 0;
td = 0;
for k = inter


    % calcul de l'autre ensemble de vecteurs
    
    for i=1:k
        X(:,i) = 1/Sing(i) * I' * W(:,i); % V = X et U = W
    end
    
    
    % calcul des meilleures approximations de rang faible
    
    Im_k = zeros(size(I));
    for j = 1:k
        Im_k = Im_k + Sing(j) * W(:, j) * X(:, j)';
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
figure;
hold on 
plot(inter, difference, "DisplayName", "subspace\_v1")
ylabel('RMSE')
xlabel('rank k')
legend
difference
pause



%% subspace_iter_v2

fprintf('subspace_iter_v2\n')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% On choisit de ne considérer que n_ev vecteurs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
[ W, D, n_ev, it, itv, flag ] = subspace_iter_v2(M, search_space, percentage, puiss, eps, maxit);
toc

  
% calcul des valeurs singulières

Sing = sqrt(diag(abs(D)));

% vecteur pour stocker la différence entre l'image et l'image reconstuite
inter = 1:40:(min(n_ev,200)+40);
inter(end) = min(n_ev,200);
difference = zeros(size(inter,2), 1);

% images reconstruites en utilisant de 1 à n_ev vecteurs (avec un pas de 40)
ti = 0;
td = 0;
for k = inter


    % calcul de l'autre ensemble de vecteurs
    
    for i=1:k
        X(:,i) = 1/Sing(i) * I' * W(:,i); % V = X et U = W
    end
    
    
    % calcul des meilleures approximations de rang faible
    
    Im_k = zeros(size(I));
    for j = 1:k
        Im_k = Im_k + Sing(j) * W(:, j) * X(:, j)';
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
figure;
hold on 
plot(inter, difference, "DisplayName", "subspace\_v2")
ylabel('RMSE')
xlabel('rank k')
legend
difference
pause


%% subspace_iter_v3

fprintf('subspace_iter_v3\n')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% On choisit de ne considérer que n_ev vecteurs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
[ W, D, n_ev, it, itv, flag ] = subspace_iter_v3(M, search_space, percentage, puiss, eps, maxit);
toc


% calcul des valeurs singulières

Sing = sqrt(diag(abs(D)));

% vecteur pour stocker la différence entre l'image et l'image reconstuite
inter = 1:40:(min(n_ev,200)+40);
inter(end) = min(n_ev,200);
difference = zeros(size(inter,2), 1);

% images reconstruites en utilisant de 1 à n_ev vecteurs (avec un pas de 40)
ti = 0;
td = 0;
for k = inter


    % calcul de l'autre ensemble de vecteurs
    
    for i=1:k
        X(:,i) = 1/Sing(i) * I' * W(:,i); % V = X et U = W
    end
    
    
    % calcul des meilleures approximations de rang faible
    
    Im_k = zeros(size(I));
    for j = 1:k
        Im_k = Im_k + Sing(j) * W(:, j) * X(:, j)';
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
figure;
hold on 
plot(inter, difference, "DisplayName", "subspace\_v3")
ylabel('RMSE')
xlabel('rank k')
legend
difference
pause

