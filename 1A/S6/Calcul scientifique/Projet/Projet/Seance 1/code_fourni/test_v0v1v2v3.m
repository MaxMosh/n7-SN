%clear all;
format long;
clc;

%%%%%%%%%%%%
% PARAMÈTRES
%%%%%%%%%%%%

% taille de la matrice symétrique
n = 400;

% type de la matrice (voir matgen_csad)
% imat == 1 valeurs propres D(i) = i
% imat == 2 valeurs propres D(i) = random(1/cond, 1) avec leur logarithmes
%                                  uniformément répartie, cond = 1e10
% imat == 3 valeurs propres D(i) = cond**(-(i-1)/(n-1)) avec cond = 1e5
% imat == 4 valeurs propres D(i) = 1 - ((i-1)/(n-1))*(1 - 1/cond) avec cond = 1e2
imat = 1;

% tolérance
eps = 1e-8;
% nombre d'itérations max pour atteindre la convergence
maxit = 10000;

% on génère la matrice (1) ou on lit dans un fichier (0)
genere = 0;

% méthode de calcul
v = 0; % subspace iteration v0

% taille du sous-espace (V1, v2, v3)
m = 60;

% puissance de la matrice A
p = 10;

% pourcentage de la trace que l'on veut atteindre (v1, v2, v3)
percentage = .25;

[W, V, flag, q, qv0] = eigen_2023(imat, n, v, m, eps, maxit, [], [], genere);
genere = 0;

fprintf('Qualité des couples propres (par rapport au critère d''arrêt) = [%0.3e , %0.3e]\n', min(qv), max(qv));
fprintf('Qualité des valeurs propres (par rapport au spectre de la matrice) = [%0.3e , %0.3e] \n', min(q), max(q));

% méthode de calcul
v = 1; % subspace iteration v1


[W, V, flag, q, qv1] = eigen_2023(imat, n, v, m, eps, maxit, percentage, [], genere);

fprintf('Qualité des couples propres (par rapport au critère d''arrêt) = [%0.3e , %0.3e]\n', min(qv), max(qv));
fprintf('Qualité des valeurs propres (par rapport au spectre de la matrice) = [%0.3e , %0.3e] \n', min(q), max(q));

% méthode de calcul
v = 2; % subspace iteration v2


[W, V, flag, q, qv2] = eigen_2023(imat, n, v, m, eps, maxit, percentage, p, genere);

fprintf('Qualité des couples propres (par rapport au critère d''arrêt) = [%0.3e , %0.3e]\n', min(qv), max(qv));
fprintf('Qualité des valeurs propres (par rapport au spectre de la matrice) = [%0.3e , %0.3e] \n', min(q), max(q));

% méthode de calcul
v = 3; % subspace iteration v3


[W, V, flag, q, qv3] = eigen_2023(imat, n, v, m, eps, maxit, percentage, p, genere);

fprintf('Qualité des couples propres (par rapport au critère d''arrêt) = [%0.3e , %0.3e]\n', min(qv), max(qv));
fprintf('Qualité des valeurs propres (par rapport au spectre de la matrice) = [%0.3e , %0.3e] \n', min(q), max(q));