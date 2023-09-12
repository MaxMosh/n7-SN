%% Génération des matrices
clear all;
Taille = 200;

[A1, ~, ~] = matgen_csad(1, Taille);
[A2, ~, ~] = matgen_csad(2, Taille);
[A3, ~, ~] = matgen_csad(3, Taille);
[A4, ~, ~] = matgen_csad(4, Taille);

Vp1 = eig(A1);
Vp2 = eig(A2);
Vp3 = eig(A3);
Vp4 = eig(A4);

%% Affichage
close all;



subplot(2,2,1);

histogram(Vp1, 30);
title("imat = 1");
xlabel("valeurs propres");
ylabel("Fréquence d'apparition");

subplot(2,2,2);

histogram(Vp2, 100);
title("imat = 2");
xlabel("valeurs propres");
ylabel("Fréquence d'apparition");
ylim([0 170]);
xlim([0 0.2]);

subplot(2,2,3);

histogram(Vp3, 100);
title("imat = 3");
xlabel("valeurs propres");
ylabel("Fréquence d'apparition");
ylim([0 125]);
xlim([0 0.2]);

subplot(2,2,4);

histogram(Vp4, 30);
title("imat = 4");
xlabel("valeurs propres");
ylabel("Fréquence d'apparition");
