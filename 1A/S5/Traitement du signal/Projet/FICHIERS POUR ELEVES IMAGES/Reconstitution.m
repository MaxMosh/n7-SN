%% Reconstitution de l'image

% Afin de construire l'image, il faut décommenter ligne 22, 79 et commenter la ligne 3 du fichier
% modem_v21_sync.m et la ligne 114 du fichier generation_signal.m

clear;
close all;

load fichier1.mat;

modem_v21_sync;
morceau1 = morceau;


load fichier2.mat;

modem_v21_sync;
morceau2 = morceau;

load fichier3.mat; 

modem_v21_sync;
morceau3 = morceau;

load fichier4.mat; 

modem_v21_sync;
morceau4 = morceau;

load fichier5.mat; 

modem_v21_sync;
morceau5 = morceau;

load fichier6.mat;

modem_v21_sync;
morceau6 = morceau;

close all;


%% Affichage de l'image

image_reconstruite = [morceau6 morceau1 morceau5; morceau2 morceau4 morceau3];

figure
image(image_reconstruite);

clear signal;

