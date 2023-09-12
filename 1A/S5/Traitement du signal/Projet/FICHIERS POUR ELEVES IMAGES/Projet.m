%% Projet - Traitement du signal
%% Importation des fichiers
clear;
%load fichier1.mat; morceau1 = signal';
%load fichier2.mat; morceau2 = signal';
%load fichier3.mat; morceau3 = signal';
%load fichier4.mat; morceau4 = signal';
%load fichier5.mat; morceau5 = signal';
%load fichier6.mat; morceau6 = signal';
%clear signal;

%% Constantes


close all;

Fe = 48000;
Te = 1/Fe;
Ts = 1/300;
Fs = 1/Ts;
Ns = round(Ts/Te);
%F0 = 6000;
%F1 = 2000;
F0 = 1180;
F1 = 980;
T0 = 1/F0;
T1 = 1/F1;
phi0 = rand*2*pi;
phi1 = rand*2*pi;
theta0 = rand*2*pi;
theta1 = rand*2*pi;
Ordre = 201;
SNR = 50;
Taille_signal = 100;


%% 3.1
%% 3.1.1 Génération NRZ
%% 3.1.2 Traçage du signal


t = linspace(0,(Ts*Taille_signal),Ns*Taille_signal)';

NRZ_Test = randi([0 1],Taille_signal,1);

t_nrz_test = 0:Taille_signal-1;
t_nrz_test = t_nrz_test*Ts;

NRZ_dup = zeros(cast(Ns,"int32")*Taille_signal,1);

for i = 1:Taille_signal
    if NRZ_Test(i) == 1
        NRZ_dup(((i-1)*cast(Ns,"int32") + 1):i*cast(Ns,"int32"),1) = ones(cast(Ns,"int32"),1);
    end
end

figure
plot(t, NRZ_dup, 'LineWidth',1.5)
title('Tracé du signal NRZ');
xlabel('t(s)')
ylabel('NRZ(t)')

%% 3.1.3 Traçage de  DSP et 3.1.4 Comparaison avec DSP théorique


[DSP, Vect_f] = pwelch(NRZ_dup, [], [], [], Fe, 'twosided');

figure

semilogy(Vect_f,DSP, "DisplayName", "DSP estimée");

hold on


DSP_Th = 1/4*Ts*sinc(Vect_f*Ts).^2;
DSP_Th(1,1) = DSP_Th(1,1) + 1/4;

semilogy(Vect_f, DSP_Th, "DisplayName", "DSP théorique");
xlabel("Fréquence (Hz)")
title('Comparaison DSP théorique et expérimentale de NRZ');

hold off

legend

%% 3.2 Génération du signal modulé en fréquence
%% 3.2.1

%t = linspace(0,(Ts*Taille_signal),Ns*Taille_signal)';

cos1 = cos(2*pi*F1*t + phi1);
cos0 = cos(2*pi*F0*t + phi0);

x = (1 - NRZ_dup).*cos0 + NRZ_dup.*cos1;
% on rallonge le vecteur x afin de résoudre le problème de décalage
x = [x ; zeros((Ordre-1)/2,1)];

%% 3.2.2

t_allonge = linspace(0,(Ts*Taille_signal),Ns*Taille_signal + (Ordre-1)/2)';
figure
plot(t,x(1:length(x)-ceil((Ordre-1)/2),:))
title('Tracé du signal x(t)');

%% 3.2.4

[DSPx, Vect_fx] = pwelch(x(1:length(x)-ceil((Ordre-1)/2),:), [], [], [], Fe, 'twosided');

figure

semilogy(Vect_fx,DSPx, "DisplayName", "DSP estimée");

hold on

%Mètre la formule kalkulé par makcime !!!!!!!!!!!!!!!
DSPx_Th = 1/4*Ts*sinc(Vect_f*Ts).^2;

semilogy(Vect_fx, DSPx_Th, "DisplayName", "DSP théorique");
xlabel("Fréquence (Hz)")
title('Comparaison DSP théorique et expérimentale de x(t)');

hold off

legend

%% 4 Canal de transmission à bruit additif, blanc et Gaussien


Px = mean(abs(x).^2);
Pb = Px*10^(-SNR/10);
sigma = sqrt(Pb);

bruit = sigma*randn(1,length(x));

x_bruite = x + bruit';

%x = x(1:length(x)-ceil((Ordre-1)/2),:);

%% 5 Canal de transmission à bruit additif, blanc et Gaussien
%% 5.1 Synthèse du filtre passe-bas


F_c = (F0 + (F0+F1)/2)/2;
t_vect = linspace(-Te*(Ordre-1)/2,Te*(Ordre-1)/2, Ordre);
rep_imp_PB = 2*F_c*Te*sinc(2*F_c*t_vect);


figure
xlabel('t (s)')
ylabel('U (V)')
plot(t_vect, rep_imp_PB);
title('Réponse impulsionnelle du FPB');

%% 5.2 Synthèse du filtre passe-haut
%% 5.2.1 Voir rapport
%% 5.2.2 

rep_imp_PH = - rep_imp_PB;
rep_imp_PH((Ordre+1)/2) = 1 + rep_imp_PH((Ordre+1)/2);
figure
xlabel('t (s)')
ylabel('U (V)')
plot(t_vect, rep_imp_PH);
title('Réponse impulsionnelle du FPH')

%% 5.3 Filtrage


filtrage_PB = filter(rep_imp_PB, 1, x_bruite);
filtrage_PH = filter(rep_imp_PH, 1, x_bruite);

% On tronque le début du signal filtré pour résoudre le problème du décalage 
filtrage_PB = filtrage_PB(ceil((Ordre+1)/2):end,:);
filtrage_PH = filtrage_PH(ceil((Ordre+1)/2):end,:);

figure
subplot(3,1,1);
xlabel('t (s)')
ylabel('U (V)')
plot(t, filtrage_PB);
title('Signal filtré par FPB')

subplot(3,1,2);
plot(t, filtrage_PH, "r");
title('Signal filtré par FPH')

subplot(3,1,3);
stairs(t, NRZ_dup, 'LineWidth',1.5)
title('Signal NRZ théorique')

x_bruite = x_bruite(1:length(x)-ceil((Ordre-1)/2),:);


%% 5.4 Les tracés
%% 5.5 Détection d'énergie
%% 5.5.1 

Energies = zeros(Taille_signal,1);

for i=0:Taille_signal-1
    Energie = sum(filtrage_PB(i*Ns+1:(i+1)*Ns, 1).^2);
    Energies(i+1) = Energie;
end

K = mean(Energies);

NRZ_retrouve = Energies > K;

figure
subplot(2,1,1);
xlabel('t (s)')
ylabel('U (V)')
stairs(t_nrz_test, NRZ_retrouve, 'LineWidth',1.5)
title('Signal NRZ reconstruit')

subplot(2,1,2);
stairs(t_nrz_test, NRZ_Test, 'LineWidth',1.5)
title('Signal NRZ théorique')


%% 5.5.2

Taux = ( Taille_signal - sum(NRZ_Test == NRZ_retrouve) )/Taille_signal;

%% 5.6 Modification du démodulateur
% Voir rapport

%% 6 Démodulateur de fréquence adapté à la norme V21
%% 6.1 Contexte de synchronisation idéale
%% 6.1.2


integrator = ones(Taille_signal,1);
v1 = x_bruite.*cos1 - x_bruite.*cos0;

for i=0:Taille_signal-1
    integrator(i+1,:) = sum(v1(i*Ns+1:(i+1)*Ns,:));
end

NRZ_retrouve_v21 = integrator >= 0;

figure
subplot(2,1,1);
xlabel('t (s)')
ylabel('U (V)')
stairs(t_nrz_test, NRZ_retrouve_v21, 'LineWidth',1.5)
title('Signal NRZ reconstruit v21')

subplot(2,1,2);
stairs(t_nrz_test, NRZ_Test, 'LineWidth',1.5)
title('Signal NRZ théorique')


Taux_v21 = ( Taille_signal - sum(NRZ_Test == NRZ_retrouve_v21) )/Taille_signal;

%% 6.2 Gestion d'une erreur de synchronisation de phase porteuse

cos_theta1 = cos(2*pi*F1*t + theta1);
cos_theta0 = cos(2*pi*F0*t + theta0);
sin_theta1 = sin(2*pi*F1*t + theta1);
sin_theta0 = sin(2*pi*F0*t + theta0);

integrator1 = ones(Taille_signal,1);
integrator2 = ones(Taille_signal,1);
integrator3 = ones(Taille_signal,1);
integrator4 = ones(Taille_signal,1);

v1 = x_bruite.*cos_theta0;
v2 = x_bruite.*sin_theta0;
v3 = x_bruite.*cos_theta1;
v4 = x_bruite.*sin_theta1;

for i=0:Taille_signal-1
    integrator1(i+1,:) = sum(v1(i*Ns+1:(i+1)*Ns,:));
    integrator2(i+1,:) = sum(v2(i*Ns+1:(i+1)*Ns,:));
    integrator3(i+1,:) = sum(v3(i*Ns+1:(i+1)*Ns,:));
    integrator4(i+1,:) = sum(v4(i*Ns+1:(i+1)*Ns,:));
end

integrator1 = integrator1.^2;
integrator2 = integrator2.^2;
integrator3 = integrator3.^2;
integrator4 = integrator4.^2;

I = integrator3 + integrator4 - (integrator1 + integrator2);



NRZ_retrouve_v21_sync = I >= 0;

figure
subplot(2,1,1);
xlabel('t (s)')
ylabel('U (V)')
stairs(t_nrz_test, NRZ_retrouve_v21_sync, 'LineWidth',1.5)
title('Signal NRZ reconstruit v21 sync')

subplot(2,1,2);
stairs(t_nrz_test, NRZ_Test, 'LineWidth',1.5)
title('Signal NRZ théorique')


Taux_v21_sync = ( Taille_signal - sum(NRZ_Test == NRZ_retrouve_v21_sync) )/Taille_signal;

%% Reconstitution de l'image

