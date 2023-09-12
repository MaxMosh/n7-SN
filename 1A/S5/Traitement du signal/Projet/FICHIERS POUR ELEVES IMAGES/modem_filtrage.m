%% Constantes

clear;
close all;

Fe = 48000;
Te = 1/Fe;
Ts = 1/300;
Fs = 1/Ts;
Ns = round(Ts/Te);
F0 = 6000;
F1 = 2000;
T0 = 1/F0;
T1 = 1/F1;
phi0 = rand*2*pi;
phi1 = rand*2*pi;
Ordre = 61;
SNR = 0;
Taille_signal = 35;


generation_signal;

%% 5 Canal de transmission à bruit additif, blanc et Gaussien
%% 5.1 Synthèse du filtre passe-bas

F_c = (F0 + (F0+F1)/2)/2;
t_vect = linspace(-Te*(Ordre-1)/2,Te*(Ordre-1)/2, Ordre);
rep_imp_PB = 2*F_c*Te*sinc(2*F_c*t_vect);


figure

plot(t_vect, rep_imp_PB);
xlabel('t (s)')
ylabel('h')
L_E = title('Réponse impulsionnelle du FPB');
set(L_E, 'fontsize', 18)

%% 5.2 Synthèse du filtre passe-haut
%% 5.2.1 Voir rapport
%% 5.2.2 

rep_imp_PH = - rep_imp_PB;
rep_imp_PH((Ordre+1)/2) = 1 + rep_imp_PH((Ordre+1)/2);
figure
plot(t_vect, rep_imp_PH);
xlabel('t (s)')
ylabel('h')
L_F = title('Réponse impulsionnelle du FPH');
set(L_F, 'fontsize', 18)

%% 5.3 Filtrage

x_bruite = [x_bruite ; zeros((Ordre-1)/2,1)];

filtrage_PB = filter(rep_imp_PB, 1, x_bruite);
filtrage_PH = filter(rep_imp_PH, 1, x_bruite);

% On tronque le début du signal filtré pour résoudre le problème du décalage 
filtrage_PB = filtrage_PB(ceil((Ordre+1)/2):end,:);
filtrage_PH = filtrage_PH(ceil((Ordre+1)/2):end,:);


figure
subplot(3,1,1);
plot(t, filtrage_PB);
xlabel('t (s)')
L_G_A = title('Signal filtré par FPB');
set(L_G_A, 'fontsize', 18)

subplot(3,1,2);
plot(t, filtrage_PH, "r");
xlabel('t (s)')
L_G_B = title('Signal filtré par FPH');
set(L_G_B, 'fontsize', 18)

subplot(3,1,3);
stairs(t, NRZ_dup, 'LineWidth',1.5)
xlabel('t (s)')
L_G_C = title('Signal NRZ théorique');
set(L_G_C, 'fontsize', 18)

x_bruite = x_bruite(1:length(x_bruite)-ceil((Ordre-1)/2),:);


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
stairs(t_nrz_test, NRZ_retrouve, 'LineWidth',1.5)
xlabel('t (s)')
ylabel('U (V)')
L_H_A = title('Signal NRZ reconstruit');
set(L_H_A, 'fontsize', 18)

subplot(2,1,2);
stairs(t_nrz_test, NRZ_Test, 'LineWidth',1.5)
xlabel('t (s)')
ylabel('U (V)')
L_H_B = title('Signal NRZ théorique');
set(L_H_B, 'fontsize', 18)


%% 5.5.2

Taux = ( Taille_signal - sum(NRZ_Test == NRZ_retrouve) )/Taille_signal