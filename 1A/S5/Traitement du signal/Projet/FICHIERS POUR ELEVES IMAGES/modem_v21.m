%% Constantes

clear;
close all;

Fe = 48000;
Te = 1/Fe;
Ts = 1/300;
Fs = 1/Ts;
Ns = round(Ts/Te);
F0 = 1180;
F1 = 980;
T0 = 1/F0;
T1 = 1/F1;
phi0 = rand*2*pi;
phi1 = rand*2*pi;
Ordre = 201;
SNR = 50;
Taille_signal = 30;

generation_signal;


%% 6 Démodulateur de fréquence adapté à la norme V21
%% 6.1 Contexte de synchronisation idéale
%% 6.1.2


integrator = ones(Taille_signal,1);
v1 = x_bruite.*cos_phi1 - x_bruite.*cos_phi0;

for i=0:Taille_signal-1
    integrator(i+1,:) = sum(v1(i*Ns+1:(i+1)*Ns,:));
end

NRZ_retrouve_v21 = integrator >= 0;

figure
subplot(2,1,1);
xlabel('t (s)')
ylabel('U (V)')
stairs(t_nrz_test, NRZ_retrouve_v21, 'LineWidth',1.5)
L_I_A = title('Signal NRZ reconstruit v21');
set(L_I_A, 'fontsize', 18)

subplot(2,1,2);
stairs(t_nrz_test, NRZ_Test, 'LineWidth',1.5)
L_I_B = title('Signal NRZ théorique');
set(L_I_B, 'fontsize', 18)


Taux_v21 = ( Taille_signal - sum(NRZ_Test == NRZ_retrouve_v21) )/Taille_signal