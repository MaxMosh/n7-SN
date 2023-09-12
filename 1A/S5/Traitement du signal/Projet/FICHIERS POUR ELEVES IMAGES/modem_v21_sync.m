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
theta0 = rand*2*pi;
theta1 = rand*2*pi;
Ordre = 201;
SNR = 1;
Taille_signal = 50;
%Taille_signal = round(length(signal)/Ns); % Décommenter pour constitution de l'image

generation_signal;


%% 6 Démodulateur de fréquence adapté à la norme V21
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
    integrator1(i+1,:) = sum(v1(i*Ns+1:(i+1)*Ns,:)*Te);
    integrator2(i+1,:) = sum(v2(i*Ns+1:(i+1)*Ns,:)*Te);
    integrator3(i+1,:) = sum(v3(i*Ns+1:(i+1)*Ns,:)*Te);
    integrator4(i+1,:) = sum(v4(i*Ns+1:(i+1)*Ns,:)*Te);
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
L_J_A = title('Signal NRZ reconstruit v21 sync');
set(L_J_A, 'fontsize', 18)

subplot(2,1,2);
stairs(t_nrz_test, NRZ_Test, 'LineWidth',1.5)
L_J_B = title('Signal NRZ théorique');
set(L_J_B, 'fontsize', 18)


Taux_v21_sync = ( Taille_signal - sum(NRZ_Test == NRZ_retrouve_v21_sync) )/Taille_signal;

%morceau = reconstitution_image(NRZ_retrouve_v21_sync); % Décommenter pour constitution de l'image