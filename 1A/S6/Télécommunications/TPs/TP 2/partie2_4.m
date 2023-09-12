clc;
clear;
close all;

%% Constantes

Taille_signal = 3*100000;

M = 8; % Ordre de modulation

Fp = 2000;
Fe1 = 6000; % Fréquences d'échantillonnage
Te1 = 1/Fe1; % Périodes d'échantillonnage

alpha1 = 0.2; L1 = 50; % Largeur du lobe

Rb = 3000; Rs = Rb/log2(M);
Ts = 1/Rs; Ns = round(Ts/Te1);


SNR_dB = 0:0.25:16; % vecteur du RSB en dB
SNR_lin = 10.^(SNR_dB/10); % vecteur du RSB linéaire

Bits = randi([0 1],1,Taille_signal); % la suite binaire

Epaisseur = 1.5; % Permet de régler l'épaisseur des courbes

%% Implantation de la modulation DVB-S2
% Modulation

dk = pskmod(Bits', M, pi/M, "gray", 'InputType', 'bit')';
dk = conj(dk);
%%

scatterplot(dk); % Constellation 
title('Constellation après mapping');
xlabel('ak')
ylabel('bk');
grid on;

h = rcosdesign(alpha1, L1, Ns); % le filtre en cosinus surelevé

Surechant = kron(dk, [1 zeros(1,Ns -1)]);

Surechant = [Surechant zeros(1,round((length(h)-1) / 2))]; % Rallonger le signal (retard)

xe = filter(h, 1, Surechant); % génération du signal

xe = xe(round((length(h) - 1) / 2) + 1:end);

x = xe;

t1 = 0:Te1:(length(xe) - 1) * Te1; % Le vecteur temporel

%{
figure; % Tracé des signaux générés sur les voies en phase et en quadrature

subplot(2,1,1);
plot(t1, real(xe)); hold on;
title("Tracé du signal xe voie en phase");
subplot(2,1,2);
plot(t1, imag(xe)); hold off;
title("Tracé du signal xe voie en quadrature");
xlabel('t (s)')
ylabel('xe(t)');


x = xe;

figure;

plot(t1, x);
title('Tracé du signal x');
xlabel('t (s)')
ylabel('x(t)');


% Traçage de  DSP estimée du signal x

DSP_estimee = pwelch(x, [], [], [], Fe1, 'twosided');
Vect_f = linspace(-Fe1/2, Fe1/2,length(DSP_estimee));


% Echelle normale

figure

plot(Vect_f,fftshift(abs(DSP_estimee)/max(DSP_estimee)));
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('Comparaison DSP estimée de x (QPSK)');


% Echelle logarithmique

figure

semilogy(Vect_f,fftshift(abs(DSP_estimee)/max(DSP_estimee)));
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('Comparaison DSP estimée de x log');


% Traçage de  DSP estimée du signal xe (en phase)

DSP_estimee_phase = pwelch(real(xe), [], [], [], Fe1, 'twosided');


% Echelle normale

figure

plot(Vect_f,fftshift(abs(DSP_estimee_phase)/max(DSP_estimee_phase)));
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('DSP estimée de xe en phase (QPSK)');


% Echelle logarithmique

figure

semilogy(Vect_f,fftshift(abs(DSP_estimee_phase)/max(DSP_estimee_phase)));
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('DSP estimée de xe en phase log (QPSK)');


% Traçage de  DSP estimée du signal xe (en quadrature)

DSP_estimee_quadrature = pwelch(x, [], [], [], Fe1, 'twosided');


% Echelle normale

figure

plot(Vect_f,fftshift(abs(DSP_estimee_quadrature)/max(DSP_estimee_quadrature)));
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('DSP estimée de xe en quadrature (QPSK)');


% Echelle logarithmique

figure

semilogy(Vect_f,fftshift(abs(DSP_estimee_quadrature)/max(DSP_estimee_quadrature)));
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('DSP estimée de xe en quadrature log (QPSK)');
%}

% Canal

Px = mean(abs(x.^2));

TEB_estime_tab = ones(size(SNR_dB));

for k=1:length(SNR_dB)
    
    sigma = sqrt(Px*Ns / (2*log2(M)*SNR_lin(k)));
    bruit = sigma*(randn(1, length(x)) + 1i*randn(1, length(x)));
    x_bruite = x + bruit;

    % Démodulation bande de base
    x_bruite = [x_bruite zeros(1,round((length(h)-1) / 2))]; % Rallonger le signal (retard)
    z = filter(h, 1, x_bruite); % Génération du signal en sortie du filtre de réception
    z = z(round((length(h) - 1) / 2) + 1:end); % Tronquer le signal (retard)

    % Échantillonnage
    echant = z(1:Ns:end);
    
    if mod(SNR_dB(k), 4) == 0
        scatterplot(echant); % Constellation 
        title(['Constellation après échantillonnage SNR = ',num2str(SNR_dB(k)), ' dB']);
        xlabel('ak')
        ylabel('bk');
        grid on;
    end

    Bits_retrouve = pskdemod(echant, M, pi/M, "gray", "OutputType",'bit');

    % Bits_retrouve = de2bi(Bits_retrouve', log2(M), 'left-msb');
    
    Bits_retrouve_resh = reshape(Bits_retrouve, 1, numel(Bits_retrouve));

    Bits_retrouve = Bits_retrouve_resh;
       
    % Calcul du TEB
    TEB_estime = 1 - sum((Bits_retrouve == Bits)) / length(Bits);

    TEB_estime_tab(k) = TEB_estime;

end

est_inf_a_6 = (SNR_dB <= 6);

figure

plot(SNR_dB(est_inf_a_6), TEB_estime_tab(est_inf_a_6), 'LineWidth', Epaisseur);
title('TEB estimé en fonction de SNR');
xlabel('SNR (dB)')
ylabel('TEB');
grid on;


%%
% Tracé du TEB estimé
% figure

semilogy(SNR_dB(est_inf_a_6), TEB_estime_tab(est_inf_a_6), "DisplayName","TEB estimé", ...
    'LineWidth', Epaisseur)

title('Tracé du TEB estimé en fonction de SNR');
xlabel('SNR (dB)')
ylabel('TEB');
grid on;

% Calcul du TEB théorique
TEB_th = 2*qfunc(sqrt(2*log2(M)*SNR_lin(est_inf_a_6))*sin(pi/M)) / log2(M);


% Comparaison TEB estimé et TEB théorique
figure

semilogy(SNR_dB(est_inf_a_6), TEB_estime_tab(est_inf_a_6), "DisplayName","TEB estimé", 'LineWidth', Epaisseur)

title('Comparaison du TEB théorique et estimé');
xlabel('SNR (dB)')
ylabel('TEB');
grid on;

hold on

semilogy(SNR_dB(est_inf_a_6), TEB_th, "DisplayName","TEB théorique", 'LineWidth', Epaisseur)
grid on;

hold off

legend


% Tracé de la DSP

% Traçage de  DSP estimée du signal xe (en quadrature)

DSP_estimee_8PSK = pwelch(z, [], [], [], Fe1, 'twosided');
Vect_f = linspace(-Fe1/2, Fe1/2,length(DSP_estimee_8PSK));


% Echelle normale

figure

plot(Vect_f,fftshift(abs(DSP_estimee_8PSK)/max(DSP_estimee_8PSK)));
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('DSP estimée de x (8PSK)');
grid on;

figure

semilogy(Vect_f,fftshift(abs(DSP_estimee_8PSK)/max(DSP_estimee_8PSK)));
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('DSP estimée de x (8PSK)');
grid on;