clc;
clear;
close all;

%% Constantes

Taille_signal = 500000;

M = 4; % Ordre de modulation

Fp = 2000;
Fe1 = 24000; % Fréquences d'échantillonnage
Te1 = 1/Fe1; % Périodes d'échantillonnage

alpha1 = 0.35; L1 = 50; % Largeur du lobe

Rb = 3000; Rs = Rb/2; % Car M = 4
Ts = 1/Rs; Ns = round(Ts/Te1);


SNR_dB = 0:1:6; % vecteur du RSB en dB
SNR_lin = 10.^(SNR_dB/10); % vecteur du RSB linéaire

Bits = randi([0 1],1,Taille_signal); % la suite binaire

Epaisseur = 1.5; % Permet de régler l'épaisseur des courbes

%% Implantation de la transmission avec transposition de fréquence
% Mapping

ak = 2*Bits(1:2:end) - 1;
bk = 2*Bits(2:2:end) - 1;


Symboles = ak + bk*1i; 
scatterplot(Symboles); % Constellation 
title('Constellation');
xlabel('ak')
ylabel('bk');
grid on;

h = rcosdesign(alpha1, L1, Ns); % le filtre en cosinus surelevé

Surechant = kron(Symboles, [1 zeros(1,Ns -1)]);

Surechant = [Surechant zeros(1,round((length(h)-1) / 2))]; % Rallonger le signal (retard)

xe = filter(h, 1, Surechant); % génération du signal

xe = xe(round((length(h) - 1) / 2) + 1:end);

t1 = 0:Te1:(length(xe) - 1) * Te1; % Le vecteur temporel

figure; % Tracé des signaux générés sur les voies en phase et en quadrature

subplot(2,1,1);
plot(t1, real(xe)); hold on;
title("Tracé du signal xe voie en phase");
xlabel('t (s)')
ylabel('xe(t)');
grid on;
subplot(2,1,2);
plot(t1, imag(xe));
title("Tracé du signal xe voie en quadrature");
xlabel('t (s)')
ylabel('xe(t)');
grid on;
hold off;

x = real(xe .* exp(1i*2*pi*Fp.*t1));

figure;

plot(t1, x);
title('Tracé du signal x');
xlabel('t (s)')
ylabel('x(t)');
grid on;

% Traçage de  DSP estimée du signal x

DSP_estimee = pwelch(x, [], [], [], Fe1, 'twosided');
Vect_f = linspace(-Fe1/2, Fe1/2,length(DSP_estimee));


% Echelle normale

figure

plot(Vect_f,fftshift(abs(DSP_estimee)/max(DSP_estimee)));
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('DSP estimée de x (QPSK)');
grid on;


% Echelle logarithmique

figure

semilogy(Vect_f,fftshift(abs(DSP_estimee)/max(DSP_estimee)));
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('DSP estimée de x log');
grid on;


% Traçage de  DSP estimée du signal xe (en phase)

DSP_estimee_phase = pwelch(real(xe), [], [], [], Fe1, 'twosided');


% Echelle normale

figure

plot(Vect_f,fftshift(abs(DSP_estimee_phase)/max(DSP_estimee_phase)));
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('DSP estimée de xe en phase (QPSK)');
grid on;


% Echelle logarithmique

figure

semilogy(Vect_f,fftshift(abs(DSP_estimee_phase)/max(DSP_estimee_phase)));
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('DSP estimée de xe en phase log (QPSK)');
grid on;


% Traçage de  DSP estimée du signal xe (en quadrature)

DSP_estimee_quadrature = pwelch(x, [], [], [], Fe1, 'twosided');


% Echelle normale

figure

plot(Vect_f,fftshift(abs(DSP_estimee_quadrature)/max(DSP_estimee_quadrature)));
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('DSP estimée de xe en quadrature (QPSK)');
grid on;


% Echelle logarithmique

figure

semilogy(Vect_f,fftshift(abs(DSP_estimee_quadrature)/max(DSP_estimee_quadrature)));
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('DSP estimée de xe en quadrature log (QPSK)');
grid on;

%% TEB

%close all;

Px = mean(abs(x.^2));
TEB_estime_Fp = ones(1,length(SNR_dB));


for k=1:length(SNR_dB)
    
    % Canal de propagation
    sigma = sqrt(Px*Ns / (2*log2(M)*SNR_lin(k)));
    bruit = sigma*randn(1, length(x));
    x_bruite = x + bruit;

    % Retour en bande de base
    x_retour = x_bruite .* cos(2*pi*Fp*t1) - 1i * sin(2*pi*Fp*t1) .* x_bruite;

    % Démodulation bande de base
    x_retour = [x_retour zeros(1,round((length(h)-1) / 2))]; % Rallonger le signal (retard)
    z = filter(h, 1, x_retour); % Génération du signal en sortie du filtre de réception
    z = z(round((length(h) - 1) / 2) + 1:end); % Tronquer le signal (retard)

    % Échantillonnage
    echant = z(1:Ns:end);

    % Décision
    Decision = sign(real(echant)) + 1i*sign(imag(echant));

    % Demapping
    Bits_retrouve = zeros(1, 2*length(Decision));

    for j=0:length(Decision)-1
        if real(Decision(j + 1)) > 0 && imag(Decision(j + 1)) > 0
            Bits_retrouve(2*j + 1) = 1;
            Bits_retrouve(2*j + 2) = 1;
        elseif real(Decision(j + 1)) > 0 && imag(Decision(j + 1)) < 0
            Bits_retrouve(2*j + 1) = 1;
            Bits_retrouve(2*j + 2) = 0;
        elseif real(Decision(j + 1)) < 0 && imag(Decision(j + 1)) > 0
            Bits_retrouve(2*j + 1) = 0;
            Bits_retrouve(2*j + 2) = 1;
        elseif real(Decision(j + 1)) < 0 && imag(Decision(j + 1)) < 0
            Bits_retrouve(2*j + 1) = 0;
            Bits_retrouve(2*j + 2) = 0;
        end
    end
    
    % Calcul du TEB
    TEB_estime_Fp(k) = 1 - sum((Bits_retrouve == Bits)) / length(Bits);

end

% Tracé du TEB estimé
figure

semilogy(SNR_dB, TEB_estime_Fp, "DisplayName","TEB", 'LineWidth', Epaisseur)

title('Tracé du TEB en fonction de SNR');
xlabel('SNR (dB)')
ylabel('TEB');
grid on;

% Calcul du TEB théorique
TEB_th = 2*qfunc(sqrt(2*log2(M)*SNR_lin)*sin(pi/M)) / log2(M);


% Comparaison TEB estimé et TEB théorique
figure

semilogy(SNR_dB, TEB_estime_Fp, "DisplayName","TEB estimé Fréq porteuse", 'LineWidth', Epaisseur)

title('Comparaison du TEB théorique et estimé');
xlabel('SNR (dB)')
ylabel('TEB');
grid on;

hold on

semilogy(SNR_dB, TEB_th, "DisplayName","TEB théorique", 'LineWidth', Epaisseur)
grid on;

hold off

legend
