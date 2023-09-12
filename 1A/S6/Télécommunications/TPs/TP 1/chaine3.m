clc;
%clear;
close all;

%% Constantes

Fe = 24000; Te = 1/Fe; 
Rb = 3000; Tb = 1/Rb;

Taille_signal = 10000;

Epaisseur = 1.5; % premet de règler l'épaisseur des courbes affichées

Bits = randi([0 1],1,Taille_signal); % la suite binaire

%% 2 Étude de modulateurs bande de base

%% Sans bruit 
Rs = Rb/2; % symboles 4-aires n = 2
Ts = 1/Rs;
Ns = round(Ts/Te);

% Génération des symboles
Symboles = reshape(Bits, 2, round(Taille_signal/2))';
Symboles = (2*bi2de(Symboles, "left-msb")-3)';

Surechant = kron(Symboles, [1 zeros(1,Ns -1)]); % surechantillonnage

h = ones(1,Ns); % filtre de mise en forme

% Génération du signal à transmettre
x = filter(h, 1, Surechant);
t = 0:Te:(length(x)-1)*Te;

figure

plot(t, x, 'LineWidth', Epaisseur)
title('Tracé du signal x');
xlabel('t (s)')
ylabel('x(t)');


hr = ones(1,Ns); % Filtre de réception

% Traçage du digramme de l'oeil

z = filter(hr, 1, x);

figure

plot(reshape(z(Ns+1:end), Ns, length(z(Ns+1:end))/Ns), 'LineWidth', Epaisseur);
title("Diagramme de l'oeil");
%ylim([-17 17])

% instant optimal 2Ts

n0 = 16; % Il suffit de modifier n0 pour observer un taux d'erreur binaire non nul

% Démodulation + demapping 

for j=0:round(length(z)/Ns)-1
    if z(n0 + Ns*j) >= 32
        Bits_retrouve(2*j+1) = 1;
        Bits_retrouve(2*j+2) = 1;
    elseif 0 <= z(n0*(j+1)) && z(n0*(j+1)) < 32
        Bits_retrouve(2*j+1) = 1;
        Bits_retrouve(2*j+2) = 0;
    elseif -32 <= z(n0*(j+1)) && z(n0*(j+1)) < 0
        Bits_retrouve(2*j+1) = 0;
        Bits_retrouve(2*j+2) = 1;
    elseif z(n0*(j+1)) < -32
        Bits_retrouve(2*j+1) = 0;
        Bits_retrouve(2*j+2) = 0;
    end
end

% Traçage des signaux émis et retrouvé

t_bits = 0:Ts:(length(Bits_retrouve)-1)*Ts; % vecteur temporel


figure

subplot(2,1,1);
stairs(t_bits, Bits); hold on;
title("Bits retrouvé");
subplot(2,1,2);
stairs(t_bits, Bits_retrouve); hold off;
title("Bits transmis");

Taux = 1 - sum((Bits_retrouve == Bits)) / length(Bits);


%% Avec bruit 
close all;

SNR_dB = 0:1:8; % vecteur du RSB en dB
SNR_lin = 10.^(SNR_dB/10); % vecteur du RSB linéaire

SNR = 10.^(15/10); % Ici on prend une valeur pour les tracés

Px = mean(abs(x.^2)); % puissance du signal
sigma = sqrt(Px * Ns / (2 * 2 * SNR)); % calcul de sigma

n = sigma * randn(1, length(x)); % bruit Gaussien
x_bruite = x + n; % ajout du bruit

% Traçage du signal x bruité

figure

plot(t, x_bruite, 'LineWidth', Epaisseur)
title('Tracé du signal x bruité');
xlabel('t (s)')
ylabel('x1(t)');

% Traçage du digramme de l'oeil

z_bruite = filter(hr, 1, x_bruite);

figure

plot(reshape(z_bruite(Ns+1:end), Ns, length(z_bruite(Ns+1:end))/Ns), 'LineWidth', Epaisseur);
title("Diagramme de l'oeil");
%ylim([-9 9])

% instant optimal = Ts

n0 = 16; % n0 = 16 est le meilleur moment pour échantiollonner

for j=0:round(length(z)/Ns)-1
    if z(n0 + Ns*j) >= 32
        Bits_retrouve_bruite(2*j+1) = 1;
        Bits_retrouve_bruite(2*j+2) = 1;
    elseif 0 <= z(n0 + Ns*j) && z(n0 + Ns*j) < 32
        Bits_retrouve_bruite(2*j+1) = 1;
        Bits_retrouve_bruite(2*j+2) = 0;
    elseif -32 <= z(n0 + Ns*j) && z(n0 + Ns*j) < 0
        Bits_retrouve_bruite(2*j+1) = 0;
        Bits_retrouve_bruite(2*j+2) = 1;
    elseif z(n0 + Ns*j) < -32
        Bits_retrouve_bruite(2*j+1) = 0;
        Bits_retrouve_bruite(2*j+2) = 0;
    end
end

% Traçage des signaux émis et retrouvé

figure

subplot(2,1,1);
stairs(t_bits, Bits); hold on;
title("Bits retrouvés");
subplot(2,1,2);
stairs(t_bits, Bits_retrouve_bruite); hold off;
title("Bits transmis");

Taux_bruite = 1 - sum((Bits_retrouve_bruite == Bits)) / length(Bits); % TEB estimé
Taux_th = qfunc(sqrt(2*SNR)); % TEB thorique

% Memes opérations pour toutes les valeurs de SNR

Taux_estime_tab3 = ones(length(SNR_lin), 1);
Taux_th_tab = zeros(length(SNR_lin), 1);
for i=0:8
    sigma = sqrt(Px * Ns / (2 * SNR_lin(i+1)));
    n = sigma * randn(1, length(x));
    x_bruite = x + n;
    z_bruite = filter(hr, 1, x_bruite);

    for j=0:round(length(z)/Ns)-1
        if z_bruite(n0 + Ns*j) >= 32
            Bits_retrouve_bruite(2*j+1) = 1;
            Bits_retrouve_bruite(2*j+2) = 1;
        elseif 0 <= z_bruite(n0 + Ns*j) && z_bruite(n0 + Ns*j) < 32
            Bits_retrouve_bruite(2*j+1) = 1;
            Bits_retrouve_bruite(2*j+2) = 0;
        elseif -32 <= z_bruite(n0 + Ns*j) && z_bruite(n0 + Ns*j) < 0
            Bits_retrouve_bruite(2*j+1) = 0;
            Bits_retrouve_bruite(2*j+2) = 1;
        elseif z_bruite(n0 + Ns*j) < -32
            Bits_retrouve_bruite(2*j+1) = 0;
            Bits_retrouve_bruite(2*j+2) = 0;
        end
    end

    Taux_estime_tab3(i+1) = 1 - sum((Bits_retrouve_bruite == Bits)) / length(Bits);
    Taux_th_tab(i+1) = 3/4*qfunc(sqrt(4/5*SNR_lin(i+1)));
end

% Comparaison des TEB théorique et estimé en fonction de SNR

figure

semilogy(SNR_dB, Taux_estime_tab3, "DisplayName","TEB estimé", 'LineWidth', Epaisseur)

title('Tracé du TEB en fonction de SNR');
xlabel('SNR (dB)')
ylabel('TEB');

hold on

semilogy(SNR_dB, Taux_th_tab, "DisplayName","TEB théorique", 'LineWidth', Epaisseur)

hold off

legend
