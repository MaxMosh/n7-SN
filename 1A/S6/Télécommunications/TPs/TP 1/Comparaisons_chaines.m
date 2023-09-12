clear;
clc;
close all;

%% ATTENTION !
% Il faut commenter les clear au début des fichiers chainei, i ∈ {1, 2, 3}

%% Comparaison

chaine1;
chaine2;
chaine3; close all;

%% Comparaison du TEB obtenu avec les chaines 1 et 2

figure

semilogy(SNR_dB, Taux_estime_tab1, "DisplayName","TEB1", 'LineWidth', Epaisseur)

title('Comparaison TEB chaînes 1 et 2');
xlabel('SNR (dB)')
ylabel('TEB');

hold on

semilogy(SNR_dB, Taux_estime_tab2, "DisplayName","TEB2", 'LineWidth', Epaisseur)

hold off

legend

% Comparaison du TEB obtenu avec les chaines 1 et 2

figure

semilogy(SNR_dB, Taux_estime_tab1, "DisplayName","TEB1", 'LineWidth', Epaisseur)

title('Comparaison TEB chaînes 1 et 3');
xlabel('SNR (dB)')
ylabel('TEB');

hold on

semilogy(SNR_dB, Taux_estime_tab3, "DisplayName","TEB3", 'LineWidth', Epaisseur)

hold off

legend