%% 3
%% 3.1.1 Génération NRZ


NRZ_Test = randi([0 1],Taille_signal,1);

t_nrz_test = 0:Taille_signal-1;
t_nrz_test = t_nrz_test*Ts;

NRZ_dup = zeros(cast(Ns,"int32")*Taille_signal,1);

for i = 1:Taille_signal
    if NRZ_Test(i) == 1
        NRZ_dup(((i-1)*cast(Ns,"int32") + 1):i*cast(Ns,"int32"),1) = ones(cast(Ns,"int32"),1);
    end
end

t = linspace(0,(Ts*Taille_signal),Ns*Taille_signal)';

%% 3.1.2 Traçage du signal

figure
plot(t, NRZ_dup, 'LineWidth',1.5)
L_A = title('Tracé du signal NRZ');
xlabel('t(s)')
ylabel('NRZ(t)');
set(L_A, 'fontsize', 18)


%% 3.1.3 Traçage de  DSP de NRZ et 3.1.4 Comparaison avec DSP théorique


[DSP, Vect_f] = pwelch(NRZ_dup, [], [], [], Fe, 'twosided');

figure

semilogy(Vect_f,DSP, "DisplayName", "DSP estimée");

hold on


DSP_Th = 1/4*Ts*sinc(Vect_f*Ts).^2;
DSP_Th(1,1) = DSP_Th(1,1) + 1/4;

semilogy(Vect_f, DSP_Th, "DisplayName", "DSP théorique");
xlabel("Fréquence (Hz)")
ylabel("DSP")
L_B = title('Comparaison DSP théorique et expérimentale de NRZ');
set(L_B, 'fontsize', 18)

hold off

lgd_B = legend;
lgd_B.FontSize = 18;

%% 3.2 Génération du signal modulé en fréquence
%% 3.2.1 Signal sans bruit

t = linspace(0,(Ts*Taille_signal),Ns*Taille_signal)';

cos_phi1 = cos(2*pi*F1*t + phi1);
cos_phi0 = cos(2*pi*F0*t + phi0);

x = (1 - NRZ_dup).*cos_phi0 + NRZ_dup.*cos_phi1;


%% 3.2.2 Affichage du signal x

figure
plot(t,x)
L_C = title('Tracé du signal x(t)');
xlabel("t(s)")
ylabel("x")
set(L_C, 'fontsize', 18)

%% 3.2.4 Affichage du signal x et comparaison avec DSP théorique

[DSPx, Vect_fx] = pwelch(x(1:length(x)-ceil((Ordre-1)/2),:), [], [], [], Fe, 'twosided');

figure

semilogy(Vect_fx,DSPx, "DisplayName", "DSP estimée");

hold on

ecart = Vect_fx(2) - Vect_fx(1);
decalage_F0 = zeros(round(F0/ecart),1);
decalage_F1 = zeros(round(F1/ecart),1);


DSPx_Th = 1/4*( [decalage_F0;DSP(1:length(DSP)-round(F0/ecart))] + [DSP(round(F0/ecart)+1:end,:);decalage_F0] + [decalage_F1;DSP(1:length(DSP)-round(F1/ecart))] + [DSP(round(F1/ecart)+1:end,:);decalage_F1] );

semilogy(Vect_fx, DSPx_Th, "DisplayName", "DSP théorique");
xlabel("Fréquence (Hz)");
ylabel("DSP")
L_D = title('Comparaison DSP théorique et expérimentale de x(t)');
set(L_D, 'fontsize', 18)

hold off

lgd_A = legend;
lgd_A.FontSize = 18;


%% 4 Canal de transmission à bruit additif, blanc et Gaussien

Px = mean(abs(x).^2);
Pb = Px*10^(-SNR/10);
sigma = sqrt(Pb);

bruit = sigma*randn(1,length(x));

x_bruite = x + bruit';
%x_bruite = signal'; % Décommenter pour constitution de l'image

%% Libération de mémoire

clearvars bruit Pb Px x;
