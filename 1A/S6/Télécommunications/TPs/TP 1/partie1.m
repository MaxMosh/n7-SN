clc;
clear;
close all;

%% Constantes

Fe = 24000; Te = 1/Fe; 
Rb = 3000; Tb = 1/Rb;

Epaisseur = 1.5; % premet de règler l'épaisseur des courbes affichées

Taille_signal = 1000;

Bits = randi([0 1],1,Taille_signal); % la suite binaire

%% 2 Étude de modulateurs bande de base

%% Modulateur 1

Rs1 = Rb; % symboles binaires n = 1
Ts1 = 1/Rs1;
Ns1 = round(Ts1/Te);

Symboles1 = 2*Bits - 1;

Surechant1 = kron(Symboles1, [1 zeros(1,Ns1 -1)]);

h1 = ones(1,Ns1); % filtre de mise en forme

x1 = filter(h1, 1, Surechant1); % génération du signal
t1 = 0:Te:(length(x1)-1)*Te; % vecteur temporel

% Traçage du signal x en sortie du filtre d'émission 

figure

plot(t1, x1, 'LineWidth', Epaisseur)
title('Tracé du signal x1');
xlabel('t (s)')
ylabel('x1(t)');

% Traçage de  DSP estimée et Comparaison avec DSP théorique

DSP1_estimee = pwelch(x1, [], [], [], Fe, 'twosided');
Vect_f1 = linspace(-Fe/2, Fe/2,length(DSP1_estimee));

DSP_Th1 = Ts1*sinc(Ts1*Vect_f1).^2; % DSP théorique

% Echelle normale

figure

plot(Vect_f1,fftshift(abs(DSP1_estimee)/max(DSP1_estimee)), "DisplayName","DSP1 estimée", 'LineWidth', Epaisseur);
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('Comparaison DSP théorique et estimée (modulateur 1)');

hold on

plot(Vect_f1,DSP_Th1/max(DSP_Th1), "DisplayName","DSP1 théorique", 'LineWidth', Epaisseur);

legend

hold off

% Echelle logarithmique

figure

semilogy(Vect_f1,fftshift(abs(DSP1_estimee)/max(DSP1_estimee)), "DisplayName", "DSP1 estimée", 'LineWidth', Epaisseur);
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('Comparaison DSP théorique et estimée log (modulateur 1)');


hold on


semilogy(Vect_f1, DSP_Th1/max(DSP_Th1), "DisplayName", "DSP1 théorique", 'LineWidth', Epaisseur);

legend

hold off

%% Modulateur 2
close all;

Rs2 = Rb/2; % symboles 4-aires n = 2
Ts2 = 1/Rs2;
Ns2 = round(Ts2/Te);

% génération des symboles
Symboles2 = reshape(Bits, 2, round(Taille_signal/2))';
Symboles2 = (2*bi2de(Symboles2, "left-msb")-3)'; 

Surechant2 = kron(Symboles2, [1 zeros(1,Ns2 -1)]); % Le suréchantillonnage

h2 = ones(1,Ns2); % filtre de mise en forme

x2 = filter(h2, 1, Surechant2); % signal en sortie du filtre d'émission
t2 = 0:Te:(length(x2)-1)*Te;

% Traçage du signal x

figure

plot(t2, x2, 'LineWidth', Epaisseur)
title('Tracé du signal x2');
xlabel('t (s)')
ylabel('x2(t)');

% Traçage de  DSP estimée et Comparaison avec DSP théorique

DSP2_estimee = pwelch(x2, [], [], [], Fe, 'twosided');
Vect_f2 = linspace(-Fe/2, Fe/2,length(DSP2_estimee));


DSP_Th2 = 5*Ts2*sinc(Ts2*Vect_f2).^2;

% Echelle normale

figure

plot(Vect_f2,fftshift(abs(DSP2_estimee)/max(DSP2_estimee)), "DisplayName","DSP2 estimée", 'LineWidth', Epaisseur);
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('Comparaison DSP théorique et estimée (modulateur 2)');

hold on

plot(Vect_f2,DSP_Th2/max(DSP_Th2), "DisplayName","DSP2 théorique", 'LineWidth', Epaisseur);

legend

hold off

% Echelle logarithmique

figure

semilogy(Vect_f2,fftshift(abs(DSP2_estimee)/max(DSP2_estimee)), "DisplayName", "DSP2 estimée", 'LineWidth', Epaisseur);


hold on



semilogy(Vect_f2, DSP_Th2/max(DSP_Th2), "DisplayName", "DSP2 théorique", 'LineWidth', Epaisseur);
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('Comparaison DSP théorique et estimée log (modulateur 2)');

legend

hold off

%% Modulateur 3 

Rs3 = Rb; % symboles binaires n = 1
Ts3 = 1/Rs3;
Ns3 = round(Ts3/Te);
alpha = 0.3; % roll-off
L = 50; % Largeur du lobe
var3 = 1/(2*Ts3)*(1/3-2/pi^2); % sigma²


Symboles3 = 2*Bits - 1; % Symboles générés

Surechant3 = kron(Symboles3, [1 zeros(1,Ns3 -1)]);

h3 = rcosdesign(alpha, L, Ns3); % le filtre en cosinus surelevé

x3 = filter(h3, 1, Surechant3); % génération du signal
t3 = 0:Te:(length(x3)-1)*Te;

% Traçage du signal x

figure

plot(t3, x3, 'LineWidth', Epaisseur)
title('Tracé du signal x3');
xlabel('t (s)')
ylabel('x3(t)');

% Traçage de  DSP estimée et Comparaison avec DSP théorique

DSP3_estimee = pwelch(x3, [], [], [], Fe, 'twosided');
Vect_f3 = linspace(-Fe/2, Fe/2,length(DSP3_estimee));

% Génération de la DSP théorique
pas = Fe / length(DSP3_estimee);
borne1 = -(1+alpha)/(2*Ts3);
borne2 = (alpha-1)/(2*Ts3);
borne3 = (1-alpha)/(2*Ts3);
borne4 = (1+alpha)/(2*Ts3);


indice1 = round( (borne1+Fe/2)/pas );
indice2 = round( (borne2+Fe/2)/pas );
indice3 = round( (borne3+Fe/2)/pas );
indice4 = round( (borne4+Fe/2)/pas );

DSP_Th3 = zeros(1, length(DSP3_estimee));
DSP_Th3(1,indice1:indice2-1) = var3/2*(1+cos(pi*Ts3/alpha*(abs(Vect_f3(1, indice1:indice2-1)) - borne3)));
DSP_Th3(1,indice2:indice3-1) = var3;
DSP_Th3(1,indice3:indice4-1) = var3/2*(1+cos(pi*Ts3/alpha*(abs(Vect_f3(1, indice3:indice4-1)) - borne3)));

% Echelle normale

figure

plot(Vect_f3,fftshift(abs(DSP3_estimee)/max(DSP3_estimee)), "DisplayName","DSP3 estimée", 'LineWidth', Epaisseur);
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('Comparaison DSP théorique et estimée (modulateur 3)');

hold on

plot(Vect_f3,DSP_Th3/max(DSP_Th3), "DisplayName","DSP3 théorique", 'LineWidth', Epaisseur);

legend

hold off

% Echelle logarithmique

figure

semilogy(Vect_f3,fftshift(abs(DSP3_estimee)/max(DSP3_estimee)), "DisplayName", "DSP3 estimée", 'LineWidth', Epaisseur);
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('Comparaison DSP théorique et estimée log (modulateur 3)');


hold on


semilogy(Vect_f3, DSP_Th3/max(DSP_Th3), "DisplayName", "DSP3 théorique", 'LineWidth', Epaisseur);

legend

hold off

%% Comparaison des DSP des 3 modulateurs
close all;

figure

semilogy(Vect_f1,fftshift(abs(DSP1_estimee)/max(DSP1_estimee)), "DisplayName", "DSP1 estimée", 'LineWidth', Epaisseur);
xlabel("Fréquence (Hz)")
ylabel("DSP")
title('Comparaison des DSP estimées (modulateur 1, 2 et 3)');

hold on

semilogy(Vect_f2,fftshift(abs(DSP2_estimee)/max(DSP2_estimee)), "DisplayName", "DSP2 estimée", 'LineWidth', Epaisseur);
xlabel("Fréquence (Hz)")
ylabel("DSP")

hold on

semilogy(Vect_f3,fftshift(abs(DSP3_estimee)/max(DSP3_estimee)), "DisplayName", "DSP3 estimée", 'LineWidth', Epaisseur);
xlabel("Fréquence (Hz)")
ylabel("DSP")

legend

hold off

%% 3. Étude des interférences entre symboles et du critère de Nyquist

%% 3.1 Étude sans canal de propagation

close all;

z = filter(h1, 1, x1); % signal en sortie du filtre de réception

% Traçage du signal z

figure

plot(t1, z, 'LineWidth', Epaisseur)
title('Tracé du signal z');
xlabel('t (s)')
ylabel('z(t)');
ylim([-9 9])

figure

g = conv(h1,h1);
plot(g, 'LineWidth', Epaisseur);
ylim([-0.5 9])
title('Réponse impulsionnelle globale sans canal');
ylabel('g');


figure

plot(reshape(z(Ns1+1:end), Ns1, length(z(Ns1+1:end))/Ns1), 'LineWidth', Epaisseur);
title("Diagramme de l'oeil");
ylim([-9 9])

n0 = 8; % Il suffit de modifier n0 pour observer un taux d'erreur binaire non nul

for i=0:round(length(z)/Ns1)-1
    if z(n0*(i+1)) > 0
        Bits_retrouve(i+1) = 1;
    else
        Bits_retrouve(i+1) = 0;
    end
end


t = 0:Ts1:(length(Bits_retrouve)-1)*Ts1;

figure

stairs(t, Bits_retrouve);
title("Bits retrouvé");

figure

subplot(2,1,1);
stairs(t, Bits); hold on;
title("Bits retrouvé");
subplot(2,1,2);
stairs(t, Bits_retrouve); hold off;
title("Bits transmis");

Taux1 = 1 - sum((Bits_retrouve == Bits)) / length(Bits);


%% 3.2 Étude avec canal de propagation sans bruit
close all;

BW1 = 1250; BW8 = 8000;
Ordre = 61; % Ordre du filtre

k = linspace(-(Ordre-1)/2,(Ordre-1)/2, Ordre);

FPB8 = 2*BW8/Fe*sinc(2*BW8/Fe*k); % filtre passe-bas avec BW = 8000 Hz
FPB1 = 2*BW1/Fe*sinc(2*BW1/Fe*k); % filtre passe-bas avec BW = 1000 Hz
gc8 = conv(g,FPB8); % réponse impulsionnelle globale avec canal BW = 8000 Hz
gc1 = conv(g,FPB1); % réponse impulsionnelle globale avec canal BW = 1000 Hz

% Traçage des réponses impulsionnelles

figure

plot(gc8, 'LineWidth', Epaisseur);
title('Réponse impulsionnelle globale avec canal BW = 8000 Hz');
ylabel('gc');

figure

plot(gc1, 'LineWidth', Epaisseur);
title('Réponse impulsionnelle globale avec canal BW = 1000 Hz');
ylabel('gc');

grc8 = conv(h1,FPB8); % hr convolué à hc BW = 8000 Hz
grc1 = conv(h1,FPB1); % hr convolué à hc BW = 1000 Hz

x1_corrige = [x1 zeros((Ordre-1)/2,1)']; % Correction du retard introduit par le filtre

% Traçage du signal z en sortie du filtre de réception

% BW = 8000 Hz

zc8 = filter(grc8, 1, x1_corrige); % signal z en sortie du filtre de réception

zc8 = zc8(:, ceil((Ordre+1)/2):end); % Correction du retard introduit par le filtre
 
figure

plot(t1, zc8, 'LineWidth', Epaisseur)
title('Tracé du signal zc BW = 8000 Hz');
xlabel('t (s)')
ylabel('zc(t)');
ylim([-9 9])

% BW = 1000 Hz


zc1 = filter(grc1, 1, x1_corrige); % signal z en sortie de du filtre de réception

zc1 = zc1(:, ceil((Ordre+1)/2):end); % Correction du retard introduit par le filtre


figure

plot(t1, zc1, 'LineWidth', Epaisseur)
title('Tracé du signal zc BW = 1000 Hz');
xlabel('t (s)')
ylabel('zc(t)');
ylim([-9 9])

% Traçage du digramme de l'oeil

% BW = 8000 Hz

figure

plot(reshape(zc8(Ns1+1:end), Ns1, length(zc8(Ns1+1:end))/Ns1), 'LineWidth', Epaisseur);
title("Diagramme de l'oeil BW = 8000 Hz");
ylim([-9 9])

% BW = 1000 Hz

figure

plot(reshape(zc1(Ns1+1:end), Ns1, length(zc1(Ns1+1:end))/Ns1), 'LineWidth', Epaisseur);
title("Diagramme de l'oeil BW = 1000 Hz");
ylim([-9 9])

% Génération des vecteurs à tracer

HHr = fftshift(abs(fft(conv(h1,h1), 2048))); % H = Hr
Hc1 = fftshift(abs(fft(FPB1, 2048)));
Hc8 = fftshift(abs(fft(FPB8, 2048)));
vect_f = linspace(-Fe/2, Fe/2, length(HHr)); % vecteur de fréquence

% Traçage des courbes demandées

% BW = 8000 Hz

figure

plot(vect_f, HHr/max(HHr), "DisplayName", "|H.Hr|", 'LineWidth', Epaisseur); % |H.Hr|
title("Comparaison HHr et Hc BW = 8000 Hz");
xlabel('f(Hz)')

hold on

plot(vect_f, Hc8/max(Hc8), "DisplayName", "|Hc|", 'LineWidth', Epaisseur) % |Hc|

legend
hold off

% BW = 1000 Hz

figure

plot(vect_f, HHr/max(HHr), "DisplayName", "|H.Hr|", 'LineWidth', Epaisseur); % |H.Hr|

hold on

plot(vect_f, Hc1/max(Hc1), "DisplayName", "|Hc|", 'LineWidth', Epaisseur) % |Hc|
title("Comparaison HHr et Hc BW = 1000 Hz");
xlabel('f(Hz)')

legend
hold off

n0 = 8; % Il suffit de modifier n0 pour observer un taux d'erreur binaire non nul

for i=0:round(length(zc8)/Ns1)-1
    if zc8(n0*(i+1)) > 0
        Bits_retrouve8000(i+1) = 1;
    else
        Bits_retrouve8000(i+1) = 0;
    end
end

for i=0:round(length(zc1)/Ns1)-1
    if zc1(n0*(i+1)) > 0
        Bits_retrouve1000(i+1) = 1;
    else
        Bits_retrouve1000(i+1) = 0;
    end
end


Taux8000 = 1 - sum((Bits_retrouve8000 == Bits)) / length(Bits);

Taux1000 = 1 - sum((Bits_retrouve1000 == Bits)) / length(Bits);


%% 4. Étude de l impact du bruit et du filtrage adapté, notion d’efficacité en puissance

% Voir les fichiers chainei.m, i ∈ {1, 2, 3}

