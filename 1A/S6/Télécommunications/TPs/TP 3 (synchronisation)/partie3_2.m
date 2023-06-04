clc;
clear;
close all;

%% Constantes

Taille_signal = 50000;

M = 2; % Ordre de modulation

Fe = 24000; % Fréquences d'échantillonnage
Te = 1/Fe; % Périodes d'échantillonnage

Rb = 6000; Rs = Rb; % Car M = 2
Ts = 1/Rs; Ns = round(Ts/Te);

SNR_dB = 0:1:6; % vecteur du RSB en dB
SNR_lin = 10.^(SNR_dB/10); % vecteur du RSB linéaire

phi_tab_deg = [0 40 100 180]
phi_tab_rad = deg2rad(phi_tab_deg);

avec_bruit = 1; % 1 si avec bruit 0 sinon
avec_dephasage = 1; % 1 si avec déphasage 0 sinon

tracer_const = 0; % 1 si on veut tracer les contellations 0 sinon

Bits = randi([0 1],1,Taille_signal); % la suite binaire

Epaisseur = 1.5; % Permet de régler l'épaisseur des courbes

%% Impact d’une erreur de phase porteuse

ak = 2*Bits - 1; % génération des symboles

Surechant = kron(ak, [1 zeros(1,Ns -1)]); % surechantillonnage

h = ones(1,Ns); % filtre de mise en forme
hr = h; % filtre de réception

% Génération du signal à transmettre
x = filter(h, 1, Surechant);
t = 0:Te:(length(x)-1)*Te;

% Canal PB eq. (AWGN) + déphasage

Px = mean(abs(x.^2)); % puissance du signal

for l = 1:length(phi_tab_rad)
    phi = phi_tab_rad(l);

    for k = 1:length(SNR_lin)
    
        SNR = SNR_lin(k);
    
        sigma = sqrt(Px * Ns / (2 * SNR)); % calcul de sigma
    
        n = sigma * randn(1, length(x)) + 1i * sigma * randn(1, length(x)); % bruit Gaussien
        x_bruite_dephase = (x + n * avec_bruit) * exp(1i*phi*avec_dephasage); % ajout du bruit
    
        % Génration du signal en sortie du filtre de réception + partie réelle
    
        z = filter(hr, 1, x_bruite_dephase);
        
        n0 = Ns; % Il suffit de modifier n0 pour observer un taux d'erreur binaire non nul
        
        % Échantillonnage
        z_echant = z(n0:Ns:end);
        
        % Estimation de l'erreur de phase
        phi_estime = 1/2 * angle(sum(z_echant.^2));
        
        % On annule le déphasage
        z_echant = z_echant .* exp(-1i * phi_estime);

        if tracer_const == 1
            scatterplot(z_echant,1,0,'gs'); % Constellation 
            title("Constellation en sortie de l'échantillonneur, \phi = " + phi_tab_deg(l) + ...
                "° et SNR = " + SNR_dB(k) + " dB");
            xlabel('ak');
            ylabel('bk');
            grid on;
        end
    
        z_echant = real(z_echant);
        
        % Décision
        for i=1:length(z_echant)
            if z_echant(i) > 0
                Decision(i) = 1;
            else
                Decision(i) = -1;
            end
        end
        
        
        % Demapping
        Bits_retrouve = (Decision == 1);
        
        TEB_estime(k) = 1 - sum((Bits_retrouve == Bits)) / length(Bits); % Taux d'erreur binaire estimé
        TEB_theorique(k) = qfunc(sqrt(2 * SNR) * cos(phi * avec_dephasage * 0)); % On retire ici le déphasage dans la formule du TEB théorique car le signal a été corrigé
        if avec_bruit == 0
            break;
        end
    end
    
    phi_estime;
    phi_estime_deg = phi_estime*(360/(2*pi))

    TEB_estime_phi(l) = TEB_estime(1);
    
    figure

    semilogy(SNR_dB, TEB_estime, "DisplayName","TEB estimé", 'LineWidth', Epaisseur)
    
    title("Tracé du TEB en fonction de SNR, \phi = " + phi_tab_deg(l) + "°");
    xlabel('SNR (dB)');
    ylabel('TEB');

    hold on;

    semilogy(SNR_dB, TEB_theorique, "DisplayName","TEB théorique", 'LineWidth', Epaisseur)

    grid on;

    hold off;

    legend;

    if avec_dephasage == 0
        break;
    end
end

phi_tab_rad;

