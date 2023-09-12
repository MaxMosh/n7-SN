clear;

Taux_vect = zeros(50,1);
SNR_vect = linspace(-50,-10,35)';
SNR_vect = [SNR_vect;linspace(-9.9,5,15)'];


for j=1:50
    SNR = SNR_vect(j,1);
    modem_filtrage;
    Taux_vect(j,1) = Taux;
    close all;
end

figure
plot(SNR_vect,Taux_vect,'LineWidth',1.5)
xlabel('SNR')
ylabel("Taux d'erreur")
L_F = title("Le taux d'erreur en fonction du bruit");
set(L_F, 'fontsize', 18)