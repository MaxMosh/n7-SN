% Ce programme est le script principal permettant d'illustrer
% un algorithme de reconnaissance de chiffres.

% Nettoyage de l'espace de travail
clear all; close all;

% Repertories contenant les donnees et leurs lectures
addpath('Data');
addpath('Utils')

rng('shuffle')


% Bruit
sig0=0.2;

%tableau des csores de classification
% intialisation aléatoire pour affichage
r=rand(6,5);
r2=rand(6,5);

for k=1:5
% Definition des donnees
file=['D' num2str(k)]

% Recuperation des donnees
disp('Generation de la base de donnees');
sD=load(file);
D=sD.(file);
%

% Bruitage des données
Db= D+sig0*randn(size(D));


%%%%%%%%%%%%%%%%%%%%%%%
% Analyse des donnees 
%%%%%%%%%%%%%%%%%%%%%%%
disp('PCA : calcul du sous-espace');
%%%%%%%%%%%%%%%%%%%%%%%%% TO DO %%%%%%%%%%%%%%%%%%
% disp('TO DO')
D_moy = mean(Db,2);
D_c = Db - D_moy;
n_Db = size(Db,2);
D_cov = (1/n_Db)*D_c*D_c';

[U,d] = svd(D_cov);

prec_approx = 0.95;
prec = 0;
nb_comp = 1;
while prec < prec_approx
    prec = 1 - sqrt(d(nb_comp + 1,nb_comp + 1)/d(1,1));
    nb_comp = nb_comp + 1;
end

U_tilde = U(1:end,1:nb_comp);

%%%%%%%%%%%%%%%%%%%%%%%%% FIN TO DO %%%%%%%%%%%%%%%%%%

disp('kernel PCA : calcul du sous-espace');
%%%%%%%%%%%%%%%%%%%%%%%%% TO DO %%%%%%%%%%%%%%%%%%
% disp('TO DO')

signoyau = 0.5
K = exp(-(dist(Db))/2*signoyau)
[U2,d2] = svd(K);

alpha = (1./sqrt(sum(d2,1))).*U2

y = sum(alpha.*K,2)

prec_approx2 = 0.90;
prec2 = 0;
nb_comp2 = 1;
while prec2 < prec_approx2 && nb_comp2 < size(d2,1)
    prec2 = 1 - sqrt(d2(nb_comp2 + 1,nb_comp2 + 1)/d2(1,1));
    nb_comp2 = nb_comp2 + 1;
end

first_alpha = alpha(1:end,1:nb_comp2);
first_y = y(1:nb_comp2);

gamma = sum(first_y.*first_alpha,2)

z0 = ones(size(d2,1),1)
zt = z0

cond_arret = 1e-5
diff = 1000

while diff > cond_arret
    ztplus1 = sum(gamma.*exp((-norm(zt - Db'))/(2*signoyau)).*Db',1)/sum(gamma.*exp((-norm(zt - Db'))/(2*signoyau)),1);
    diff = ztplus1 - zt
    zt = ztplus1
end

%%%%%%%%%%%%%%%%%%%%%%%%% FIN TO DO %%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reconnaissance de chiffres
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 % Lecture des chiffres à reconnaitre
 disp('test des chiffres :');
 tes(:,1) = importerIm('test1.jpg',1,1,16,16);
 tes(:,2) = importerIm('test2.jpg',1,1,16,16);
 tes(:,3) = importerIm('test3.jpg',1,1,16,16);
 tes(:,4) = importerIm('test4.jpg',1,1,16,16);
 tes(:,5) = importerIm('test5.jpg',1,1,16,16);
 tes(:,6) = importerIm('test9.jpg',1,1,16,16);


 for tests=1:6
    % Bruitage
    tes(:,tests)=tes(:,tests)+sig0*randn(length(tes(:,tests)),1);
    
    % Classification depuis ACP
     %%%%%%%%%%%%%%%%%%%%%%%%% TO DO %%%%%%%%%%%%%%%%%%
     disp('PCA : classification');
     % disp('TO DO')
     x_barre = tes(:,tests) - D_moy
     score_ACP = norm(x_barre - U_tilde*U_tilde'*x_barre,2)/norm(x_barre,2);
     r(tests,k) = score_ACP;
     if(tests==k)
       figure(100+k)
       subplot(1,2,1); 
       imshow(reshape(tes(:,tests),[16,16]));
       subplot(1,2,2);
       imshow(reshape(U_tilde*U_tilde'*x_barre + D_moy, [16,16]));
       title('image reconstruite par ACP');
     end  
    %%%%%%%%%%%%%%%%%%%%%%%%% FIN TO DO %%%%%%%%%%%%%%%%%%
  
   % Classification depuis kernel ACP
     %%%%%%%%%%%%%%%%%%%%%%%%% TO DO %%%%%%%%%%%%%%%%%%
     disp('kernel PCA : classification');
     disp('TO DO')
     

    
    %%%%%%%%%%%%%%%%%%%%%%%%% FIN TO DO %%%%%%%%%%%%%%%%%%    
 end
 
end


% Affichage du résultat de l'analyse par PCA
couleur = hsv(6);

figure(11)
for tests=1:6
     hold on
     plot(1:5, r(tests,:),  '+', 'Color', couleur(tests,:));
     hold off
 
     for i = 1:4
        hold on
         plot(i:0.1:(i+1),r(tests,i):(r(tests,i+1)-r(tests,i))/10:r(tests,i+1), 'Color', couleur(tests,:),'LineWidth',2)
         hold off
     end
     hold on
     if(tests==6)
       testa=9;
     else
       testa=tests;  
     end
     text(5,r(tests,5),num2str(testa));
     hold off
 end

% Affichage du résultat de l'analyse par kernel PCA
figure(12)
for tests=1:6
     hold on
     plot(1:5, r2(tests,:),  '+', 'Color', couleur(tests,:));
     hold off
 
     for i = 1:4
        hold on
         plot(i:0.1:(i+1),r2(tests,i):(r2(tests,i+1)-r2(tests,i))/10:r2(tests,i+1), 'Color', couleur(tests,:),'LineWidth',2)
         hold off
     end
     hold on
     if(tests==6)
       testa=9;
     else
       testa=tests;  
     end
     text(5,r2(tests,5),num2str(testa));
     hold off
end