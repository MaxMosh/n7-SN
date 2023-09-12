function image_retrouvee = reconstitution_image(suite_binaire_reconstruite)

%Reconstruction de l'image à partir de la suite de bits retrouvée
mat_image_binaire_retrouvee=reshape(suite_binaire_reconstruite,105*100,8);
mat_image_decimal_retrouvee=bi2de(mat_image_binaire_retrouvee);
image_retrouvee=reshape(mat_image_decimal_retrouvee,105,100);