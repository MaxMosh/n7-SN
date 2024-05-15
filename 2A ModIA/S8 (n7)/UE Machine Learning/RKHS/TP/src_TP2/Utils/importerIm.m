function [ matrice ] = importerIm( NomImage, m, n, nPix, mPix )
% Fonction permettant l'importation d'une image jpg et effectuatn un pretraitement
% en vu de l'etape de reconnaissance
% NE PAS MODIFIER
epsilon=1;
image = imread(NomImage);
image = rgb2gray(image);
image = im2double(image);
M = zeros(m+1,1);
N = zeros(n+1,1);

matrice = zeros(nPix*mPix,m*n);
ligne = image(1,:);
colonne = image(:,1);
nImg = size(image,1);
mImg = size(image,2);
j = 1;
jaux = 1;


for i=1:m+1
    while j<mImg && norm(image(:,j)-colonne,1) <epsilon
        j = j+1;
    end
    M(i) = floor((j+jaux)/2);
    while j< mImg && norm(image(:,j)-colonne,1) > epsilon
        j = j+1;
    end

    jaux = j;
end

j= 1;
jaux = 1;

for i=1:n+1
    while j<nImg && norm(image(j,:)-ligne,1) < epsilon
        j = j+1;
    end
    N(i) = floor((j+jaux)/2);
    while j<nImg && norm(image(j,:)-ligne,1) > epsilon
        j = j+1;
    end

    jaux = j;
end
index = 1;
for I=1:n
    for J=1:m
        clear A;
        clear B;
        A = image(N(I):N(I+1),M(J):M(J+1));

        A = definirZone(A);
        B = zeros(n,m);
        nA = size(A,1);
        mA = size(A,2);
        aN = nA/nPix;
        aM = mA/mPix;
        for i=1:mPix
            for j=1:nPix
                B(i,j) = densite(A,floor((i-1)*aN)+1, floor((j-1)*aM)+1, floor((i)*aN), floor((j)*aM));
            end
        end
         
        B = B(:);
        matrice(:,index) = B;
        index = index+1;
    end
end
end

