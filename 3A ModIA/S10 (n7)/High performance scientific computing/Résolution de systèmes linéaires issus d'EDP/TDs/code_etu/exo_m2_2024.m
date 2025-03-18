%echo on
clear all;
close all;

figure(1);
%
%  0/  chargement de la matrice de nom mat_matlab
%
load mat_matlab;
mat=mat_matlab;
clear mat_matlab;
A=sparse(mat(:,1),mat(:,2),1);
subplot(2,3,1);
spy(A);
title('Original matrix A');
%%pause
%%close
%
% 1/ factorisation symbolique de la matrice originale
%
[count, h, parent, post, R] = symbfact(A);
ALU=R+R';
%figure(2);
subplot(2,3,2)
spy(ALU);
title('Factors of A')
fillin=nnz(ALU)-nnz(A)
% visualisation du fill
C=spones(A);
CLU=spones(ALU);
FILL=CLU-C;
subplot(2,3,3)
spy(FILL)
title('Fill on original A')
%%pause
%%close
%figure(2);
%%close;
%
% 2/ Lire tableau de permutation de nom 
%  perm_matlabMD  (ordering basé sur le minimum degree) ou
%  perm_matlabCM  ( ..           Cuthill Mc Kee)        ou 
%  perm_matlabRCM ( ..           Reverse Cuthill Mc Kee)
%    puis permuter la matrice
%
P = [1 2 3 4 5 6 7 8 9];
%MD
%P =

%CMK 1 passe
%P =

%CMK 2 passes
%P = 

% CMK 3 passes idem
%P = 

B=A(P,P);
%figure(2);
subplot(2,3,4)
spy(B);
%%pause
%%close
%
%    factoriser la matrice permutee
%
[count,h,parent,post,R] = symbfact(B);
BLU=R+R';
%figure(3);
spy(BLU);
fillin=nnz(BLU)-nnz(A)
%%pause
%%close
%
%        visualiser le remplissage
%
%figure(4);
B=spones(B);
BLU=spones(BLU);
FILLB=BLU-B;
subplot(2,3,6)
spy(FILLB);
%%pause
%%close;

%
%     Experimenter les outils matlab (symmmd, amd, rcm)
%

%P = [3  8 9 6 1 2 4 5 7];
%P = [1 3 8 9 6 2 4 5 7]
B=A(P,P);
subplot(2,3,4)
spy(B);
title('Permuted matrix')
%%pause
%%close
%
%    factoriser la matrice permutee
%
[count,h,parent,post,R] = symbfact(B);
BLU=R+R';
%figure(3);
subplot(2,3,5)
spy(BLU);
fillin=nnz(BLU)-nnz(A)
title('Factors of permuted A');
%%pause
%%close
%
%        visualiser le remplissage
%
%figure(4);
B=spones(B);
BLU=spones(BLU);
FILL=BLU-B;
subplot(2,3,6)
spy(FILL);
title('Fill on permuted A');

