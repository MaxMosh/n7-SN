close all;
clear all;

A = [1 0 0 9 0 0 9;
     0 2 9 0 0 0 9;
     0 9 3 0 9 9 0;
     9 0 0 4 0 0 0;
     0 0 9 0 5 0 9;
     0 0 9 0 0 6 0;
     9 9 0 0 9 0 7]

subplot(2,3,1);
spy(A);
title('Original matrix A');
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

%P = symamd(A)
%P = P(end:-1:1)
%P = [4 1 6 2 3 5 7]
P = symrcm(A)

B = A(P,P)

subplot(2,3,4)
spy(B);
title('Permuted matrix A');
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
title('Factors of permuted A')
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
title('Fill on permuted A')