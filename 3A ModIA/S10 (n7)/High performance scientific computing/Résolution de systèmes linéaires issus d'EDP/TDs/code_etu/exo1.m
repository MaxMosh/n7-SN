close all
clear all

A = [1 1 0 1 0;
     1 1 0 1 1; 
     0 0 1 0 1; 
     1 1 0 1 0; 
     0 1 1 0 1];

% pas de permutation
P = [1 2 3 4 5]

subplot(2,3,1);
spy(A);
title('Original matrix A');

[count, h, parent, post, R] = symbfact(A);
ALU = R + R';
%figure(2);
subplot(2, 3, 2)
spy(ALU);
title('Factors of A')
fillin = nnz(ALU) - nnz(A)
% visualisation du fill
C = spones(A);
CLU = spones(ALU);
FILL = CLU - C;
subplot(2, 3, 3)
spy(FILL)
title('Fill on original A')
B = A(P, P);
subplot(2, 3, 4)
spy(B);
title('Permuted matrix')
%%pause
%%close
%
%    factoriser la matrice permutee
%
[count, h, parent, post, R] = symbfact(B);
BLU = R + R';
%figure(3);
subplot(2, 3, 5)
spy(BLU);
fillin = nnz(BLU) - nnz(A)
title('Factors of permuted A');
%%pause
%%close
%
%        visualiser le remplissage
%
%figure(4);
B = spones(B);
BLU = spones(BLU);
FILL = BLU - B;
subplot(2, 3, 6)
spy(FILL);
title('Fill on permuted A');