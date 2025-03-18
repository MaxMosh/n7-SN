close all;
clear all;

load mat0;

n = size(A,1);

subplot(2,3,1);
spy(A);
title('Original matrix A');
 
[count, h, parent, post, R] = symbfact(A);
ALU = R+R';
subplot(2,3,2)
spy(ALU);
title('Factors of A')
fillin = nnz(ALU)-nnz(A)

% visualisation du fill
C = spones(A);
CLU = spones(ALU);
FILL = CLU-C;
subplot(2,3,3)
spy(FILL)
title('Fill on original A')

% Permutation (à modifier)
P = [1:n]

B = A(P,P);
subplot(2,3,4)
spy(B);
title('Permuted matrix')

[count, h, parent, post, R] = symbfact(B);
BLU = R+R';
subplot(2,3,5)
spy(BLU);
fillin = nnz(BLU)-nnz(A)
title('Factors of permuted A');

B = spones(B);
BLU = spones(BLU);
FILL = BLU-B;
subplot(2,3,6)
spy(FILL);
title('Fill on permuted A');
