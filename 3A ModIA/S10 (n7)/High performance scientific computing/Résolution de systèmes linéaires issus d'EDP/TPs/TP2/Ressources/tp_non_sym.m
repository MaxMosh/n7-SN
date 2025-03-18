close all;
clear all;

load hydcar20;

n = size(A,1);

subplot(1,4,1);
spy(A);
title('Original matrix A');
 
[L, U, Pn] = lu(A);
subplot(1,4,2)
spy(Pn*A);
title('Pn*A');

subplot(1,4,3)
ALU = L+U;
spy(ALU);
title('Factors of Pn*A')
fillin = nnz(ALU)-nnz(A)

% visualisation du fill
C = spones(Pn*A);
CLU = spones(ALU);
FILL = CLU-C;
subplot(1,4,4)
spy(FILL)
title('Fill on Pn*A')
