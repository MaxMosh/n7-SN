A = [1 0 0 1 0 0 0 1 0;
     0 1 0 0 1 1 0 0 0;
     0 0 1 0 0 0 0 1 0;
     1 0 0 1 0 1 1 0 0;
     0 1 0 0 1 0 0 0 1;
     0 1 0 1 0 1 0 1 0;
     0 0 0 1 0 0 1 0 0;
     1 0 1 0 0 1 0 1 0;
     0 0 0 0 1 0 0 0 1];
 
 %norm(A - A')
 n = size(A,1);
 
subplot(2,3,1);
spy(A);
title('Original matrix A');

 
[count,h,parent,post,R] = symbfact(A);
ALU=R+R';
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

%%%%%%%%%%%%%%% Cuthill-McKee
% pas de permutation (à modifier)
P = [1:n]

B = A(P,P);
subplot(2,3,4)
spy(B);
title('Permuted matrix (CM)')

[count,h,parent,post,R] = symbfact(B);
BLU=R+R';
subplot(2,3,5)
spy(BLU);
fillin=nnz(BLU)-nnz(A)
title('Factors of permuted A (CM)');

B=spones(B);
BLU=spones(BLU);
FILL=BLU-B;
subplot(2,3,6)
spy(FILL);
title('Fill on permuted A (CM)');

pause

%%%%%%%%%%%%%%% Reverse Cuthill-McKee
% pas de permutation (à modifier)
P = [1:n]

B = A(P,P);
subplot(2,3,4)
spy(B);
title('Permuted matrix (RCM)')

[count,h,parent,post,R] = symbfact(B);
BLU=R+R';
subplot(2,3,5)
spy(BLU);
fillin=nnz(BLU)-nnz(A)
title('Factors of permuted A (RCM)');

B=spones(B);
BLU=spones(BLU);
FILL=BLU-B;
subplot(2,3,6)
spy(FILL);
title('Fill on permuted A (RCM)');

pause

%%%%%%%%%%%%%%% Minimum-degree et Approximate
% pas de permutation (à modifier)
P = [1:n]

C = A(P,P);

subplot(2,3,4)
spy(C);
title('Permuted matrix (MD)')

[count,h,parent,post,R] = symbfact(C);
CLU=R+R';
subplot(2,3,5)
spy(CLU);
fillin=nnz(CLU)-nnz(A)
title('Factors of permuted A (MD)');

C=spones(C);
CLU=spones(CLU);
FILL=CLU-C;
subplot(2,3,6)
spy(FILL);
title('Fill on permuted A (MD)');