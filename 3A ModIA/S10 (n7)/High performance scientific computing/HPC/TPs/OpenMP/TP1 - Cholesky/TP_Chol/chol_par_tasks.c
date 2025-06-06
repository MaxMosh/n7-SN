#include "trace.h"
#include "common.h"

/* This is a sequential routine for the LU factorization of a square
   matrix in block-columns */
void chol_par_tasks(matrix_t A){


    int i, j, k;
    int prio;


    #pragma omp parallel
    {
        #pragma omp single
        {
            for(k=0; k<A.NB; k++){
                /* reduce the diagonal block */
                #pragma omp task depend(inout:A.blocks[k][k]) firstprivate(k) priority(4)
                {
                potrf(A.blocks[k][k]);
                }
                for(i=k+1; i<A.NB; i++){
                    prio = (i == k + 1 ? 3 : 1);
                    
                    /* compute the A[i][k] sub-diagonal block */
                    #pragma omp task depend(in:A.blocks[k][k]) depend(inout:A.blocks[i][k]) firstprivate(k,i) priority(prio)
                    {
                    trsm(A.blocks[k][k], A.blocks[i][k]);
                    }
                    for(j=k+1; j<=i; j++){
                        prio = (i == k + 1 && j == i ? 2 : 0);

                        /* update the A[i][j] block in the trailing submatrix */
                        #pragma omp task depend(in:A.blocks[i][k], A.blocks[j][k]) depend(inout:A.blocks[i][j]) firstprivate(k,i,j) priority(prio)
                        {
                        gemm(A.blocks[i][k], A.blocks[j][k], A.blocks[i][j]);
                        }
                    }    
                }
            }
        }
    }
    return;

}

