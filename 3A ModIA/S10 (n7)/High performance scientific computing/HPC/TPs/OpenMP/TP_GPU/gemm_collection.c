#include <stdio.h>
#include <omp.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cblas.h>
#include "utils.h"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

void gemm_seq(size_t M, size_t N, size_t K, float alpha, float *A,
              size_t lda, float *B, size_t ldb, float beta, float *C,
              size_t ldc) {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M,
                N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}


void gemm_cpu(size_t M, size_t N, size_t K, float alpha, float *A,
              size_t lda, float *B, size_t ldb, float beta, int nb_th,
              float *C, size_t ldc) {

    int th;         // Thread id number
    int nb_col;     // Number of columns in a block column
    int firstcol;   // Index of the first column of the current block column

    #pragma omp parallel private(th, nb_col, firstcol)
    {
        // TODO: Separate the GEMM work between the available OMP threads
        th = omp_get_thread_num(); //
        nb_col = N/nb_th + (th < N%nb_th ? 1 : 0);  //
        firstcol = ((th < N%nb_th) ? th*nb_col : th*nb_col + N%nb_th); //
        printf("th = %d - nb_col = %d - first_col = %d\n", th, nb_col , firstcol); //

        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M,
                    nb_col, // TODO: To replace -> N en nb_col
                    K, alpha, A, lda,
                    &B[firstcol*K], // TODO: To replace -> 0 en first_col*K
                    ldb, beta,
                    &C[firstcol*M], // TODO: To replace -> 0 en first_col*M
                    ldc);

    }
    return;
}


void gemm_gpu(size_t M, size_t N, size_t K, float alpha, float *A,
              size_t lda, float *B, size_t ldb, float beta, float *C,
              size_t ldc) {
    int ierr;
    cublasHandle_t handle;

    cublasCreate(&handle);

    // TODO: Send/Retrieve the data on the GPU
    // TODO: Execute the cublasSgemm routine on the GPU

    # pragma omp target data map(to:A[0:K*M]) map(to:B[0:K*N]) map(tofrom:C[0:M*N])
    {
        # pragma omp target data use_device_ptr(A,B,C)
            ierr = cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A,
                            M, B, K, &beta, C, M);
    }
    if(ierr != CUBLAS_STATUS_SUCCESS)
    {
       printf( "failed %d %f.\n", ierr, C[0]);
       exit(1);
    } 
    // TODO (Optional): Deallocate data on the GPU

    cublasDestroy(handle);
    return;
}


void gemm_gpu_cpu(size_t M, size_t N, size_t K, float alpha, float *A,
              size_t lda, float *B, size_t ldb, float beta, int nb_th,
              int per, float *C, size_t ldc) {

    size_t nb_col_gpu = per*N/100; // = ? TODO // Number of column for the GPU block
    size_t nb_col_cpu = N - nb_col_gpu; // = ? TODO // Number of column for the CPU block

    int th;         // Thread id number
    int nb_col;     // Number of columns in a block column
    int firstcol;   // Index of the first column of the current block column

    // TODO: Separate the GEMM work such that the GPU compute a percentage 
    // "per" of the GEMM and the CPU threads compute the remaining 
    
    # pragma omp parallel private(th, nb_col, firstcol)
    {
    // IF (I AM THE 0th THREAD) THEN
    //      I COMPUTE THE GPU WORKLOAD
    // ELSE
    //      I COMPUTE A PART OF THE CPU WORKLOAD
    // ENDIF
    th = omp_get_thread_num();

    if (th == 0)
    {
        if (nb_col_gpu != 0)
        {
            gemm_gpu(M,nb_col_gpu,K,1.,A,lda,B,ldb,0.,C,M);
        }
    } else 
    {
        nb_col = nb_col_cpu/(nb_th - 1) + (th - 1 < nb_col_cpu%(nb_th - 1) ? 1 : 0);
        firstcol = nb_col_gpu + ((th - 1 < nb_col_cpu%(nb_th - 1)) ? (th - 1)*nb_col : (th - 1)*nb_col * nb_col_cpu%(nb_th - 1));
        printf("th = %d - nb_col = %d - firstcol = %d\n", th, nb_col, firstcol);

        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M,   \
                nb_col, K, alpha, A, lda, &B[firstcol*K], ldb,           \
                beta, &C[firstcol*M], ldc);
    } 

    }
    return;

    // TODO : ne converge pas
}

