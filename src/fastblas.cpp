/*
    Dynamic Genomic Best linear Unbiased Prediction  (DynGBLUP)
    Copyright © 2026, Meixia Ye
    
*/

#include "gsl/gsl_matrix.h"
#include <algorithm>    
#include <cmath>
#include <iomanip>
#include <vector>
#include "debug.h"
#include "fastblas.h"
#include "mathfunc.h"
#include <string.h>
#include "eigenlib.h"

using namespace std;

/*
   Reasonably fast function to copy data from standard C array into
   gsl_matrix. Avoid it for performance critical sections.
*/
gsl_matrix *fast_copy(gsl_matrix *m, const double *mem) {
  auto rows = m->size1;
  auto cols = m->size2;
  if (is_strict_mode()) { 
    for (size_t r=0; r<rows; r++) {
      for (size_t c=0; c<cols; c++) {
        gsl_matrix_set(m,r,c,mem[r*cols+c]);
      }
    }
  } else { 
    auto v = gsl_vector_calloc(cols);
    enforce(v); 
    for (size_t r=0; r<rows; r++) {
      assert(v->size == cols);
      assert(v->block->size == cols);
      assert(v->stride == 1);
      memcpy(v->block->data,&mem[r*cols],cols*sizeof(double));
      gsl_matrix_set_row(m,r,v);
    }
    gsl_vector_free(v);
  }
  return m;
}

/*
    Helper function fast_cblas_dgemm runs the local dgemm
*/
void fast_cblas_dgemm(const enum CBLAS_ORDER Order,
                      const enum CBLAS_TRANSPOSE TransA,
                      const enum CBLAS_TRANSPOSE TransB,
                      const size_t M,
                      const size_t N,
                      const size_t K,
                      const double alpha,
                      const double *A,
                      const size_t lda,
                      const double *B,
                      const size_t ldb,
                      const double beta,
                      double *C,
                      const size_t ldc) {
#ifndef NDEBUG
  if (is_debug_mode()) {
    #ifdef DISABLED
    size_t i,j;
    printf (" Top left corner of matrix A: \n");
    for (i=0; i<min(M,6); i++) {
      for (j=0; j<min(K,6); j++) {
        printf ("%12.0f", A[j+i*K]);
      }
      printf ("\n");
    }

    printf ("\n Top left corner of matrix B: \n");
    for (i=0; i<min(K,6); i++) {
      for (j=0; j<min(N,6); j++) {
        printf ("%12.0f", B[j+i*N]);
      }
      printf ("\n");
    }

    printf ("\n Top left corner of matrix C: \n");
    for (i=0; i<min(M,6); i++) {
      for (j=0; j<min(N,6); j++) {
        printf ("%12.5G", C[j+i*N]);
      }
      printf ("\n");
    }
    #endif

    cout << scientific << setprecision(3) << "* RowMajor " << Order << "\t" ;
    cout << "transA " << TransA << "\t" ;
    cout << "transB " << TransB << "\t" ;
    cout << "m " << M << "\t" ;
    cout << "n " << N << "\t" ;
    cout << "k " << K << "\n" ;
    cout << "* lda " << lda << "\t" ;
    cout << "ldb " << ldb << "\t" ;
    cout << "ldc " << ldc << "\t" ;
    cout << "alpha " << alpha << "\t" ;
    cout << "beta " << beta << "\n" ;
    cout << "* A03 " << A[3] << "\t" ;
    cout << "B03 " << B[3] << "\t" ;
    cout << "C03 " << C[3] << "\t" ;
    cout << "Asum " << sum(A,M,K) << "\t" ;
    cout << "Bsum " << sum(B,K,N) << "\n" ;
    cout << "Csum " << sum(C,M,N) << "\n" ;
  }
#endif 

  enforce(M>0);
  enforce(N>0);
  enforce(K>0);

  check_int_mult_overflow(M,K);
  check_int_mult_overflow(N,K);
  check_int_mult_overflow(M,N);

  cblas_dgemm(Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);

#ifndef NDEBUG
  #ifdef DISABLED
  if (is_debug_mode()) {
    printf (" Top left corner of matrix A (cols=k %i, rows=m %i): \n",K,M);
    for (i=0; i<min(M,6); i++) {
      for (j=0; j<min(K,6); j++) {
        printf ("%12.0f", A[j+i*K]);
      }
      printf ("\n");
    }

    printf ("\n Top left corner of matrix B: \n");
    for (i=0; i<min(K,6); i++) {
      for (j=0; j<min(N,6); j++) {
        printf ("%12.0f", B[j+i*N]);
      }
      printf ("\n");
    }

    printf ("\n Top left corner of matrix C: \n");
    for (i=0; i<min(M,6); i++) {
      for (j=0; j<min(N,6); j++) {
      printf ("%12.5G", C[j+i*N]);
      }
      printf ("\n");
    }
  }
  #endif
#endif 
}

/*
    Helper function fast_cblas_dgemm converts a GEMMA layout to cblas_dgemm.
*/
static void fast_cblas_dgemm(const char *TransA, const char *TransB, const double alpha,
                             const gsl_matrix *A, const gsl_matrix *B, const double beta,
                             gsl_matrix *C) {

  auto transA = (*TransA == 'N' || *TransA == 'n' ? CblasNoTrans : CblasTrans);
  auto transB = (*TransB == 'N' || *TransB == 'n' ? CblasNoTrans : CblasTrans);
  const size_t M   = C->size1;
  const size_t N   = C->size2;
  const size_t MA  = (transA == CblasNoTrans) ? A->size1 : A->size2;
  const size_t NA  = (transA == CblasNoTrans) ? A->size2 : A->size1;
  const size_t MBx = (transB == CblasNoTrans) ? B->size1 : B->size2;
  const size_t NB  = (transB == CblasNoTrans) ? B->size2 : B->size1;

  if (M == MA && N == NB && NA == MBx) {  /* [MxN] = [MAxNA][MBxNB] */

    auto K = NA;

   
    enforce(M>0);
    enforce(N>0);
    enforce(K>0);

   
    check_int_mult_overflow(M,K);
    check_int_mult_overflow(N,K);
    check_int_mult_overflow(M,N);

    cblas_dgemm (CblasRowMajor, transA, transB, M, N, NA,
                 alpha, A->data, A->tda, B->data, B->tda, beta,
                 C->data, C->tda);

  } else {
    fail_msg("Range error in dgemm");
  }
}


/*
   Use the fast/supported way to call BLAS dgemm
*/

void fast_dgemm(const char *TransA, const char *TransB, const double alpha,
                const gsl_matrix *A, const gsl_matrix *B, const double beta,
                gsl_matrix *C) {
  fast_cblas_dgemm(TransA,TransB,alpha,A,B,beta,C);

#ifdef DISABLE
  if (is_check_mode()) {
   
    gsl_matrix *C1 = gsl_matrix_alloc(C->size1,C->size2);
    eigenlib_dgemm(TransA,TransB,alpha,A,B,beta,C1);
    enforce_msg(gsl_matrix_equal(C,C1),"dgemm outcomes are not equal for fast & eigenlib");
    gsl_matrix_free(C1);
  }
#endif
}

void fast_eigen_dgemm(const char *TransA, const char *TransB, const double alpha,
                      const gsl_matrix *A, const gsl_matrix *B, const double beta,
                      gsl_matrix *C) {
  if (is_legacy_mode())
    eigenlib_dgemm(TransA,TransB,alpha,A,B,beta,C);
  else
    fast_cblas_dgemm(TransA,TransB,alpha,A,B,beta,C);
}
