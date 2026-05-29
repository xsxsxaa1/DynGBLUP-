/*
    Dynamic Genomic Best linear Unbiased Prediction  (DynGBLUP)
    Copyright © 2026, Meixia Ye
    
*/


#include <fstream>
#include <sstream>

#include <assert.h>
#include <bitset>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>


#include "gsl/gsl_blas.h"
#include "gsl/gsl_cdf.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_min.h"
#include "gsl/gsl_roots.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_sort.h"
#include "gsl/gsl_eigen.h"
#include "gsl/gsl_statistics.h"
#include "gsl/gsl_multimin.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_multifit_nlinear.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "io.h"

#include "mathfunc.h"
#include "gzstream.h"
#include "lapack.h"
#include "param.h"
#include "fastblas.h"
#include "iterate.h"


using namespace std;

void MVLMM::CopyFromParam(PARAM &cPar) {
  a_mode = cPar.a_mode;
  d_pace = cPar.d_pace;

  file_bfile = cPar.file_bfile;
  file_geno  = cPar.file_geno;
  file_pheno = cPar.file_pheno;
  file_anno  = cPar.file_anno;
  file_cvt   = cPar.file_cvt;
  file_kin   = cPar.file_kin;
  file_ku    = cPar.file_ku;
  file_kd    = cPar.file_kd;
  file_out = cPar.file_out;
  path_out = cPar.path_out;

  logl_mle_H0 =  cPar.logl_mle_H0;
  logl_remle_H0 = cPar.logl_remle_H0;
  Vg_remle_null = cPar.Vg_remle_null;
  Vp_remle_null = cPar.Vp_remle_null;
  Ve_remle_null = cPar.Ve_remle_null;
  beta_remle_null = cPar.beta_remle_null;
  ai_remle_null   = cPar.ai_remle_null;
  pi_remle_null   = cPar.pi_remle_null;

  nrf = cPar.nrf;
  nr1_size = cPar.nr1_size;
  nr2_size = cPar.nr2_size;
  max_iter = cPar.max_iter;
  cc_par = cPar.cc_par;
  cc_gra = cPar.cc_gra;
  
  
  curve_type  = cPar.curve_type;
 

  ni_total = cPar.ni_total;
  ns_total = cPar.ns_total;
  ni_test = cPar.ni_test;
  ns_test = cPar.ns_test;
  n_cvt = cPar.n_cvt;
  n_ph = cPar.n_ph;
  time_opt = 0.0;

  indicator_idv = cPar.indicator_idv;
  indicator_snp = cPar.indicator_snp;
  snpInfo = cPar.snpInfo;

  return;
}

void MVLMM::WriteFiles_yxx() {
printf("snpInfo.size()=%zu, indicator_snp.size()=%zu \n", snpInfo.size(), indicator_snp.size() );

  string file_str;
  file_str = path_out + "/" + file_out;
  file_str += ".chisq_pval_ui.txt";

  ofstream outfile(file_str.c_str(), ofstream::out);
  if (!outfile) {
    cout << "error writing file: " << file_str.c_str() << endl;
    return;
  }

  outfile << "chr" << "\t"
          << "rs" << "\t"
          << "ps" << "\t"
          << "n_miss" << "\t"
          << "allele1" << "\t"
          << "allele0" << "\t"
          << "af" << "\t"
          << "chisq" << "\t"
          << "pval" << "\t";

  for (size_t i = 0; i < sumStat0[1].v_ui_i.size(); i++) { 
    outfile << "ui_" << i << "\t"; 
  }
  outfile << "" << endl;
  
  

  int t = 0, c = 0; 
  for (size_t i = 0; i < snpInfo.size(); ++i) {
    if (indicator_snp[i] == 0) {
    continue;
    }

    outfile << snpInfo[i].chr << "\t" << snpInfo[i].rs_number << "\t"
            << snpInfo[i].base_position << "\t" << snpInfo[i].n_miss << "\t"
            << snpInfo[i].a_minor << "\t" << snpInfo[i].a_major << "\t" << fixed
            << setprecision(3) << snpInfo[i].maf << "\t";
    outfile << scientific << setprecision(6);

    

    outfile << sumStat0[c].v_chisq << "\t";
    outfile << sumStat0[c].v_pval << "\t";
    for (size_t ii = 0; ii < sumStat0[c].v_ui_i.size(); ii++) {
      outfile << sumStat0[c].v_ui_i[ii] << "\t";
    }
    outfile << "" << endl;
    t++;
    c++;
  }
  printf("c = %zu \n", c);
  outfile.close();
  outfile.clear();
  return;
}


void MVLMM::CopyToParam(PARAM &cPar) {
  cPar.logl_mle_H0 = logl_mle_H0;
  cPar.logl_remle_H0 = logl_remle_H0;

  cPar.Vg_remle_null = Vg_remle_null;
  cPar.Vp_remle_null = Vp_remle_null;
  cPar.Ve_remle_null = Ve_remle_null;
  cPar.beta_remle_null = beta_remle_null;
  cPar.ai_remle_null = ai_remle_null;
  cPar.pi_remle_null = pi_remle_null;

  cPar.time_opt = time_opt;

  return;
}


//----part1: basic function ----//
extern "C" void dgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K,
                       double *ALPHA, double *A, int *LDA, double *B, int *LDB,
                       double *BETA, double *C, int *LDC);
extern "C" void dpotrf_(char *UPLO, int *N, double *A, int *LDA, int *INFO);
extern "C" void dpotrs_(char *UPLO, int *N, int *NRHS, double *A, int *LDA,
                        double *B, int *LDB, int *INFO);
extern "C" void dsyev_(char *JOBZ, char *UPLO, int *N, double *A, int *LDA,
                       double *W, double *WORK, int *LWORK, int *INFO);
extern "C" void dsyevr_(char *JOBZ, char *RANGE, char *UPLO, int *N, double *A,
                        int *LDA, double *VL, double *VU, int *IL, int *IU,
                        double *ABSTOL, int *M, double *W, double *Z, int *LDZ,
                        int *ISUPPZ, double *WORK, int *LWORK, int *IWORK,
                        int *LIWORK, int *INFO);
extern "C" double ddot_(int *N, double *DX, int *INCX, double *DY, int *INCY);



void fast_cblas_dgemm(const char *TransA, const char *TransB, const double alpha,
                      const gsl_matrix *A, const gsl_matrix *B, const double beta,
                      gsl_matrix *C) 
{
  
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





double EigenProc(const gsl_matrix *V_g, const gsl_matrix *V_e, 
             gsl_vector *D_l, gsl_matrix *UltVeh, gsl_matrix *UltVehi) 
{
  size_t d_size = V_g->size1;
  double d, logdet_Ve = 0.0;

 
  gsl_matrix *Lambda = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix *V_e_temp = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix *V_e_h = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix *V_e_hi = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix *VgVehi = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix_set_zero( Lambda );
  gsl_matrix_set_zero( V_e_temp );
  gsl_matrix_set_zero( V_e_h );
  gsl_matrix_set_zero( V_e_hi );
  gsl_matrix_set_zero( VgVehi );

  gsl_matrix *U_l = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix_set_zero(U_l);
  gsl_vector_set_zero(D_l);

  gsl_matrix_memcpy(V_e_temp, V_e);
  EigenDecomp(V_e_temp, U_l, D_l, 0);
 
  gsl_matrix_set_zero(V_e_h);
  gsl_matrix_set_zero(V_e_hi);
  for (size_t i = 0; i < d_size; i++) {
    d = gsl_vector_get(D_l, i);
    if (d <= 0) {
      continue;
    }
    logdet_Ve += log(d);

    gsl_vector_view U_col = gsl_matrix_column(U_l, i);
    d = sqrt(d);
    gsl_blas_dsyr(CblasUpper, d, &U_col.vector, V_e_h);
    d = 1.0 / d;
    gsl_blas_dsyr(CblasUpper, d, &U_col.vector, V_e_hi);
  }

  
  for (size_t i = 0; i < d_size; i++) {
    for (size_t j = 0; j < i; j++) {
      gsl_matrix_set(V_e_h, i, j, gsl_matrix_get(V_e_h, j, i));
      gsl_matrix_set(V_e_hi, i, j, gsl_matrix_get(V_e_hi, j, i));
    }
  }

  
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, V_g, V_e_hi, 0.0, VgVehi);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, V_e_hi, VgVehi, 0.0, Lambda);
 
  
  gsl_matrix_set_zero(U_l);
  gsl_vector_set_zero(D_l);
  EigenDecomp_Zeroed(Lambda, U_l, D_l, 0);
 

  
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, U_l, V_e_h, 0.0, UltVeh);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, U_l, V_e_hi, 0.0, UltVehi);
  

  // free memory
  gsl_matrix_free(Lambda);
  gsl_matrix_free(V_e_temp);
  gsl_matrix_free(V_e_h);
  gsl_matrix_free(V_e_hi);
  gsl_matrix_free(VgVehi);
  gsl_matrix_free(U_l);

  return logdet_Ve;
}

double CalcQi(const gsl_vector *eval, const gsl_vector *D_l, 
              const gsl_matrix *X, const gsl_matrix *UltVehi,
              gsl_matrix *Qi ) 
{
  size_t n_size = eval->size; 
  size_t d_size = D_l->size;
  size_t p_size = Qi->size1;

  double delta, dl, d, logdet_Q;


  gsl_matrix *UltVehi_Xl      = gsl_matrix_alloc(d_size, p_size);
  gsl_matrix *mat_dd          = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix *UltVehiXl_matdd = gsl_matrix_alloc(p_size, d_size);

  gsl_matrix *Q = gsl_matrix_alloc(p_size, p_size);
  gsl_matrix_set_zero(Q);

  for (size_t k = 0; k < n_size; k++) {
     gsl_matrix_const_view X_sub = 
                gsl_matrix_const_submatrix(X, k * d_size, 0, d_size, p_size);
     
     gsl_matrix_set_zero(UltVehi_Xl);
     gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, UltVehi, &X_sub.matrix, 0.0, UltVehi_Xl);

     gsl_matrix_set_zero(mat_dd);
     delta = gsl_vector_get(eval, k);
     for (size_t i = 0; i < d_size; i++) {
       dl = gsl_vector_get(D_l, i);
       d = delta * dl + 1.0;
       gsl_matrix_set(mat_dd, i, i, 1.0 / d);
     }

     gsl_matrix_set_zero( UltVehiXl_matdd );
     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, UltVehi_Xl, mat_dd, 0.0, UltVehiXl_matdd);
     gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, UltVehiXl_matdd, UltVehi_Xl, 1.0, Q);
  }

 
  int sig;
  gsl_permutation *pmt = gsl_permutation_alloc(p_size);
  LUDecomp(Q, pmt, &sig);
  LUInvert(Q, pmt, Qi);

  logdet_Q = LULndet(Q);

  gsl_matrix_free( UltVehi_Xl );
  gsl_matrix_free( mat_dd );
  gsl_matrix_free( UltVehiXl_matdd );
  gsl_matrix_free( Q );
  gsl_permutation_free(pmt);

  return logdet_Q;
}

void CalcHiQi(const gsl_vector *eval, const gsl_vector *Times, 
              const gsl_matrix *X, const gsl_matrix *Z10, const gsl_matrix *Z20, 
              const gsl_matrix *G0, const gsl_matrix *P0, const gsl_matrix *R0, 
              gsl_matrix *Hi_all, gsl_matrix *Qi, double &logdet_H, double &logdet_Q) 
{
  gsl_matrix_set_zero(Hi_all);
  gsl_matrix_set_zero(Qi);
  logdet_H = 0.0;
  logdet_Q = 0.0;

  size_t   n_size = eval->size;
  size_t  dn_size = X->size1; 
  size_t   d_size = dn_size / n_size;
  size_t nr1_size = Z10->size2 - 1;
  size_t nr2_size = Z20->size2 - 1;

  double logdet_Ve = 0.0, delta, dl, d;

  gsl_matrix *Vg = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix *ZG = gsl_matrix_alloc(d_size, nr1_size+1);
  gsl_matrix_set_zero(Vg);
  gsl_matrix_set_zero(ZG);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Z10, G0, 0.0, ZG);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, ZG, Z10, 0.0, Vg);
  gsl_matrix_free(ZG);

  gsl_matrix *Vp = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix *ZP = gsl_matrix_alloc(d_size, nr2_size+1);
  gsl_matrix_set_zero(Vp);
  gsl_matrix_set_zero(ZP);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Z20, P0, 0.0, ZP);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, ZP, Z20, 0.0, Vp);
  gsl_matrix_free(ZP);

  gsl_matrix *Ve = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix_set_zero(Ve);
  gsl_matrix_memcpy(Ve, R0);
  gsl_matrix_add(Ve, Vp);



  gsl_matrix *UltVeh = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix *UltVehi = gsl_matrix_alloc(d_size, d_size);
  gsl_vector *D_l = gsl_vector_alloc(d_size);
  gsl_matrix_set_zero(UltVeh);
  gsl_matrix_set_zero(UltVehi);
  gsl_vector_set_zero(D_l);

  logdet_Ve = EigenProc(Vg, Ve, D_l, UltVeh, UltVehi);
 

  gsl_matrix *mat_dd = gsl_matrix_alloc(d_size, d_size);

  logdet_H = (double)n_size * logdet_Ve;
  for (size_t k = 0; k < n_size; k++) {
    delta = gsl_vector_get(eval, k);

    gsl_matrix_memcpy(mat_dd, UltVehi);
    for (size_t i = 0; i < d_size; i++) {
      dl = gsl_vector_get(D_l, i);
      d = delta * dl + 1.0;
        
      gsl_vector_view mat_row = gsl_matrix_row(mat_dd, i);
      gsl_vector_scale(&mat_row.vector, 1.0 / d); 
      logdet_H += log(d);
    }

    gsl_matrix_view Hi_k = gsl_matrix_submatrix(Hi_all, 0, k * d_size, d_size, d_size);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, UltVehi, mat_dd, 0.0, &Hi_k.matrix);
  }

  
  logdet_Q = CalcQi( eval, D_l, X, UltVehi, Qi ); 
  
  
  gsl_matrix_free( Vg );
  gsl_matrix_free( Vp );
  gsl_matrix_free( Ve );
  gsl_matrix_free( UltVeh );
  gsl_matrix_free( UltVehi );
  gsl_vector_free( D_l );
  gsl_matrix_free( mat_dd );
  
  return;
}

void Calc_Hiy_all(const gsl_matrix *Y, const gsl_matrix *Hi_all,
                  gsl_matrix *Hiy_all) 
{
  size_t d_size  = Hi_all->size1 ;
  size_t dn_size = Hi_all->size2 ;
  size_t n_size = dn_size/d_size ; 

  gsl_matrix_set_zero( Hiy_all );

  for (size_t k = 0; k < n_size; k++) {
    gsl_matrix_const_view Hi_k = gsl_matrix_const_submatrix(Hi_all, 0, k * d_size, d_size, d_size);
    gsl_matrix_const_view Y_k = gsl_matrix_const_submatrix(Y, k*d_size,0, d_size,1);
    gsl_vector_const_view y_k = gsl_matrix_const_column(&Y_k.matrix, 0);

    gsl_vector_view Hiy_k = gsl_matrix_column(Hiy_all, k);

    gsl_blas_dgemv(CblasNoTrans, 1.0, &Hi_k.matrix, &y_k.vector, 0.0, &Hiy_k.vector);
  }

  return;
}

double Calc_yHiy(const gsl_matrix *Y, const gsl_matrix *Hiy_all) 
{
  size_t d_size = Hiy_all->size1;
  size_t n_size = Hiy_all->size2;
  double yHiy = 0.0;
  double d = 0.0;
  
  for (size_t k = 0; k < n_size; k++) {
    gsl_matrix_const_view Y_k = gsl_matrix_const_submatrix(Y, k*d_size,0, d_size,1);
    gsl_vector_const_view y_k = gsl_matrix_const_column(&Y_k.matrix, 0);

    gsl_vector_const_view Hiy_k = gsl_matrix_const_column(Hiy_all, k);

    gsl_blas_ddot(&Hiy_k.vector, &y_k.vector, &d);
    yHiy += d;
  }

  return yHiy;
}

void Calc_XHi_all(const gsl_matrix *X, const gsl_matrix *Hi_all,
                  gsl_matrix *XHi_all) 
{
  gsl_matrix_set_zero(XHi_all);

  size_t  d_size = Hi_all->size1;
  size_t dn_size = Hi_all->size2;
  size_t  n_size = dn_size / d_size;
  size_t  p_size = X->size2;

  for (size_t k = 0; k < n_size; k++) {
    gsl_matrix_const_view X_k =
        gsl_matrix_const_submatrix(X, k * d_size, 0, d_size, p_size);
    gsl_matrix_const_view Hi_k =
        gsl_matrix_const_submatrix(Hi_all, 0, k * d_size, d_size, d_size);  
    gsl_matrix_view XHi_sub =
        gsl_matrix_submatrix(XHi_all, 0, k * d_size, p_size, d_size);

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &X_k.matrix, &Hi_k.matrix, 0.0, &XHi_sub.matrix);
  }

  return;
}

void Calc_XHiy(const gsl_matrix *Y, const gsl_matrix *XHi_all, 
               const gsl_vector *eval,
               gsl_vector *XHiy) 
{
  size_t dn_size = Y->size1;
  size_t  n_size = eval->size;
  size_t  d_size = dn_size / n_size;
  size_t  p_size = XHi_all->size1;

  gsl_vector_set_zero(XHiy);

  for (size_t k = 0; k < n_size; k++) {
    gsl_matrix_const_view XHi_k =
        gsl_matrix_const_submatrix(XHi_all, 0, k * d_size, p_size, d_size);
    gsl_matrix_const_view Y_k = gsl_matrix_const_submatrix(Y, k*d_size,0, d_size,1);
    gsl_vector_const_view y_k = gsl_matrix_const_column(&Y_k.matrix, 0);

    gsl_blas_dgemv(CblasNoTrans, 1.0, &XHi_k.matrix, &y_k.vector, 1.0, XHiy);
  }

  return;
}

double Calc_trace(const gsl_matrix *mat)
{
  size_t d_size = mat->size1;
  double tmp = 0.0;
  double trace = 0.0;

  for (size_t i = 0; i < d_size; ++i) {
    tmp = gsl_matrix_get(mat, i, i);
    trace = trace + tmp;
  }

  return trace;
}



void Calc_Z10_Z20( const gsl_vector *Time, gsl_matrix *Z10, gsl_matrix *Z20 )
{ 
  size_t    d_size = Time->size;
  size_t  nr1_size = Z10->size2 - 1;
  size_t  nr2_size = Z20->size2 - 1;

  gsl_matrix_set_zero(Z10);  
  gsl_matrix_set_zero(Z20);

  //-- get tx ---//
  double tmin = 1; 
  double tmax = d_size; gsl_vector_get(Time, d_size - 1);
  gsl_vector *tx = gsl_vector_alloc(d_size);
  for (size_t i = 0; i < d_size; ++i) {
    double Time_i = i+1; 
    double tmp = -1 + 2 * (Time_i - tmin)/(tmax - tmin);
    gsl_vector_set(tx, i, tmp);
  }

  //-- calc Z10, Z20 ---//
  double tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;
  for (size_t i = 0; i < d_size; ++i) {
    double tx_i = gsl_vector_get(tx, i);
    tmp0 = 1;
    tmp1 = tx_i;
    tmp2 = 0.5 * (3 * (tx_i * tx_i) - 1);
    tmp3 = 0.5 * (5 * (tx_i * tx_i * tx_i) - 3 * tx_i);
    tmp4 = 0.125 * (35 * (tx_i * tx_i * tx_i * tx_i) - 30 * (tx_i * tx_i) + 3);
    tmp5 = 0.125 * (63 * (tx_i * tx_i * tx_i * tx_i * tx_i) - 70 * (tx_i * tx_i * tx_i) + 15 * tx_i);

    if(nr1_size >= 0){ gsl_matrix_set(Z10, i, 0, tmp0); }
    if(nr1_size >= 1){ gsl_matrix_set(Z10, i, 1, tmp1); }
    if(nr1_size >= 2){ gsl_matrix_set(Z10, i, 2, tmp2); }
    if(nr1_size >= 3){ gsl_matrix_set(Z10, i, 3, tmp3); }
    if(nr1_size >= 4){ gsl_matrix_set(Z10, i, 4, tmp4); }
    if(nr1_size >= 5){ gsl_matrix_set(Z10, i, 5, tmp5); }

    if(nr2_size >= 0){ gsl_matrix_set(Z20, i, 0, tmp0); }
    if(nr2_size >= 1){ gsl_matrix_set(Z20, i, 1, tmp1); }
    if(nr2_size >= 2){ gsl_matrix_set(Z20, i, 2, tmp2); }
    if(nr2_size >= 3){ gsl_matrix_set(Z20, i, 3, tmp3); }
    if(nr2_size >= 4){ gsl_matrix_set(Z20, i, 4, tmp4); }
    if(nr2_size >= 5){ gsl_matrix_set(Z20, i, 5, tmp5); }
  }
 

  gsl_vector_free(tx);
  return;  
}

void Calc_Z10_Z20_G0_G0inv_P0_P0inv_R0_R0inv(const gsl_vector *par_vg, const gsl_vector *par_vp, 
                                             const gsl_vector *par_vr, const gsl_vector *Time, 
                                   gsl_matrix *Z10, gsl_matrix *Z20, 
                                   gsl_matrix *G0,  gsl_matrix *P0,  gsl_matrix *R0, 
                                   gsl_matrix *G0_inv, gsl_matrix *P0_inv, gsl_matrix *R0_inv ) 
{
  size_t    d_size = Time->size;
  size_t  nr1_size = Z10->size2 - 1;
  size_t  nr2_size = Z20->size2 - 1;

  //-- set zero ---//
  gsl_matrix_set_zero(Z10);  gsl_matrix_set_zero(Z20);
  gsl_matrix_set_zero(G0);   gsl_matrix_set_zero(G0_inv);
  gsl_matrix_set_zero(P0);   gsl_matrix_set_zero(P0_inv);
  gsl_matrix_set_zero(R0);   gsl_matrix_set_zero(R0_inv);

  //-- get tx ---//
  double tmin = 1; 
  double tmax = d_size; 
  gsl_vector *tx = gsl_vector_alloc(d_size);
  for (size_t i = 0; i < d_size; ++i) {
    double Time_i = i+1; 
    double tmp = -1 + 2 * (Time_i - tmin)/(tmax - tmin);
    gsl_vector_set(tx, i, tmp);
  }

  //-- calc Z10, Z20 ---//
  double tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;
  for (size_t i = 0; i < d_size; ++i) {
    double tx_i = gsl_vector_get(tx, i);
    tmp0 = 1;
    tmp1 = tx_i;
    tmp2 = 0.5 * (3 * (tx_i * tx_i) - 1);
    tmp3 = 0.5 * (5 * (tx_i * tx_i * tx_i) - 3 * tx_i);
    tmp4 = 0.125 * (35 * (tx_i * tx_i * tx_i * tx_i) - 30 * (tx_i * tx_i) + 3);
    tmp5 = 0.125 * (63 * (tx_i * tx_i * tx_i * tx_i * tx_i) - 70 * (tx_i * tx_i * tx_i) + 15 * tx_i);

    if(nr1_size >= 0){ gsl_matrix_set(Z10, i, 0, tmp0); }
    if(nr1_size >= 1){ gsl_matrix_set(Z10, i, 1, tmp1); }
    if(nr1_size >= 2){ gsl_matrix_set(Z10, i, 2, tmp2); }
    if(nr1_size >= 3){ gsl_matrix_set(Z10, i, 3, tmp3); }
    if(nr1_size >= 4){ gsl_matrix_set(Z10, i, 4, tmp4); }
    if(nr1_size >= 5){ gsl_matrix_set(Z10, i, 5, tmp5); }

    if(nr2_size >= 0){ gsl_matrix_set(Z20, i, 0, tmp0); }
    if(nr2_size >= 1){ gsl_matrix_set(Z20, i, 1, tmp1); }
    if(nr2_size >= 2){ gsl_matrix_set(Z20, i, 2, tmp2); }
    if(nr2_size >= 3){ gsl_matrix_set(Z20, i, 3, tmp3); }
    if(nr2_size >= 4){ gsl_matrix_set(Z20, i, 4, tmp4); }
    if(nr2_size >= 5){ gsl_matrix_set(Z20, i, 5, tmp5); }
  }


  //-- set G0, P0, R0 ---//
  for (size_t i = 0; i < nr1_size + 1; ++i) {
    tmp2 = gsl_vector_get(par_vg, i);
    gsl_matrix_set(G0, i, i, tmp2);
    gsl_matrix_set(G0_inv, i, i, 1.0/tmp2);
  }

  for (size_t i = 0; i < nr2_size + 1; ++i) {
    tmp2 = gsl_vector_get(par_vp, i);
    gsl_matrix_set(P0, i, i, tmp2);
    gsl_matrix_set(P0_inv, i, i, 1.0/tmp2);
  }


  //-- Lambda --//
  double phi = gsl_vector_get(par_vr, d_size);
  gsl_matrix *Lambda = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix_set_zero( Lambda );
  for (size_t i = 0; i < d_size; ++i) {
    double Time_i = i+1;
    for (size_t j = 0; j <= i; ++j) {
      double Time_j = j+1; 
      double tmp = pow(phi, abs(Time_i - Time_j));
      gsl_matrix_set(Lambda, i, j, tmp);
    }
  }

  //-- middle --//
  gsl_matrix *middle = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix_set_zero( middle );
  for (size_t i = 0; i < d_size; ++i) {
    double middle_i = gsl_vector_get(par_vr, i);
    gsl_matrix_set(middle, i, i, middle_i);
  }

  //-- R0 --//
  gsl_matrix *Lambda_middle = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix_set_zero( Lambda_middle );
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Lambda, middle, 0.0, Lambda_middle);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Lambda_middle, Lambda, 0.0, R0);
  
  gsl_matrix_free( Lambda );
  gsl_matrix_free( middle );
  gsl_matrix_free( Lambda_middle );


  //-- R0_inv --//
  gsl_matrix_set_zero( R0_inv );
  for (size_t i = 0; i < (d_size-1); ++i) {
    double vi  = gsl_vector_get(par_vr, i); 
    double vi1 = gsl_vector_get(par_vr, i+1);

    
    double Time_i  = i+1; 
    double Time_i1 = i+2; 

    double tmp1 = 1.0/vi + (1.0/vi1) * pow(phi, 2*(Time_i1 - Time_i));
    double tmp2 = -(1.0/vi1) * pow(phi, Time_i1 - Time_i);

    gsl_matrix_set(R0_inv, i, i, tmp1);
    gsl_matrix_set(R0_inv, i, i+1, tmp2);
    gsl_matrix_set(R0_inv, i+1, i, tmp2);
  }
  double vi  = gsl_vector_get(par_vr, d_size-1);
  gsl_matrix_set(R0_inv, d_size-1, d_size-1, 1.0/vi);
  

  gsl_vector_free( tx );

  return;
}


void Calc_alltemp(const gsl_vector *eval, const gsl_matrix *Z10, const gsl_matrix *Z20, 
                  const gsl_matrix *G0, const gsl_matrix *G0_inv, const gsl_matrix *R0_inv, const gsl_matrix *P0_inv,  
                gsl_matrix *ZGi, gsl_matrix *p22p22p12ZGip12p22, 
                gsl_matrix *final, gsl_matrix *M, gsl_matrix *N )
{
  size_t n_size  = eval->size;
  size_t d_size  = Z10->size1;
  size_t nr1_size = Z10->size2 - 1;
  size_t nr2_size = Z20->size2 - 1;
  
  gsl_matrix_set_zero( ZGi );
  gsl_matrix_set_zero( p22p22p12ZGip12p22 );
  gsl_matrix_set_zero( final );
  gsl_matrix_set_zero( M );
  gsl_matrix_set_zero( N );


  //-- step 1 --//-- R0invZ10, R0invZ20 --//
  gsl_matrix *R0invZ10 = gsl_matrix_alloc(d_size, nr1_size+1);
  gsl_matrix *R0invZ20 = gsl_matrix_alloc(d_size, nr2_size+1);
  gsl_matrix_set_zero(R0invZ10);
  gsl_matrix_set_zero(R0invZ20);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, R0_inv, Z10, 0.0, R0invZ10);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, R0_inv, Z20, 0.0, R0invZ20);
  
  //-- step 2 --//
  //-- p11 -- p12 --//
  gsl_matrix *p11 = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
  gsl_matrix *p12 = gsl_matrix_alloc(nr1_size+1, nr2_size+1);
  gsl_matrix_set_zero(p11);
  gsl_matrix_set_zero(p12);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Z10, R0invZ10, 0.0, p11);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Z10, R0invZ20, 0.0, p12);
  
  //-- p22 --//
  gsl_matrix *p22_raw = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
  gsl_matrix *p22 = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
  gsl_matrix_set_zero( p22_raw );
  gsl_matrix_set_zero( p22 );
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Z20, R0invZ20, 0.0, p22_raw);
  gsl_matrix_add(p22_raw, P0_inv);
  int sig;
  gsl_permutation *pmt1 = gsl_permutation_alloc(nr2_size + 1);
  LUDecomp(p22_raw, pmt1, &sig);
  LUInvert(p22_raw, pmt1, p22);
  gsl_permutation_free(pmt1);
  gsl_matrix_free(p22_raw);

  //-- p12p22 --//
  gsl_matrix *p12p22 = gsl_matrix_alloc(nr1_size+1, nr2_size+1);
  gsl_matrix_set_zero(p12p22);  
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, p12, p22, 0.0, p12p22);

  //-- zrzp --//
  gsl_matrix *zrzp = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
  gsl_matrix_set_zero( zrzp );
  gsl_matrix_add( zrzp, p11 );
  gsl_matrix *p12p22p12 = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
  gsl_matrix_set_zero( p12p22p12 );
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, p12p22, p12, 0.0, p12p22p12);
  gsl_matrix_sub( zrzp, p12p22p12 );
  gsl_matrix_free( p12p22p12 );

  //-- ZGi --//

  if(1 == 1){
    gsl_matrix_set_zero( ZGi );
    gsl_permutation *pmt5 = gsl_permutation_alloc(nr1_size + 1);

    gsl_matrix *M0  = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
    gsl_matrix *M0_inv = gsl_matrix_alloc(nr1_size+1, nr1_size+1);

    for (size_t i = 0; i < n_size; ++i) {
      double delta_l = gsl_vector_get(eval, i);
  
      gsl_matrix_set_zero( M0 );
      gsl_matrix_add(M0, G0_inv);
      gsl_matrix_scale(M0, 1.0/delta_l);
      gsl_matrix_add(M0, zrzp);

      gsl_matrix_set_zero( M0_inv );
      LUDecomp(M0, pmt5, &sig);
      LUInvert(M0, pmt5, M0_inv);

      gsl_matrix_view ZGi_sub = gsl_matrix_submatrix(ZGi, 0, i*(nr1_size+1), nr1_size+1, nr1_size+1);
      gsl_matrix_add( &ZGi_sub.matrix, M0_inv );
    }

    gsl_matrix_free( M0 ); 
    gsl_matrix_free( M0_inv ); 
    gsl_permutation_free( pmt5 );
  
  }


  //-- ZGip12p22 --//
  gsl_matrix *ZGip12p22 = gsl_matrix_alloc(nr1_size+1, (nr2_size+1)*n_size);
  gsl_matrix_set_zero( ZGip12p22 );
  for (size_t i = 0; i < n_size; ++i) {
    gsl_matrix_view ZGi_sub = gsl_matrix_submatrix(ZGi, 0, i*(nr1_size+1), nr1_size+1, nr1_size+1);  
    gsl_matrix_view ZGip12p22_sub = gsl_matrix_submatrix(ZGip12p22, 0, i*(nr2_size+1), nr1_size+1, nr2_size+1);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &ZGi_sub.matrix, p12p22, 0.0, &ZGip12p22_sub.matrix);
  }


  //-- p22p22p12ZGip12p22 --//
 
  gsl_matrix_set_zero( p22p22p12ZGip12p22 );
  for (size_t i = 0; i < n_size; ++i) {
    gsl_matrix_view ZGip12p22_sub = gsl_matrix_submatrix(ZGip12p22, 0, i*(nr2_size+1), nr1_size+1, nr2_size+1);
    gsl_matrix_view p22p22p12ZGip12p22_sub = gsl_matrix_submatrix(p22p22p12ZGip12p22, 0, i*(nr2_size+1), nr2_size+1, nr2_size+1);
    
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, p12p22, &ZGip12p22_sub.matrix, 0.0, &p22p22p12ZGip12p22_sub.matrix);
    gsl_matrix_add(&p22p22p12ZGip12p22_sub.matrix, p22);
  }
  
  


  //-- step 3 --//
 
  gsl_matrix                *ZGiz10 = gsl_matrix_alloc(nr1_size+1, d_size);
  gsl_matrix          *p22p12ZGiz10 = gsl_matrix_alloc(nr2_size+1, d_size);
  gsl_matrix          *ZGip12p22z20 = gsl_matrix_alloc(nr1_size+1, d_size);
  gsl_matrix *p22p22p12ZGip12p22z20 = gsl_matrix_alloc(nr2_size+1, d_size);

  for (size_t i = 0; i < n_size; ++i) {
    gsl_matrix_view                ZGi_sub = gsl_matrix_submatrix(ZGi,                0, i*(nr1_size+1), nr1_size+1, nr1_size+1); 
    gsl_matrix_view          ZGip12p22_sub = gsl_matrix_submatrix(ZGip12p22,          0, i*(nr2_size+1), nr1_size+1, nr2_size+1); 
    gsl_matrix_view p22p22p12ZGip12p22_sub = gsl_matrix_submatrix(p22p22p12ZGip12p22, 0, i*(nr2_size+1), nr2_size+1, nr2_size+1); 
    
    gsl_matrix_set_zero(ZGiz10);
    gsl_matrix_set_zero(p22p12ZGiz10);
    gsl_matrix_set_zero(ZGip12p22z20);
    gsl_matrix_set_zero(p22p22p12ZGip12p22z20);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, &ZGi_sub.matrix,                Z10, 0.0, ZGiz10);
    gsl_blas_dgemm(CblasTrans,   CblasTrans, 1.0, &ZGip12p22_sub.matrix,          Z10, 0.0, p22p12ZGiz10);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, &ZGip12p22_sub.matrix,          Z20, 0.0, ZGip12p22z20);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, &p22p22p12ZGip12p22_sub.matrix, Z20, 0.0, p22p22p12ZGip12p22z20);    

    gsl_matrix_view  final_sub = gsl_matrix_submatrix(final, 0, i*d_size, d_size, d_size); 
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,  -1.0, R0invZ10, ZGiz10,                1.0, &final_sub.matrix );
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,   1.0, R0invZ20, p22p12ZGiz10,          1.0, &final_sub.matrix );
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,   1.0, R0invZ10, ZGip12p22z20,          1.0, &final_sub.matrix );
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,  -1.0, R0invZ20, p22p22p12ZGip12p22z20, 1.0, &final_sub.matrix );

    gsl_matrix_view  M_sub = gsl_matrix_submatrix(M, 0, i*d_size, nr1_size+1, d_size);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans,  1.0, &ZGi_sub.matrix,       R0invZ10, 1.0, &M_sub.matrix);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, -1.0, &ZGip12p22_sub.matrix, R0invZ20, 1.0, &M_sub.matrix);

    gsl_matrix_view  N_sub = gsl_matrix_submatrix(N, 0, i*(nr2_size+1), d_size, nr2_size+1); 
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,  1.0, R0invZ10, &ZGip12p22_sub.matrix,          1.0, &N_sub.matrix);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, R0invZ20, &p22p22p12ZGip12p22_sub.matrix, 1.0, &N_sub.matrix);
  }
  gsl_matrix_free(ZGiz10);
  gsl_matrix_free(p22p12ZGiz10);
  gsl_matrix_free(ZGip12p22z20);
  gsl_matrix_free(p22p22p12ZGip12p22z20);


  //-- free --//
  gsl_matrix_free( R0invZ10 );
  gsl_matrix_free( R0invZ20 );
  gsl_matrix_free( p11 );
  gsl_matrix_free( p12 );
  gsl_matrix_free( p22 );
  gsl_matrix_free( p12p22 );
  gsl_matrix_free( zrzp );

  gsl_matrix_free( ZGip12p22 );
 
  return;
}


void Calc_F1(const gsl_matrix *X, const gsl_matrix *R0_inv, const gsl_matrix *final,
             gsl_matrix *F1 )
{
  size_t dn_size = X->size1;
  size_t  p_size = X->size2;
  size_t d_size  = R0_inv->size1;
  size_t n_size  = dn_size / d_size;
  
  gsl_matrix *finalR0inv = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix   *M = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix  *XM = gsl_matrix_alloc(p_size, d_size);
  gsl_matrix *XMX = gsl_matrix_alloc(p_size, p_size);
  gsl_matrix_set_zero( XMX );

  for (size_t i = 0; i < n_size; ++i) {
    gsl_matrix_set_zero( finalR0inv );
    gsl_matrix_const_view final_sub = gsl_matrix_const_submatrix(final, 0, i*d_size, d_size, d_size);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &final_sub.matrix, R0_inv, 1.0, finalR0inv);

    gsl_matrix_set_zero( M );
    gsl_matrix_add(M, R0_inv );
    gsl_matrix_add(M, finalR0inv );

    gsl_matrix_set_zero( XM );
    gsl_matrix_const_view X_sub = gsl_matrix_const_submatrix(X, i*d_size, 0, d_size, p_size);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &X_sub.matrix, M, 0.0, XM);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, XM, &X_sub.matrix, 1.0, XMX);
  }

  
  gsl_matrix_set_zero( F1 );
  int sig;
  gsl_permutation *pmt = gsl_permutation_alloc(p_size);
  LUDecomp(XMX, pmt, &sig);
  LUInvert(XMX, pmt, F1);

  gsl_matrix_free( finalR0inv );
  gsl_matrix_free( M );
  gsl_matrix_free( XM );
  gsl_matrix_free( XMX );
  gsl_permutation_free(pmt);

  return;
}


void Calc_Ti_TTi_TTTi(const gsl_vector *eval, 
                      const gsl_matrix *ZGi, const gsl_matrix *p22p22p12ZGip12p22,
                      const gsl_matrix *X,    const gsl_matrix *F1, 
                      const gsl_matrix *M,    const gsl_matrix *N,
                      const gsl_matrix *final, const gsl_matrix *R0,
                  gsl_matrix *Ifinal, 
                  gsl_matrix *Ti, gsl_matrix *TTi, gsl_matrix *TTTi )
{
  size_t  n_size = eval->size;
  size_t nr1_size = ZGi->size1 - 1;
  size_t nr2_size = p22p22p12ZGip12p22->size1 - 1;

  size_t dn_size = X->size1;
  size_t  p_size = X->size2;
  size_t  d_size = dn_size / n_size;
    
  gsl_matrix_set_zero( Ti );
  gsl_matrix_set_zero( TTi );
  gsl_matrix_set_zero( TTTi );


  //-- Ifinal --//
  gsl_matrix_set_zero( Ifinal );
  gsl_matrix_add(Ifinal, final);

  gsl_matrix *I = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix_set_identity(I);
  for (size_t i = 0; i < n_size; ++i) {
    gsl_matrix_view Ifinal_i = gsl_matrix_submatrix(Ifinal, 0, i*d_size, d_size, d_size);
    gsl_matrix_add(&Ifinal_i.matrix, I);
  }

  //-- XF1X_diag --//
  gsl_matrix *XF1X_diag = gsl_matrix_alloc(d_size, dn_size);
  gsl_matrix_set_zero( XF1X_diag );
  
  gsl_matrix *XiF1 = gsl_matrix_alloc(d_size, p_size); 
  for (size_t i = 0; i < n_size; ++i) {
    gsl_matrix_view XF1X_diag_i = gsl_matrix_submatrix(XF1X_diag, 0, i*d_size, d_size, d_size);
    gsl_matrix_const_view X_i = gsl_matrix_const_submatrix(X, i*d_size, 0, d_size, p_size);
    gsl_matrix_set_zero( XiF1 );
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &X_i.matrix, F1,           0.0, XiF1 );
    gsl_blas_dgemm(CblasNoTrans, CblasTrans,   1.0, XiF1,        &X_i.matrix,  0.0, &XF1X_diag_i.matrix );
  }
  gsl_matrix_free(XiF1);

  //-- B, Ti --//
  gsl_matrix *B = gsl_matrix_alloc( nr1_size+1, (nr1_size+1)*n_size );
  gsl_matrix_set_zero( B );
  gsl_matrix_add(B, ZGi);

  gsl_matrix *M_XFX = gsl_matrix_alloc(d_size, dn_size);
  gsl_matrix_set_zero( M_XFX );
  for (size_t i = 0; i < n_size; ++i) {
    gsl_matrix_view B_sub = gsl_matrix_submatrix(B, 0, i*(nr1_size+1), nr1_size+1, nr1_size+1);

    gsl_matrix_const_view M_sub = gsl_matrix_const_submatrix(M, 0, i*(d_size), nr1_size+1, d_size);
    gsl_matrix_view XFX_sub = gsl_matrix_submatrix(XF1X_diag, 0, i*d_size, d_size, d_size);
    gsl_matrix_view M_XFX_sub = gsl_matrix_submatrix(M_XFX, 0, i*(d_size), nr1_size+1, d_size);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &M_sub.matrix,  &XFX_sub.matrix, 0.0, &M_XFX_sub.matrix );
    gsl_blas_dgemm(CblasNoTrans, CblasTrans,   1.0, &M_XFX_sub.matrix, &M_sub.matrix, 1.0, &B_sub.matrix );

    double delta_l = gsl_vector_get(eval, i);
    gsl_matrix_scale( &B_sub.matrix, 1.0/delta_l );
    gsl_matrix_add(Ti, &B_sub.matrix);
  }
  gsl_matrix_free( M_XFX );
  gsl_matrix_free( B );


  //-- J, TTi --//
  gsl_matrix *J = gsl_matrix_alloc( nr2_size+1, (nr2_size+1)*n_size );
  gsl_matrix_set_zero( J );
  gsl_matrix_add(J, p22p22p12ZGip12p22);

  gsl_matrix *N_XFX = gsl_matrix_alloc(nr2_size+1, dn_size);
  gsl_matrix_set_zero( N_XFX );

  for (size_t i = 0; i < n_size; ++i) {
    gsl_matrix_view J_sub = gsl_matrix_submatrix(J, 0, i*(nr2_size+1), nr2_size+1, nr2_size+1);
    
    gsl_matrix_const_view N_sub = gsl_matrix_const_submatrix(N, 0, i*(nr2_size+1), d_size, nr2_size+1);
    gsl_matrix_view XFX_sub = gsl_matrix_submatrix(XF1X_diag, 0, i*d_size, d_size, d_size);
    gsl_matrix_view N_XFX_sub = gsl_matrix_submatrix(N_XFX, 0, i*d_size, nr2_size+1, d_size);
    
    gsl_blas_dgemm(CblasTrans,   CblasNoTrans, 1.0, &N_sub.matrix,  &XFX_sub.matrix, 0.0, &N_XFX_sub.matrix );
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &N_XFX_sub.matrix, &N_sub.matrix, 1.0, &J_sub.matrix );
    gsl_matrix_add(TTi, &J_sub.matrix);
  }
  gsl_matrix_free( N_XFX );
  gsl_matrix_free( J );


  //-- TTTi --//
  gsl_matrix *part12 = gsl_matrix_alloc(d_size, d_size);
  for (size_t i = 0; i < n_size; ++i) {
    gsl_matrix_view XFX_sub = gsl_matrix_submatrix(XF1X_diag, 0, i*d_size, d_size, d_size);
    gsl_matrix_view Ifinal_sub = gsl_matrix_submatrix(Ifinal, 0, i*d_size, d_size, d_size);

    gsl_matrix_set_zero( part12 );
    gsl_blas_dgemm(CblasTrans,   CblasNoTrans, 1.0, &Ifinal_sub.matrix,  &XFX_sub.matrix, 0.0, part12 );

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, part12, &Ifinal_sub.matrix, 1.0, TTTi );

    gsl_matrix_const_view final_sub = gsl_matrix_const_submatrix(final, 0, i*d_size, d_size, d_size);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, R0, &final_sub.matrix, 1.0, TTTi );
  }
  gsl_matrix_free( part12 );

  gsl_matrix_free( XF1X_diag );
  gsl_matrix_free(I);
  return;
}


void Calc_ei(const gsl_vector *eval, const gsl_vector *beta, const gsl_vector *ai, const gsl_vector *pi, 
             const gsl_matrix *Uk, const gsl_matrix *W0, const gsl_matrix *Y0,
             const gsl_matrix *Z10, const gsl_matrix *Z20,
             gsl_vector *ei )
{
  

  size_t    n_size = eval->size;
  size_t  nr1_size = Z10->size2 - 1;
  size_t  nr2_size = Z20->size2 - 1;

  size_t dn_size = W0->size1;
  size_t  p_size = W0->size2;
  size_t  d_size = dn_size / n_size;


  gsl_matrix *Ai = gsl_matrix_alloc(n_size*(nr1_size+1), 1);
  gsl_matrix_set_zero(Ai);
  gsl_vector_view Ai_0 = gsl_matrix_column(Ai, 0);
  gsl_vector_memcpy(&Ai_0.vector, ai);

  gsl_matrix *Pi = gsl_matrix_alloc(n_size*(nr2_size+1), 1);
  gsl_matrix_set_zero(Pi);
  gsl_vector_view Pi_0 = gsl_matrix_column(Pi, 0);
  gsl_vector_memcpy(&Pi_0.vector, pi);

  gsl_matrix *Z1u1_i = gsl_matrix_alloc(d_size, 1);
  gsl_matrix *Z2u2_i = gsl_matrix_alloc(d_size, 1);
  gsl_matrix *Ei = gsl_matrix_alloc(dn_size, 1);
  gsl_matrix_set_zero(Ei);

  gsl_matrix *Beta = gsl_matrix_alloc(p_size, 1);
  gsl_matrix_set_zero(Beta);
  gsl_vector_view Beta_0 = gsl_matrix_column(Beta, 0);
  gsl_vector_memcpy(&Beta_0.vector, beta);

  gsl_matrix *miu = gsl_matrix_alloc(d_size, 1);

  for (size_t i = 0; i < n_size; ++i) {
      gsl_matrix_const_view Yi_sub = gsl_matrix_const_submatrix(Y0, i*d_size, 0, d_size, 1);
      gsl_matrix_const_view Xi_sub = gsl_matrix_const_submatrix(W0, i*d_size, 0, d_size, p_size);
      gsl_matrix_view Ei_sub = gsl_matrix_submatrix(Ei, i*d_size, 0, d_size, 1);

      gsl_matrix_set_zero(miu);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &Xi_sub.matrix, Beta, 0.0, miu );

      gsl_matrix_set_zero( Z1u1_i );
      gsl_matrix_set_zero( Z2u2_i );
      gsl_matrix_view ai_i = gsl_matrix_submatrix(Ai, i*(nr1_size+1), 0, nr1_size+1, 1 );
      gsl_matrix_view pi_i = gsl_matrix_submatrix(Pi, i*(nr2_size+1), 0, nr2_size+1, 1 );
      gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, 1.0, Z10, &ai_i.matrix, 1.0, Z1u1_i );
      gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, 1.0, Z20, &pi_i.matrix, 1.0, Z2u2_i );

      gsl_matrix_add(&Ei_sub.matrix, &Yi_sub.matrix);
      gsl_matrix_sub(&Ei_sub.matrix, miu);
      gsl_matrix_sub(&Ei_sub.matrix, Z1u1_i);
      gsl_matrix_sub(&Ei_sub.matrix, Z2u2_i);
  }
  
  
  //--- for ei ---//
  gsl_vector_set_zero(ei);
  
  gsl_vector *ttt = gsl_vector_alloc(d_size);

  for (size_t j = 0; j < n_size; ++j) {
     gsl_vector_view ei_j = gsl_vector_subvector(ei, j*d_size, d_size);

     for (size_t i = 0; i < n_size; ++i) {
       double uij = gsl_matrix_get(Uk, i, j);
       gsl_vector_view Ei_i = gsl_matrix_subcolumn(Ei, 0, i*d_size, d_size);

       gsl_vector_set_zero(ttt);
       gsl_vector_add(ttt, &Ei_i.vector);
       gsl_vector_scale(ttt, uij);
       gsl_vector_add(&ei_j.vector, ttt);
     }
  }
  gsl_vector_free(ttt);



  //-- free --//
  gsl_matrix_free( Ai );
  gsl_matrix_free( Pi );

  gsl_matrix_free( Z1u1_i );
  gsl_matrix_free( Z2u2_i );
  gsl_matrix_free( Ei );
  gsl_matrix_free( miu );
  gsl_matrix_free(Beta);

  return;
}

void Calc_A1( const gsl_vector *par_vg, const gsl_vector *par_vp, const gsl_vector *par_vr, 
              const gsl_matrix *R0_inv, const gsl_matrix *Z10, const gsl_matrix *Z20,
            const gsl_vector *Time, gsl_matrix *A1 )
{
  size_t d_size = par_vr->size - 1;
  size_t nr1_size = Z10->size2 - 1;
  size_t nr2_size = Z20->size2 - 1;
  
  size_t uniG_size = nr1_size+1;
  size_t uniP_size = nr2_size+1;
  size_t uniGP_size  = uniG_size + uniP_size;

  //-- A1,A2,A3: d*[d*uniGPR_size] --//
  gsl_matrix_set_zero( A1 );

  gsl_matrix  *D1jk = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
  gsl_matrix *Z1D1 = gsl_matrix_alloc(d_size, nr1_size+1);
  gsl_matrix  *Z1D1Z1 = gsl_matrix_alloc(d_size, d_size);
  
  gsl_matrix  *D2jk = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
  gsl_matrix *Z2D2 = gsl_matrix_alloc(d_size, nr2_size+1);
  gsl_matrix  *Z2D2Z2 = gsl_matrix_alloc(d_size, d_size);

  for (size_t j = 0; j < nr1_size+1; ++j) {
    for (size_t k = j; k <= j; ++k) {
      //-- part1 --//
      gsl_matrix_set_zero( D1jk );
      gsl_matrix_set_zero( Z1D1 );
      gsl_matrix_set_zero( Z1D1Z1 );

      gsl_matrix_set(D1jk, j, k, 1.0);
      gsl_matrix_set(D1jk, k, j, 1.0);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Z10,  D1jk, 0.0, Z1D1);
      gsl_blas_dgemm(CblasNoTrans, CblasTrans,   1.0, Z1D1, Z10,  0.0, Z1D1Z1);      
      if(j != k){ gsl_matrix_scale(Z1D1Z1, 0.5); }

      size_t xh1 = j; 
      gsl_matrix_view A1_sub = gsl_matrix_submatrix(A1, 0, xh1*d_size, d_size, d_size);
      gsl_matrix_add( &A1_sub.matrix, Z1D1Z1);
    }
  }

  for (size_t j = 0; j < nr2_size+1; ++j) {
    for (size_t k = j; k <= j; ++k) {
      //-- part1 --//
      gsl_matrix_set_zero( D2jk );
      gsl_matrix_set_zero( Z2D2 );
      gsl_matrix_set_zero( Z2D2Z2 );

      gsl_matrix_set(D2jk, j, k, 1.0);
      gsl_matrix_set(D2jk, k, j, 1.0);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Z20,  D2jk, 0.0, Z2D2);
      gsl_blas_dgemm(CblasNoTrans, CblasTrans,   1.0, Z2D2, Z20,  0.0, Z2D2Z2);
      if(j != k){ gsl_matrix_scale(Z2D2Z2, 0.5); }

      size_t xh1 = j; 
      gsl_matrix_view A1_sub = gsl_matrix_submatrix(A1, 0, (uniG_size+xh1)*d_size, d_size, d_size);
      gsl_matrix_add( &A1_sub.matrix, Z2D2Z2);
    }
  }


  //-- A1 for v1^2,v2^2,...,vd^2 --//  
  
  double phi = gsl_vector_get(par_vr, d_size);
  gsl_matrix  *A0 = gsl_matrix_alloc(d_size, d_size);
  
  for (size_t i = 0; i < d_size; ++i) {
	   gsl_matrix_view A1_sub = gsl_matrix_submatrix(A1, 0, (uniGP_size+i)*d_size, d_size, d_size); 
     double ti = i+1; 

     gsl_matrix_set_zero( A0 ); 
     for (size_t ii = i; ii < d_size; ++ii) {
       double tii = ii+1; 
       for (size_t jj = i; jj < d_size; ++jj) {
         double tjj = jj+1; 

         double tmp = pow(phi, abs(tii + tjj - 2*ti) );
         gsl_matrix_set(A0, ii,jj, tmp);
       }
     }

     gsl_matrix_add(&A1_sub.matrix, A0);
  }


  //-- A1 for phi --// 
  
  for (size_t i = 0; i <= (d_size-1); ++i) {
    gsl_matrix_view A1_sub = gsl_matrix_submatrix(A1, 0, (uniGP_size+d_size)*d_size, d_size, d_size); 
    double ti = i+1; 
    double visqr = gsl_vector_get(par_vr, i);

    gsl_matrix_set_zero( A0 ); 
    for (size_t ii = i; ii < d_size; ++ii) {
      double tii = ii+1;
      for (size_t jj = i; jj < d_size; ++jj) {
        double tjj = jj+1; 

        double tmp = abs(tii + tjj - 2*ti) * pow(phi, abs(tii + tjj - 2*ti)-1);
        gsl_matrix_set(A0, ii,jj, tmp);
      }
    }
    gsl_matrix_scale(A0, visqr);
    
    gsl_matrix_add(&A1_sub.matrix, A0);
  }

  
 

  gsl_matrix_free( D1jk );
  gsl_matrix_free( Z1D1 );
  gsl_matrix_free( Z1D1Z1 );
  
  gsl_matrix_free( D2jk );
  gsl_matrix_free( Z2D2 );
  gsl_matrix_free( Z2D2Z2 );

  gsl_matrix_free( A0 ); 
}

void Calc_A11( const gsl_vector *par_vr, const gsl_vector *Time, gsl_matrix *A11 )
{
  gsl_matrix_set_zero(A11);
  size_t d_size = par_vr->size - 1;
  

  gsl_matrix *A0 = gsl_matrix_alloc(d_size, d_size);
  double phi = gsl_vector_get(par_vr, d_size);

  //-- A11 (the front d blocks) --//
  for (size_t i = 0; i < d_size; ++i) {
    gsl_matrix_view A11_sub = gsl_matrix_submatrix(A11, 0, i*d_size, d_size, d_size);
    double ti = i+1;

    gsl_matrix_set_zero( A0 );
    for (size_t ii = i; ii < d_size; ++ii) {
      double tii = ii+1; 
      for (size_t jj = i; jj < d_size; ++jj) {
        double tjj = jj+1; 
        double tmp = abs(tii + tjj - 2*ti) * pow(phi, abs(tii + tjj - 2*ti)-1);
        gsl_matrix_set(A0, ii,jj, tmp);
      }
    }
    
    gsl_matrix_add(&A11_sub.matrix, A0);
  }


  //-- A11 (the (d+1)th block) --//
  gsl_matrix_view A11_sub = gsl_matrix_submatrix(A11, 0, d_size*d_size, d_size, d_size); 
  
  for (size_t i = 0; i < (d_size-1); ++i) {
    double ti = i+1; //gsl_vector_get(Time, i);
    double visqr = gsl_vector_get(par_vr, i);

    gsl_matrix_set_zero( A0 ); 
    for (size_t ii = i; ii < d_size; ++ii) {
      double tii = ii+1; 
      for (size_t jj = i; jj < d_size; ++jj) {
        double tjj = jj+1; 
        double tmp = abs(tii + tjj - 2*ti)* (abs(tii + tjj - 2*ti)-1) * pow(phi, abs(tii + tjj - 2*ti)-2);
        gsl_matrix_set(A0, ii,jj, tmp);
      }
    }
    gsl_matrix_scale(A0, visqr);
    
    gsl_matrix_add(&A11_sub.matrix, A0);
  }

  gsl_matrix_free( A0 );
}



void Calc_fd_GPR_test0(const gsl_vector *par_vr, 
                       const gsl_matrix *G0, const gsl_matrix *P0, const gsl_matrix *R0, 
                       const gsl_matrix *G0_inv, const gsl_matrix *P0_inv, const gsl_matrix *R0_inv, 
                       const gsl_matrix *Z10, const gsl_matrix *Z20, const gsl_vector *eval, 
                       const gsl_matrix *X, const gsl_matrix *Y, const gsl_vector *Time, 
                       const gsl_matrix *A1, 
                gsl_matrix *Qi, gsl_matrix *Hi_all, gsl_matrix *Py_all, 
                gsl_matrix *sgm_Pii, gsl_matrix *sgm_dk_Pii,
                gsl_matrix *fd_GPR )
{
  size_t d_size = R0->size1;
  size_t dn_size = X->size1;
  size_t p_size = X->size2;
  size_t n_size = dn_size / d_size;
  size_t nr1_size = G0->size1-1;
  size_t nr2_size = P0->size1-1;

  size_t uniG_size = nr1_size+1;
  size_t uniP_size = nr2_size+1;
  size_t uniR_size = d_size + 1;
  size_t uniGP_size  = uniG_size + uniP_size;


 
  double logdet_H = 0.0, logdet_Q = 0.0;
  CalcHiQi(eval, Time, X, Z10, Z20, G0, P0, R0,
           Hi_all, Qi, logdet_H, logdet_Q );

  gsl_matrix *XHi_all = gsl_matrix_alloc(p_size, dn_size);
  Calc_XHi_all(X, Hi_all, XHi_all); 

  gsl_matrix *Hiy_all = gsl_matrix_alloc(d_size, n_size);
  Calc_Hiy_all(Y, Hi_all, Hiy_all);

  gsl_vector *XHiy   = gsl_vector_alloc(p_size);  
  Calc_XHiy(Y, XHi_all, Time, XHiy); 

  gsl_vector *QiXHiy = gsl_vector_alloc(p_size);
  gsl_vector_set_zero( QiXHiy );
  gsl_blas_dgemv(CblasNoTrans, 1.0, Qi, XHiy, 0.0, QiXHiy);
  
  
  gsl_matrix_set_zero( Py_all );
  for (size_t i = 0; i < n_size; ++i) {
    gsl_vector_view Hiy_sub = gsl_matrix_column(Hiy_all, i);
    gsl_matrix_view XHi_sub = gsl_matrix_submatrix(XHi_all, 0, i*d_size, p_size, d_size);

    gsl_vector_view Py_sub = gsl_matrix_column(Py_all, i);
    gsl_vector_add( &Py_sub.vector, &Hiy_sub.vector);
    gsl_blas_dgemv(CblasTrans, -1.0, &XHi_sub.matrix, QiXHiy, 1.0, &Py_sub.vector);
  }

  
  gsl_matrix_set_zero(sgm_Pii);
  gsl_matrix_set_zero(sgm_dk_Pii);

  gsl_matrix *sgm_tmp = gsl_matrix_alloc(d_size,d_size);
  gsl_matrix *VXQ = gsl_matrix_alloc(d_size,p_size);
  for (size_t i = 0; i < n_size; ++i) {
    gsl_matrix_view  Vinv_sub = gsl_matrix_submatrix(Hi_all,  0, i * d_size,  d_size,d_size);

    gsl_matrix_set_zero( VXQ );
    gsl_matrix_view XHi_sub = gsl_matrix_submatrix(XHi_all, 0, i*d_size, p_size, d_size);
    gsl_blas_dgemm(CblasTrans,   CblasNoTrans,  1.0, &XHi_sub.matrix, Qi, 0.0, VXQ);

    gsl_matrix_add( sgm_Pii, &Vinv_sub.matrix );
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, VXQ, &XHi_sub.matrix, 1.0, sgm_Pii);

    double dk = gsl_vector_get(eval,i);
    gsl_matrix_set_zero( sgm_tmp );
    gsl_matrix_add( sgm_tmp, &Vinv_sub.matrix );
    gsl_matrix_scale( sgm_tmp, dk );
    gsl_matrix_add( sgm_dk_Pii, sgm_tmp );
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -dk, VXQ, &XHi_sub.matrix, 1.0, sgm_dk_Pii);
  }
  

  gsl_matrix *fd_G = gsl_matrix_alloc(uniG_size,1);
  gsl_matrix *fd_P = gsl_matrix_alloc(uniP_size,1);
  gsl_matrix *fd_R = gsl_matrix_alloc(uniR_size,1);
  gsl_matrix_set_zero(fd_G);
  gsl_matrix_set_zero(fd_P);
  gsl_matrix_set_zero(fd_R);


  gsl_matrix *part1 = gsl_matrix_alloc(d_size, d_size);
  gsl_vector *A1_py = gsl_vector_alloc(d_size);

  for (size_t j = 0; j < uniG_size; ++j) {
      //-- part1 --//
      gsl_matrix_const_view A1_sub = gsl_matrix_const_submatrix(A1, 0, j*d_size, d_size, d_size);
      double fd = 0.0;
      //-- part1 --//
      gsl_matrix_set_zero( part1 );
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &A1_sub.matrix, sgm_dk_Pii, 0.0, part1);
      for (size_t s = 0; s < d_size; ++s) { fd = fd + gsl_matrix_get(part1, s, s);}

      //-- part2 --//
      for (size_t r = 0; r < n_size; ++r) {
        double tmp = 0.0;
        gsl_vector_set_zero( A1_py );

        double dk = gsl_vector_get(eval, r);
        gsl_vector_view Py_r = gsl_matrix_column(Py_all, r);
        
        gsl_blas_dgemv(CblasNoTrans, dk, &A1_sub.matrix, &Py_r.vector, 0.0, A1_py);
        gsl_blas_ddot( &Py_r.vector, A1_py, &tmp);
        fd = fd - tmp;
      }

      //-- part3 --//
      gsl_matrix_set(fd_G, j, 0, fd);
  }

  
  for (size_t j = 0; j < uniP_size; ++j) {
      //-- part1 --//
      gsl_matrix_const_view A1_sub = gsl_matrix_const_submatrix(A1, 0, (uniG_size+j)*d_size, d_size, d_size);
      double fd = 0.0;
      //-- part1 --//
      gsl_matrix_set_zero( part1 );
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &A1_sub.matrix, sgm_Pii, 0.0, part1);
      for (size_t s = 0; s < d_size; ++s) { fd = fd + gsl_matrix_get(part1, s, s);}

      //-- part2 --//
      for (size_t r = 0; r < n_size; ++r) {
        double tmp = 0.0;
        gsl_vector_set_zero( A1_py );

        gsl_vector_view Py_r = gsl_matrix_column(Py_all, r);
        gsl_blas_dgemv(CblasNoTrans, 1.0, &A1_sub.matrix, &Py_r.vector,  0.0, A1_py);
        gsl_blas_ddot( &Py_r.vector, A1_py, &tmp);
        fd = fd - tmp;
      }

      //-- part3 --//
      gsl_matrix_set(fd_P, j, 0, fd);
  }

  for (size_t j = 0; j < (d_size+1); ++j) {
     //-- part1 --//
     gsl_matrix_const_view A1_sub = gsl_matrix_const_submatrix(A1, 0, (uniGP_size+j)*d_size, d_size, d_size);
     double fd = 0.0;
     //-- part1 --//
     gsl_matrix_set_zero( part1 );
     gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &A1_sub.matrix, sgm_Pii, 0.0, part1 );
     for (size_t s = 0; s < d_size; ++s) { fd = fd + gsl_matrix_get(part1, s, s); }
    
     //-- part2 --//
     for (size_t r = 0; r < n_size; ++r) {
       double tmp = 0.0;
       gsl_vector_set_zero( A1_py );

       gsl_vector_view Py_r = gsl_matrix_column(Py_all, r);
       gsl_blas_dgemv(CblasNoTrans, 1.0, &A1_sub.matrix, &Py_r.vector,  0.0, A1_py);
       gsl_blas_ddot( &Py_r.vector, A1_py, &tmp);
       fd = fd - tmp;
     }

     //-- part3 --//
   
     gsl_matrix_set(fd_R, j, 0, fd);
  }

  gsl_matrix_set_zero( fd_GPR );
  for(size_t i =0; i < uniG_size; ++i){ gsl_matrix_set(fd_GPR, i, 0, gsl_matrix_get(fd_G,i,0)); }
  for(size_t i =0; i < uniP_size; ++i){ gsl_matrix_set(fd_GPR, uniG_size+i, 0, gsl_matrix_get(fd_P,i,0)); }
  for(size_t i =0; i < uniR_size; ++i){ gsl_matrix_set(fd_GPR, uniGP_size+i, 0, gsl_matrix_get(fd_R,i,0)); }

  

  //-- free --//
  
  gsl_matrix_free( XHi_all ); 
  gsl_matrix_free( Hiy_all );
  gsl_vector_free( XHiy );
  gsl_vector_free( QiXHiy );
  gsl_matrix_free( sgm_tmp );
  gsl_matrix_free( VXQ );

  gsl_matrix_free( fd_G );
  gsl_matrix_free( fd_P );
  gsl_matrix_free( fd_R );

  gsl_matrix_free( part1 );
  gsl_vector_free( A1_py );

  return;
}

void Calc_IAmat_test0(const gsl_vector *eval, const gsl_matrix *X, const gsl_matrix *A1, 
                      const gsl_matrix *Hi_all, const gsl_matrix *Py_all, const gsl_matrix *Qi,
                      const gsl_matrix *Z10, const gsl_matrix *Z20,
                gsl_matrix *IAmat )
{
  size_t n_size = eval->size;
  size_t dn_size = X->size1;
  size_t p_size = X->size2;
  size_t d_size = dn_size/n_size;
  
  size_t nr1_size = Z10->size2-1;
  size_t nr2_size = Z20->size2-1;

  size_t uniG_size = nr1_size+1;
  size_t uniP_size = nr2_size+1;
  size_t uniR_size = d_size + 1;
  size_t uniGP_size  = uniG_size + uniP_size;
  size_t uniGPR_size = uniG_size + uniP_size + uniR_size;  


  //-- part2 for fj_G, fj_P, fj_R --//
  gsl_matrix *fj_G = gsl_matrix_alloc(dn_size, uniG_size);
  gsl_matrix *fj_P = gsl_matrix_alloc(dn_size, uniP_size);
  gsl_matrix *fj_R = gsl_matrix_alloc(dn_size, uniR_size);
  gsl_matrix_set_zero(fj_G);
  gsl_matrix_set_zero(fj_P);
  gsl_matrix_set_zero(fj_R);


  for (size_t j = 0; j < uniG_size; ++j) {
      //-- part1 --//
      gsl_matrix_const_view A1_sub = gsl_matrix_const_submatrix(A1, 0, j*d_size, d_size, d_size);

      //-- part2 --//
      for (size_t r = 0; r < n_size; ++r) {
        double dk = gsl_vector_get(eval, r);
        gsl_vector_const_view Py_r = gsl_matrix_const_column(Py_all, r);

        gsl_vector_view fj_G_sub = gsl_matrix_subcolumn(fj_G, j, r * d_size, d_size);
        gsl_blas_dgemv(CblasNoTrans, dk, &A1_sub.matrix, &Py_r.vector,  1.0, &fj_G_sub.vector);
      }
  }


  for (size_t j = 0; j < uniP_size; ++j) {
      //-- part1 --//
      gsl_matrix_const_view A1_sub = gsl_matrix_const_submatrix(A1, 0, (uniG_size+j)*d_size, d_size, d_size);

      //-- part2 --//
      for (size_t r = 0; r < n_size; ++r) {
        gsl_vector_const_view Py_r = gsl_matrix_const_column(Py_all, r);

        gsl_vector_view fj_P_sub = gsl_matrix_subcolumn(fj_P, j, r * d_size, d_size);
        gsl_blas_dgemv(CblasNoTrans, 1.0, &A1_sub.matrix, &Py_r.vector,  1.0, &fj_P_sub.vector);        
      }
  }


  for (size_t j = 0; j < (d_size+1); ++j) {
     //-- part1 --//
     gsl_matrix_const_view A1_sub = gsl_matrix_const_submatrix(A1, 0, (uniGP_size+j)*d_size, d_size, d_size);

     for (size_t r = 0; r < n_size; ++r) {
       gsl_vector_const_view Py_r = gsl_matrix_const_column(Py_all, r);

       gsl_vector_view fj_R_sub = gsl_matrix_subcolumn(fj_R, j, r * d_size, d_size);
       gsl_blas_dgemv(CblasNoTrans, 1.0, &A1_sub.matrix, &Py_r.vector,  1.0, &fj_R_sub.vector);        
     }
  }


  //-- part3 for F --//
  gsl_matrix *F = gsl_matrix_alloc(dn_size, uniGPR_size);
  gsl_matrix_set_zero(F);
  gsl_matrix_view F_sub1 = gsl_matrix_submatrix(F, 0, 0,          dn_size, uniG_size);
  gsl_matrix_view F_sub2 = gsl_matrix_submatrix(F, 0, uniG_size,  dn_size, uniP_size);
  gsl_matrix_view F_sub3 = gsl_matrix_submatrix(F, 0, uniGP_size, dn_size, uniR_size);
  gsl_matrix_add(&F_sub1.matrix, fj_G);
  gsl_matrix_add(&F_sub2.matrix, fj_P);
  gsl_matrix_add(&F_sub3.matrix, fj_R);




  //-- FVinv --//

  gsl_matrix *FVinv = gsl_matrix_alloc(uniGPR_size, dn_size);
  gsl_matrix_set_zero(FVinv);
  for (size_t i = 0; i < uniGPR_size; ++i) {
    for (size_t j = 0; j < n_size; ++j) {
  	  gsl_matrix_view     F_sub = gsl_matrix_submatrix(F, j * d_size, i, d_size, 1);
  	  gsl_matrix_const_view  Vinv_sub = gsl_matrix_const_submatrix(Hi_all,  0, j * d_size,  d_size,d_size);

  	  gsl_matrix_view FVinv_sub = gsl_matrix_submatrix(FVinv, i, j * d_size, 1, d_size);
  	  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &F_sub.matrix, &Vinv_sub.matrix, 1.0, &FVinv_sub.matrix);
    }
  }

  //-- FVinvF --//

  gsl_matrix *FVinvF = gsl_matrix_alloc(uniGPR_size, uniGPR_size);
  gsl_matrix_set_zero(FVinvF);
  for (size_t i = 0; i < uniGPR_size; ++i) {
	gsl_matrix_view FVinv_sub = gsl_matrix_submatrix(FVinv, i, 0, 1, dn_size);
  
  	for (size_t j = 0; j < uniGPR_size; ++j) {
  	  gsl_matrix_view     F_sub = gsl_matrix_submatrix(F, 0, j, dn_size, 1);

  	  gsl_matrix_view FVinvF_sub = gsl_matrix_submatrix(FVinvF, i, j, 1, 1);	
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &FVinv_sub.matrix, &F_sub.matrix, 1.0, &FVinvF_sub.matrix);
    }
  }

  //-- FVinvX --//

  gsl_matrix *FVinvX = gsl_matrix_alloc(uniGPR_size, p_size);
  gsl_matrix_set_zero( FVinvX );
  for (size_t i = 0; i < uniGPR_size; ++i) {
	gsl_matrix_view FVinv_sub = gsl_matrix_submatrix(FVinv, i, 0, 1, dn_size);
  
  	for (size_t j = 0; j < p_size; ++j) {
  	  gsl_matrix_const_view  X_sub = gsl_matrix_const_submatrix(X, 0, j, dn_size, 1);

  	  gsl_matrix_view FVinvX_sub = gsl_matrix_submatrix(FVinvX, i, j, 1, 1);	
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &FVinv_sub.matrix, &X_sub.matrix, 0.0, &FVinvX_sub.matrix);
    }
  }


  //-- FVinvX_Qi_XVinvF --//

  gsl_matrix *FVinvX_Qi = gsl_matrix_alloc(uniGPR_size, p_size);
  gsl_matrix_set_zero( FVinvX_Qi );
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, FVinvX, Qi, 0.0, FVinvX_Qi );


  gsl_matrix *FVinvX_Qi_XVinvF = gsl_matrix_alloc(uniGPR_size, uniGPR_size);
  gsl_matrix_set_zero( FVinvX_Qi_XVinvF );
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, FVinvX_Qi, FVinvX, 0.0, FVinvX_Qi_XVinvF );



  //-- unit 2 parts --//
  gsl_matrix_set_zero(IAmat);
  gsl_matrix_add(IAmat, FVinvF);
  gsl_matrix_sub(IAmat, FVinvX_Qi_XVinvF);
  


  

  //-- free --//
  gsl_matrix_free(fj_G );
  gsl_matrix_free(fj_P );
  gsl_matrix_free(fj_R );

  gsl_matrix_free( F );
  gsl_matrix_free( FVinv );
  gsl_matrix_free( FVinvF );
  gsl_matrix_free( FVinvX );
  gsl_matrix_free( FVinvX_Qi );
  gsl_matrix_free( FVinvX_Qi_XVinvF );

  return;
}

void Calc_IAmat_test0_add33part( const gsl_vector *par_vr, const gsl_matrix *R0_inv, 
                                 const gsl_matrix *Py_all, const gsl_matrix *sgm_Pii, 
                const gsl_matrix *A11, 
                gsl_matrix *IAmat_addpart33 )
{
  size_t d_size = R0_inv->size1;
  size_t n_size = Py_all->size2;
 
  
 
  gsl_matrix *A1 = gsl_matrix_alloc(d_size, d_size*(d_size+1));
  gsl_matrix_set_zero(A1);
  gsl_matrix_memcpy(A1, A11);

  gsl_matrix_set_zero( IAmat_addpart33 );

  gsl_matrix *IAmat_addpart33_1 = gsl_matrix_alloc(d_size+1, d_size+1);
  gsl_matrix_set_zero( IAmat_addpart33_1 );
  
  gsl_vector  *d2V_Py = gsl_vector_alloc(d_size);
  for (size_t j = 0; j < (d_size+1); ++j) {
    gsl_matrix_view A1_sub = gsl_matrix_submatrix(A1, 0, j*d_size, d_size, d_size);

    double tmp_val = 0.0;
    double tmp_value = 0.0;
    for (size_t r = 0; r < n_size; ++r) {
      tmp_value = 0.0;
      gsl_vector_set_zero( d2V_Py );

      gsl_vector_const_view Py_r = gsl_matrix_const_column(Py_all, r);
      gsl_blas_dgemv(CblasNoTrans, 1.0, &A1_sub.matrix, &Py_r.vector, 0.0, d2V_Py);
      gsl_blas_ddot( &Py_r.vector, d2V_Py, &tmp_value);
      tmp_val = tmp_val + tmp_value;
    }

    gsl_matrix_set(IAmat_addpart33_1, j, d_size, -0.5*tmp_val);
    gsl_matrix_set(IAmat_addpart33_1, d_size, j, -0.5*tmp_val);  
  }


  gsl_matrix *IAmat_addpart33_2 = gsl_matrix_alloc(d_size+1, d_size+1);
  gsl_matrix_set_zero( IAmat_addpart33_2 );

  gsl_matrix  *Pd2V = gsl_matrix_alloc(d_size, d_size);
  for (size_t j = 0; j < (d_size+1); ++j) {
    gsl_matrix_set_zero( Pd2V );

    gsl_matrix_view A1_sub = gsl_matrix_submatrix(A1, 0, j*d_size, d_size, d_size);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, sgm_Pii, &A1_sub.matrix, 0.0, Pd2V );

    double tmp_val = 0.0;
    for (size_t r = 0; r < d_size; ++r) { tmp_val = tmp_val + gsl_matrix_get(Pd2V, r, r); }
    
    gsl_matrix_set(IAmat_addpart33_2, j, d_size, 0.5*tmp_val);
    gsl_matrix_set(IAmat_addpart33_2, d_size, j, 0.5*tmp_val);
  }


  gsl_matrix_add( IAmat_addpart33, IAmat_addpart33_1 );
  gsl_matrix_add( IAmat_addpart33, IAmat_addpart33_2 ); 
 

  gsl_matrix_free( A1 );

  gsl_matrix_free( IAmat_addpart33_1 );
  gsl_matrix_free( IAmat_addpart33_2 );
  
  gsl_vector_free( d2V_Py );
  gsl_matrix_free( Pd2V );

  return;
}




void Calc_fdGPR_IAmat(const gsl_vector *par_vg, const gsl_vector *par_vp, const gsl_vector *par_vr,
                    const gsl_vector *beta, const gsl_vector *ai, const gsl_vector *pi, 
                    const gsl_vector *Time, const gsl_vector *eval, const gsl_matrix *Uk, 
                    const gsl_matrix *X, const gsl_matrix *Y, 
                    const gsl_matrix *W0, const gsl_matrix *Y0, 
                    gsl_matrix *fd_GPR, gsl_matrix *IAmat,
                    gsl_matrix *TTTi, gsl_vector *ei )
{
    size_t uniG_size = par_vg->size;  
    size_t uniP_size = par_vp->size;
    size_t uniR_size = par_vr->size;
    size_t uniGP_size  = uniG_size + uniP_size;
    size_t uniGPR_size = uniG_size + uniP_size + uniR_size;
    size_t    d_size = Time->size;
    size_t    p_size = X->size2;
    size_t   dn_size = X->size1;
    size_t    n_size = dn_size / d_size;
    size_t  nr1_size = uniG_size - 1.0;
    size_t  nr2_size = uniP_size - 1.0;

    gsl_matrix_set_zero(fd_GPR);
    gsl_matrix_set_zero(IAmat);
    gsl_matrix_set_zero(TTTi);
    gsl_vector_set_zero(ei);
    
    //-- part1: basic variables --//
    gsl_matrix *Z10    = gsl_matrix_alloc(d_size, nr1_size+1);
    gsl_matrix *Z20    = gsl_matrix_alloc(d_size, nr2_size+1);
    gsl_matrix *G0     = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
    gsl_matrix *P0     = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
    gsl_matrix *R0     = gsl_matrix_alloc(d_size, d_size);
    gsl_matrix *G0_inv = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
    gsl_matrix *P0_inv = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
    gsl_matrix *R0_inv = gsl_matrix_alloc(d_size, d_size);
    printf("    func1: Calc_Z10_Z20_G0_G0inv_P0_P0inv_R0_R0inv \n");
    Calc_Z10_Z20_G0_G0inv_P0_P0inv_R0_R0inv( par_vg, par_vp, par_vr, Time, 
                                   Z10, Z20, G0, P0, R0, G0_inv, P0_inv, R0_inv );

    //-- part2: all temp variables --//
    gsl_matrix *ZGi = gsl_matrix_alloc(nr1_size+1, (nr1_size+1)*n_size);
    gsl_matrix *p22p22p12ZGip12p22 = gsl_matrix_alloc(nr2_size+1, (nr2_size+1)*n_size);
    gsl_matrix *final = gsl_matrix_alloc(d_size, dn_size);
    gsl_matrix *M = gsl_matrix_alloc(nr1_size+1, dn_size);
    gsl_matrix *N = gsl_matrix_alloc(d_size, (nr2_size+1)*n_size);
    printf("    func2: Calc_alltemp \n");
    Calc_alltemp( eval, Z10, Z20, G0, G0_inv, R0_inv, P0_inv, 
             ZGi, p22p22p12ZGip12p22, final, M, N );

    //-- part3: F1 --//
    gsl_matrix *F1 = gsl_matrix_alloc(p_size, p_size);
    printf("    func3: Calc_F1 \n");
    Calc_F1( X, R0_inv, final,
             F1 );

    //-- part4: Ti_TTi_TTTi --//
    gsl_matrix    *Ifinal = gsl_matrix_alloc(d_size, dn_size);
    gsl_matrix        *Ti = gsl_matrix_alloc(nr1_size+1, nr1_size+1); 
    gsl_matrix       *TTi = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
    //gsl_matrix      *TTTi = gsl_matrix_alloc(d_size, d_size);
    printf("    func4: Calc_Ti_TTi_TTTi \n");
    Calc_Ti_TTi_TTTi( eval, ZGi, p22p22p12ZGip12p22, X, F1, M, N, final, R0,
                  Ifinal, Ti, TTi, TTTi );

    //-- part5: Si_SSi_SSSi --//
   
    printf("    func6: Calc_ei \n");
    Calc_ei(eval, beta, ai, pi, Uk, W0, Y0, Z10, Z20,
            ei );
    
    //-- A1,A11 --//
    printf("    func7: Calc_A1 \n");
    gsl_matrix *A1 = gsl_matrix_alloc(d_size, d_size * uniGPR_size);
    gsl_matrix_set_zero( A1 );
    Calc_A1( par_vg, par_vp, par_vr, R0_inv, Z10, Z20, Time,
             A1 );

    gsl_matrix *A11 = gsl_matrix_alloc(d_size, d_size*(d_size+1) );
    gsl_matrix_set_zero( A11 );
    Calc_A11( par_vr, Time, A11 );

   

  
    printf("    func9: Calc_fdGPK_test0 \n");
    gsl_matrix *Qi     = gsl_matrix_alloc(p_size,  p_size);
    gsl_matrix *Hi_all = gsl_matrix_alloc(d_size, dn_size);
    gsl_matrix *Py_all = gsl_matrix_alloc(d_size,  n_size);
    gsl_matrix *sgm_Pii = gsl_matrix_alloc(d_size, d_size);
    gsl_matrix *sgm_dk_Pii = gsl_matrix_alloc(d_size, d_size);
    Calc_fd_GPR_test0( par_vr, G0, P0, R0, G0_inv, P0_inv, R0_inv, Z10, Z20, eval, X, Y, Time, A1,
                       Qi, Hi_all, Py_all, sgm_Pii, sgm_dk_Pii, fd_GPR );
    if(1 == 1){
      printf("    fd_GPR_test0: \n        "); 
      for(size_t i =0; i < uniG_size; ++i){ printf("%8.5f ", gsl_matrix_get(fd_GPR, i, 0) ); }
      printf("\n        "); 
      for(size_t i =0; i < uniP_size; ++i){ printf("%8.5f ", gsl_matrix_get(fd_GPR, uniG_size+i, 0) ); }
      printf("\n        "); 
      for(size_t i =0; i < uniR_size; ++i){ printf("%8.5f ", gsl_matrix_get(fd_GPR, uniG_size+uniP_size+i, 0) ); }
      printf("\n"); 
    }
  

    printf("    func10: Calc_IAmat_test0 \n");
    Calc_IAmat_test0( eval, X, A1, Hi_all, Py_all, Qi, Z10, Z20,
                      IAmat );

    printf("    func11: Calc_add33part \n");
    gsl_matrix *IAmat_addpart33 = gsl_matrix_alloc(d_size+1, d_size+1);
    gsl_matrix_set_zero( IAmat_addpart33 );
    Calc_IAmat_test0_add33part( par_vr, R0_inv, Py_all, sgm_Pii, A11,
                                IAmat_addpart33 );

    printf("    func12: IAmat + 33part \n");
    gsl_matrix_view IAmat_test33 = gsl_matrix_submatrix(IAmat, uniGP_size, uniGP_size, uniR_size, uniR_size);
    gsl_matrix_add( &IAmat_test33.matrix, IAmat_addpart33 );
    gsl_matrix_free( IAmat_addpart33 );

    

    //-- free --//
    gsl_matrix_free( Z10 );
    gsl_matrix_free( Z20 );
    gsl_matrix_free( G0 ); 
    gsl_matrix_free( P0 ); 
    gsl_matrix_free( R0 ); 
    gsl_matrix_free( G0_inv );
    gsl_matrix_free( P0_inv );
    gsl_matrix_free( R0_inv );

    gsl_matrix_free( ZGi ); 
    gsl_matrix_free( p22p22p12ZGip12p22 );
    gsl_matrix_free( final );
    gsl_matrix_free( M );
    gsl_matrix_free( N );

    gsl_matrix_free( F1 );

    gsl_matrix_free( Ifinal );
    gsl_matrix_free( Ti ); 
    gsl_matrix_free( TTi ); 
    

    gsl_matrix_free( A1 );
    gsl_matrix_free( A11 );
 

    gsl_matrix_free( Qi );
    gsl_matrix_free( Hi_all );
    gsl_matrix_free( Py_all );
    gsl_matrix_free( sgm_Pii );
    gsl_matrix_free( sgm_dk_Pii );
    
    return;
}




void Update_par_vgvpvr(const gsl_matrix *var_com_new, 
                       const size_t uniG_size, const size_t uniP_size, const size_t uniR_size,
                       gsl_vector *par_vg, gsl_vector *par_vp, gsl_vector *par_vr )
{
    

    for(size_t i = 0; i < uniG_size; ++i){
      double tmp1 = gsl_matrix_get(var_com_new, i, 0);
      gsl_vector_set(par_vg, i, tmp1);
    }
    for(size_t i = 0; i < uniP_size; ++i){
      double tmp2 = gsl_matrix_get(var_com_new, i + uniG_size, 0);
      gsl_vector_set(par_vp, i, tmp2);
    }
    for(size_t i = 0; i < uniR_size; ++i){
      double tmp3 = gsl_matrix_get(var_com_new, i + uniG_size + uniP_size, 0);
      gsl_vector_set(par_vr, i, tmp3);
    }
}



void Calc_flagpd(const gsl_vector *par_vg, const gsl_vector *par_vp, const gsl_vector *par_vr,
                 size_t &flag_pd )
{
  size_t nr1_size = par_vg->size - 1;
  size_t nr2_size = par_vp->size - 1;
  size_t   d_size = par_vr->size - 1;

    //-- G0 whether positive definite --//
    size_t flag_pd1 = 1.0;
    for (size_t i = 0; i < nr1_size+1; i++) {
      double d = gsl_vector_get(par_vg, i);
      if (d <= 0) { flag_pd1 = 0.0; }
    }

    

  //-- P0 whether positive definite --//
    size_t flag_pd2 = 1.0;
    for (size_t i = 0; i < nr2_size+1; i++) {
      double d = gsl_vector_get(par_vp, i);
      if (d <= 0) { flag_pd2 = 0.0; }
    }

    

  
  //-- R0 whether positive definite --//
    size_t flag_pd3 = 1.0;
    for (size_t i = 0; i < d_size; i++) {
      double d = gsl_vector_get(par_vr, i);
      if (d <= 0) { flag_pd3 = 0.0; } 
    }

   
  
    if(flag_pd1 == 1.0 && flag_pd2 == 1.0 && flag_pd3 == 1.0){ flag_pd = 1.0; }else{ flag_pd = 0.0; }
    
}

double Calc_ccparval(const gsl_matrix *delta, const gsl_matrix *var_com_update)
{
  
  double cc_par_val = 0.0;
  size_t uniGPR_size = delta->size1;
  
  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double fenzi = 0.0;
  double fenmu = 0.0;

  for(size_t i = 0; i < uniGPR_size; ++i){
    tmp1 = gsl_matrix_get(delta, i, 0);
    tmp2 = gsl_matrix_get(var_com_update, i, 0);
    fenzi = fenzi + tmp1 * tmp1;
    fenmu = fenmu + tmp2 * tmp2;
  }
  cc_par_val = sqrt(fenzi / fenmu);

  return cc_par_val;
}

double Calc_ccgraval(const gsl_matrix *fd_GPR, const gsl_matrix *var_com)
{
  
  double cc_gra_val = 0.0;
  size_t uniGPR_size = var_com->size1;
  
  double tmp1 = 0.0;
  double fenzi = 0.0;

  for(size_t i = 0; i < uniGPR_size; ++i){
    tmp1 = gsl_matrix_get(fd_GPR, i, 0);
    fenzi = fenzi + tmp1 * tmp1;
  }
  cc_gra_val = sqrt(fenzi) / uniGPR_size;

  return cc_gra_val;
}


void Calc2_logRL_beta_pval(const gsl_vector *eval, const gsl_vector *Time,
                     const gsl_matrix *X, const gsl_matrix *Y,                     
                     const gsl_vector *par_vg, const gsl_vector *par_vp, const gsl_vector *par_vr,         
                     const size_t npar_bsnp, 
                  double &log_RL, gsl_vector *beta, double &pvalue, gsl_vector *logRL_logL_2parts ) 
{
  size_t  dn_size = X->size1;
  size_t   p_size = X->size2; 
  size_t   n_size = eval->size; 
  size_t   d_size = dn_size / n_size;
  size_t nr1_size = par_vg->size - 1;
  size_t nr2_size = par_vp->size - 1;
  
  gsl_vector_set_zero( logRL_logL_2parts );

  //-- part1: basic variables --//
  gsl_matrix *Z10    = gsl_matrix_alloc(d_size, nr1_size+1);
  gsl_matrix *Z20    = gsl_matrix_alloc(d_size, nr2_size+1);
  gsl_matrix *G0     = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
  gsl_matrix *P0     = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
  gsl_matrix *R0     = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix *G0_inv = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
  gsl_matrix *P0_inv = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
  gsl_matrix *R0_inv = gsl_matrix_alloc(d_size, d_size);
  Calc_Z10_Z20_G0_G0inv_P0_P0inv_R0_R0inv( par_vg, par_vp, par_vr, Time, 
                                           Z10, Z20, G0, P0, R0, G0_inv, P0_inv, R0_inv );

  //-- Hi_all --//
  gsl_matrix *Hi_all = gsl_matrix_alloc(d_size, dn_size);
  gsl_matrix *Qi     = gsl_matrix_alloc(p_size, p_size);
  double logdet_H = 0.0, logdet_Q = 0.0;
  CalcHiQi(eval, Time, X, Z10, Z20, G0, P0, R0,
           Hi_all, Qi, logdet_H, logdet_Q );

  //-- yPy --//
  gsl_matrix *Hiy_all = gsl_matrix_alloc(d_size, n_size);
  gsl_matrix *XHi_all = gsl_matrix_alloc(p_size, dn_size);
  gsl_vector *XHiy   = gsl_vector_alloc(p_size);
  gsl_vector *QiXHiy = gsl_vector_alloc(p_size);
  
  Calc_Hiy_all(Y, Hi_all, Hiy_all);
  double yHiy = Calc_yHiy(Y, Hiy_all); 

  double yPy = 0.0;
  Calc_XHi_all(X, Hi_all, XHi_all); 
  Calc_XHiy(Y, XHi_all, Time, XHiy); 
  gsl_blas_dgemv(CblasNoTrans, 1.0, Qi, XHiy, 0.0, QiXHiy);
  gsl_blas_ddot(QiXHiy, XHiy, &yPy); 

  yPy = yHiy - yPy;


  gsl_matrix *XXt = gsl_matrix_alloc(p_size, p_size);
  gsl_matrix_set_zero( XXt );
  gsl_blas_dsyrk(CblasUpper, CblasTrans, 1.0, X, 0.0, XXt);
  for (size_t i = 0; i < p_size; ++i) { 
    for (size_t j = 0; j < i; ++j) { 
      gsl_matrix_set(XXt, i, j, gsl_matrix_get(XXt, j, i));
    }
  }
  int sig;
  gsl_permutation *pmt = gsl_permutation_alloc(p_size); 
  LUDecomp(XXt, pmt, &sig); 
  gsl_permutation_free(pmt);
  
  char func_name = 'R';
  double logl_const = 0.0;
  if (func_name == 'R' || func_name == 'r') {
    logl_const = 0 - 0.5 * (double)(n_size - p_size) * (double)d_size * log(2.0 * M_PI)
                 + 0.5 * LULndet(XXt);
    gsl_vector_set( logRL_logL_2parts, 0, logl_const );
    gsl_vector_set( logRL_logL_2parts, 2,  0-0.5 * (double)n_size * (double)d_size * log(2.0 * M_PI) );
  } else {
    logl_const = 0 - 0.5 * (double)n_size * (double)d_size * log(2.0 * M_PI);
  }
  gsl_matrix_free(XXt);

  
  if (func_name == 'R' || func_name == 'r') {
     log_RL = logl_const - 0.5 * logdet_H - 0.5 * logdet_Q - 0.5 * yPy;
     
     gsl_vector_set( logRL_logL_2parts, 1, - 0.5 * logdet_H - 0.5 * logdet_Q - 0.5 * yPy );
     gsl_vector_set( logRL_logL_2parts, 3, - 0.5 * logdet_H - 0.5 * yPy );
  } else {
     
     log_RL = - 0.5 * logdet_H - 0.5 * yPy; 
  }




  //-- pvalue --//
  gsl_vector_set_zero( beta );
  gsl_vector_memcpy(beta, QiXHiy);

  gsl_vector_view bsnp =
        gsl_vector_subvector(beta, p_size - npar_bsnp, npar_bsnp);  
  gsl_matrix_view var_bsnp =
        gsl_matrix_submatrix(Qi, p_size - npar_bsnp, p_size - npar_bsnp, npar_bsnp, npar_bsnp );

  gsl_matrix *var_bsnp_inv = gsl_matrix_alloc( npar_bsnp, npar_bsnp );
  gsl_matrix_set_zero( var_bsnp_inv );
  
  gsl_permutation *pmt2 = gsl_permutation_alloc( npar_bsnp );
  LUDecomp(&var_bsnp.matrix, pmt2, &sig);
  LUInvert(&var_bsnp.matrix, pmt2, var_bsnp_inv);
  gsl_permutation_free( pmt2 );
  

  double zval = 0.0;
  gsl_vector *part23 = gsl_vector_alloc( npar_bsnp );
  gsl_vector_set_zero( part23 );
  gsl_blas_dgemv(CblasNoTrans, 1.0, var_bsnp_inv, &bsnp.vector, 0.0, part23);
  gsl_blas_ddot(part23, &bsnp.vector, &zval);
  
  pvalue = gsl_cdf_chisq_Q(zval, (double)npar_bsnp );


  //-- free --//
  gsl_matrix_free( Z10 );
  gsl_matrix_free( Z20 );
  gsl_matrix_free( G0 );
  gsl_matrix_free( P0 );
  gsl_matrix_free( R0 );
  gsl_matrix_free( G0_inv );
  gsl_matrix_free( P0_inv );
  gsl_matrix_free( R0_inv );

  gsl_matrix_free(Hi_all);
  gsl_matrix_free(Qi);
  gsl_matrix_free(Hiy_all);
  gsl_matrix_free(XHi_all);
  gsl_vector_free(XHiy);
  gsl_vector_free(QiXHiy);
  
  gsl_matrix_free(var_bsnp_inv);
  gsl_vector_free(part23);

  return;
}


void Calc_aipi2(const gsl_vector *par_vg, const gsl_vector *par_vp, const gsl_vector *par_vr, const gsl_vector *beta, 
                const gsl_vector *eval, const gsl_matrix *Uk, const gsl_vector *Time, 
                const gsl_matrix *X, const gsl_matrix *Y, 
             gsl_vector *a2, gsl_vector *p2 )
{
    size_t uniG_size = par_vg->size;
    size_t uniP_size = par_vp->size;
    size_t    d_size = Time->size;
    size_t    p_size = X->size2;
    size_t   dn_size = X->size1;
    size_t    n_size = dn_size / d_size;
    size_t  nr1_size = uniG_size - 1;
    size_t  nr2_size = uniP_size - 1;

    gsl_vector_set_zero( a2 );
    gsl_vector_set_zero( p2 ); 


    //-- part1: basic variables --//
    gsl_matrix *Z10    = gsl_matrix_alloc(d_size, nr1_size+1);
    gsl_matrix *Z20    = gsl_matrix_alloc(d_size, nr2_size+1);
    gsl_matrix *G0     = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
    gsl_matrix *P0     = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
    gsl_matrix *R0     = gsl_matrix_alloc(d_size, d_size);
    gsl_matrix *G0_inv = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
    gsl_matrix *P0_inv = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
    gsl_matrix *R0_inv = gsl_matrix_alloc(d_size, d_size);
    Calc_Z10_Z20_G0_G0inv_P0_P0inv_R0_R0inv( par_vg, par_vp, par_vr, Time, 
                                             Z10, Z20, G0, P0, R0, G0_inv, P0_inv, R0_inv );

    //-- ZGZ, ZPZ --//
    gsl_matrix *ZG = gsl_matrix_alloc(d_size, nr1_size+1);
    gsl_matrix *ZGZ = gsl_matrix_alloc(d_size, d_size);
    gsl_matrix_set_zero(ZG);
    gsl_matrix_set_zero(ZGZ);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Z10, G0,  0.0, ZG );
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, ZG, Z10,  0.0, ZGZ );

    gsl_matrix *ZP = gsl_matrix_alloc(d_size, nr2_size+1);
    gsl_matrix *ZPZ = gsl_matrix_alloc(d_size, d_size);
    gsl_matrix_set_zero(ZP);
    gsl_matrix_set_zero(ZPZ);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Z20, P0,  0.0, ZP );
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, ZP, Z20,  0.0, ZPZ );

    //-- ZZZ1,ZZZ2 --//
    gsl_matrix *ZZ1 = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
    gsl_matrix *ZZ2 = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
    gsl_matrix *ZZ1_inv = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
    gsl_matrix *ZZ2_inv = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
    gsl_matrix *ZZZ1 = gsl_matrix_alloc(nr1_size+1, d_size);
    gsl_matrix *ZZZ2 = gsl_matrix_alloc(nr2_size+1, d_size);
    gsl_matrix_set_zero(ZZ1);
    gsl_matrix_set_zero(ZZ2);
    gsl_matrix_set_zero(ZZ1_inv);
    gsl_matrix_set_zero(ZZ2_inv);
    gsl_matrix_set_zero(ZZZ1);
    gsl_matrix_set_zero(ZZZ2);

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Z10, Z10,  0.0, ZZ1 );
    int sig;
    gsl_permutation *pmt1 = gsl_permutation_alloc(nr1_size + 1);
    LUDecomp(ZZ1, pmt1, &sig);
    LUInvert(ZZ1, pmt1, ZZ1_inv);
    gsl_permutation_free(pmt1);
    gsl_blas_dgemm(CblasTrans, CblasTrans, 1.0, ZZ1_inv, Z10,  0.0, ZZZ1 );

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Z20, Z20,  0.0, ZZ2 );
    gsl_permutation *pmt2 = gsl_permutation_alloc(nr2_size + 1);
    LUDecomp(ZZ2, pmt2, &sig);
    LUInvert(ZZ2, pmt2, ZZ2_inv);
    gsl_permutation_free(pmt2);
    gsl_blas_dgemm(CblasTrans, CblasTrans, 1.0, ZZ2_inv, Z20,  0.0, ZZZ2 );

    //-- Hi_all --//
    gsl_matrix *Hi_all = gsl_matrix_alloc(d_size, dn_size);
    gsl_matrix *Qi     = gsl_matrix_alloc(p_size, p_size);
    double logdet_H = 0.0, logdet_Q = 0.0;
    CalcHiQi(eval, Time, X, Z10, Z20, G0, P0, R0,
           Hi_all, Qi, logdet_H, logdet_Q );

    //-- (Y - Xb) --//
    gsl_vector *YXb = gsl_vector_alloc(dn_size);
    gsl_vector_set_zero( YXb );
    gsl_vector_const_view y = gsl_matrix_const_column(Y,0);
    gsl_vector_add(YXb, &y.vector);
    gsl_blas_dgemv(CblasNoTrans, -1.0, X, beta,  1.0, YXb );
    
    //-- --//
    gsl_vector *Ai = gsl_vector_alloc(dn_size);
    gsl_vector *Pi = gsl_vector_alloc(dn_size);
    gsl_vector_set_zero( Ai );
    gsl_vector_set_zero( Pi );

    gsl_matrix *mat_dd1 = gsl_matrix_alloc(d_size, d_size);
    gsl_matrix *mat_dd2 = gsl_matrix_alloc(d_size, d_size);
    for (size_t i = 0; i < n_size; ++i) {
      gsl_matrix_set_zero( mat_dd1 );
      gsl_matrix_set_zero( mat_dd2 );
      double delta = gsl_vector_get(eval, i);
      gsl_matrix_view Hi_sub = gsl_matrix_submatrix(Hi_all, 0, i*d_size, d_size, d_size);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, delta, ZGZ, &Hi_sub.matrix, 0.0, mat_dd1 );
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,   ZPZ, &Hi_sub.matrix, 0.0, mat_dd2 );

      gsl_vector_view Ai_sub = gsl_vector_subvector(Ai, i*d_size, d_size);
      gsl_vector_view Pi_sub = gsl_vector_subvector(Pi, i*d_size, d_size);
      gsl_vector_view YXb_sub = gsl_vector_subvector(YXb, i*d_size, d_size);
      gsl_blas_dgemv(CblasNoTrans, 1.0, mat_dd1, &YXb_sub.vector,  1.0, &Ai_sub.vector );
      gsl_blas_dgemv(CblasNoTrans, 1.0, mat_dd2, &YXb_sub.vector,  1.0, &Pi_sub.vector );
    }

    //-- --//
    gsl_vector *UAi = gsl_vector_alloc(dn_size);
    gsl_vector *UPi = gsl_vector_alloc(dn_size);
    gsl_vector *tmp1 = gsl_vector_alloc(d_size);
    gsl_vector *tmp2 = gsl_vector_alloc(d_size);

    gsl_vector_set_zero( UAi );
    gsl_vector_set_zero( UPi );
    for (size_t i = 0; i < n_size; ++i) {
      gsl_vector_view UAi_sub = gsl_vector_subvector(UAi, i*d_size, d_size);
      gsl_vector_view UPi_sub = gsl_vector_subvector(UPi, i*d_size, d_size);

      for (size_t j = 0; j < n_size; ++j) {
        double uij = gsl_matrix_get(Uk, i, j);
        gsl_vector_view Ai_sub = gsl_vector_subvector(Ai, j*d_size, d_size);
        gsl_vector_view Pi_sub = gsl_vector_subvector(Pi, j*d_size, d_size);

        gsl_vector_set_zero(tmp1);
        gsl_vector_add(tmp1, &Ai_sub.vector);
        gsl_vector_scale(tmp1, uij);
        gsl_vector_add(&UAi_sub.vector, tmp1);

        gsl_vector_set_zero(tmp2);
        gsl_vector_add(tmp2, &Pi_sub.vector);
        gsl_vector_scale(tmp2, uij);
        gsl_vector_add(&UPi_sub.vector, tmp2);
      }
    }

    //-- --//
    gsl_vector_set_zero( a2 );
    gsl_vector_set_zero( p2 );

    for (size_t i = 0; i < n_size; ++i) {
      gsl_vector_view a2_sub = gsl_vector_subvector(a2, i*(nr1_size+1), nr1_size+1);
      gsl_vector_view p2_sub = gsl_vector_subvector(p2, i*(nr2_size+1), nr2_size+1);

      gsl_vector_view UAi_sub = gsl_vector_subvector(UAi, i*d_size, d_size);
      gsl_vector_view UPi_sub = gsl_vector_subvector(UPi, i*d_size, d_size);

      gsl_blas_dgemv(CblasNoTrans, 1.0, ZZZ1, &UAi_sub.vector,  1.0, &a2_sub.vector );  
      gsl_blas_dgemv(CblasNoTrans, 1.0, ZZZ2, &UPi_sub.vector,  1.0, &p2_sub.vector );  
    }



    //-- free --//
    gsl_matrix_free( Z10 );
    gsl_matrix_free( Z20 );
    gsl_matrix_free( G0 );
    gsl_matrix_free( P0 );
    gsl_matrix_free( R0 );
    gsl_matrix_free( G0_inv );
    gsl_matrix_free( P0_inv );
    gsl_matrix_free( R0_inv );

    gsl_matrix_free(ZG);
    gsl_matrix_free(ZGZ);
    gsl_matrix_free(ZP);
    gsl_matrix_free(ZPZ);

    gsl_matrix_free(ZZ1);
    gsl_matrix_free(ZZ2);
    gsl_matrix_free(ZZ1_inv);
    gsl_matrix_free(ZZ2_inv);
    gsl_matrix_free(ZZZ1);
    gsl_matrix_free(ZZZ2);

    gsl_matrix_free( Hi_all );
    gsl_matrix_free( Qi  );

    
    gsl_vector_free( YXb );
    gsl_vector_free( Ai );
    gsl_vector_free( Pi );

    gsl_matrix_free( mat_dd1 );
    gsl_matrix_free( mat_dd2 );
    
    gsl_vector_free( UAi );
    gsl_vector_free( UPi );

    gsl_vector_free( tmp1 );
    gsl_vector_free( tmp2 );

    return;
}



void Calc_Hessian_inv(const gsl_matrix *trvpvp_mat, const gsl_matrix *IAmat, 
                      gsl_matrix *Hessian_inv )
{
  size_t n1_size = Hessian_inv->size1;
  size_t n2_size = Hessian_inv->size2;
  gsl_matrix_set_zero( Hessian_inv );

  gsl_matrix *Hessian = gsl_matrix_alloc(n1_size, n2_size); 
  gsl_matrix_set_zero( Hessian );
  gsl_matrix_set_zero( Hessian_inv );

  gsl_matrix_add(Hessian, trvpvp_mat);
  gsl_matrix_scale(Hessian, 0.5);
  gsl_matrix_sub(Hessian, IAmat);

  int sig;
  gsl_permutation *pmt2 = gsl_permutation_alloc( n1_size );
  LUDecomp(Hessian, pmt2, &sig);
  LUInvert(Hessian, pmt2, Hessian_inv);
  gsl_permutation_free( pmt2 );

  gsl_matrix_free(Hessian);
}




void Calc_Z10(const gsl_vector *Times,
              gsl_matrix *Z10 ) 
{
  size_t d_size = Times->size;
  size_t nr1_size = Z10->size2 - 1;
  gsl_matrix_set_zero(Z10);

  //-- legendre --//
  double tmin = gsl_vector_get(Times, 0);
  double tmax = gsl_vector_get(Times, d_size-1);

  gsl_vector *tx = gsl_vector_alloc(d_size);
  for (size_t i = 0; i < d_size; ++i) {
    double tmp = -1 + 2 * (gsl_vector_get(Times,i) - tmin)/(tmax - tmin);
    gsl_vector_set(tx, i, tmp);
  }

  for (size_t i = 0; i < d_size; ++i) {
    double tx_i = gsl_vector_get(tx, i);

    double phi0 = 1.0;
    double phi1 = tx_i; 
    double phi2 = 0.5 * (3*(tx_i*tx_i) - 1);
    double phi3 = 0.5 * (5*(tx_i*tx_i*tx_i) - 3*tx_i);
    double phi4 = 0.125 * (35*(tx_i*tx_i*tx_i*tx_i) - 30*(tx_i*tx_i) + 3);
    double phi5 = 0.125 * (63*(tx_i*tx_i*tx_i*tx_i*tx_i) - 70*(tx_i*tx_i*tx_i) + 15*tx_i);

    if(nr1_size >= 0){ gsl_matrix_set(Z10, i, 0, phi0); }
    if(nr1_size >= 1){ gsl_matrix_set(Z10, i, 1, phi1); }
    if(nr1_size >= 2){ gsl_matrix_set(Z10, i, 2, phi2); }
    if(nr1_size >= 3){ gsl_matrix_set(Z10, i, 3, phi3); }
    if(nr1_size >= 4){ gsl_matrix_set(Z10, i, 4, phi4); }
    if(nr1_size >= 5){ gsl_matrix_set(Z10, i, 5, phi5); }
  }

  gsl_vector_free( tx );
}

void LC_get_mu(const gsl_vector *par, const gsl_vector *Times, const size_t curve_num,
               gsl_vector *miu ) 
{
  size_t d_size = Times->size;
  gsl_vector_set_zero(miu);


  //-- gompertz --//
  if(curve_num == 1){ 
    double a = gsl_vector_get(par, 0);
    double b = gsl_vector_get(par, 1);  
    double k = gsl_vector_get(par, 2);

    for (size_t i = 0; i < d_size; ++i) {
      double tm = gsl_vector_get(Times, i);
      double res = a*exp( -b*exp(-k*tm) );
      gsl_vector_set(miu, i, res );
    }
  }
  

}

void LC_get_derv( const gsl_vector *par, const gsl_vector *Times, const size_t curve_num,
                  gsl_matrix *W_basic )
{
  
  size_t d_size = Times->size;
  gsl_matrix_set_zero( W_basic );

  //-- gompertz
  if(curve_num == 1){ 
    double a = gsl_vector_get(par, 0);
    double b = gsl_vector_get(par, 1);  
    double k = gsl_vector_get(par, 2);

    for (size_t i = 0; i < d_size; ++i) {
      double tm = gsl_vector_get(Times, i);
      
      double res1 = exp( -b*exp(-k*tm) );
      double res2 = a * exp(-b*exp(-k*tm)) * (-1.0) * exp(-k*tm);
      double res3 = a * exp(-b*exp(-k*tm)) * (-b) * exp(-k*tm) * (-1.0) * tm;

      gsl_matrix_set(W_basic, i, 0, res1);
      gsl_matrix_set(W_basic, i, 1, res2);
      gsl_matrix_set(W_basic, i, 2, res3);
    }
  }
  //-- end --//
}

void Build_W0YO( const size_t nrf, const gsl_vector *betas, const gsl_vector *a0, const gsl_vector *p0, 
                 const gsl_matrix *Pheno, const gsl_matrix *Qmat, const gsl_vector *Times, const size_t curve_num,
                 gsl_matrix *W0, gsl_matrix *Y0 )
{
  size_t p_size = W0->size2;
  size_t dn_size = Y0->size1;
  size_t d_size = Times->size;
  size_t n_size = dn_size / d_size;
  size_t nr1_size = a0->size / n_size - 1;
  size_t nr2_size = p0->size / n_size - 1;

  //-- ne --//
  size_t ne = 0.0;
  if(curve_num == 1){ ne = 3; }  //--gompertz  
  


  //-- W0 --//
  gsl_matrix_set_zero( W0 );

  //-- W_basic --//
  gsl_vector *par = gsl_vector_alloc(ne);
  gsl_matrix *W_basic = gsl_matrix_alloc(d_size, ne);
  gsl_matrix *Betas = gsl_matrix_alloc(p_size/ne, ne);
  gsl_matrix_set_zero( Betas );
  for (size_t i = 0; i < (p_size/ne); ++i) {
    gsl_vector_view Betas_i = gsl_matrix_row(Betas, i);
    gsl_vector_const_view betas_i = gsl_vector_const_subvector(betas, i*ne, ne);
    gsl_vector_add(&Betas_i.vector, &betas_i.vector);
  }

  for (size_t i = 0; i < n_size; ++i) {
    //-- par --//
    gsl_vector_set_zero( par );
    gsl_vector_const_view coefs = gsl_matrix_const_row(Qmat, i);
    gsl_blas_dgemv(CblasTrans, 1.0, Betas, &coefs.vector, 0.0, par);

    //-- W_basic --//
    gsl_matrix_set_zero( W_basic );
    LC_get_derv( par, Times, curve_num,  W_basic );

    for (size_t j = 0; j < (p_size/ne); ++j) {
      gsl_matrix_view  W0_sub = gsl_matrix_submatrix(W0, i*d_size, j*ne, d_size, ne);
      double coefs_j = gsl_vector_get(&coefs.vector, j);
      gsl_matrix_add(&W0_sub.matrix, W_basic);
      gsl_matrix_scale(&W0_sub.matrix, coefs_j);
    }
  }

  //-- Y0 --//
  gsl_matrix_set_zero( Y0 );

  gsl_matrix *Z10 = gsl_matrix_alloc(d_size, nr1_size+1);
  gsl_matrix *Z20 = gsl_matrix_alloc(d_size, nr2_size+1);
  gsl_matrix_set_zero( Z10 );
  gsl_matrix_set_zero( Z20 );
  Calc_Z10( Times, Z10 );
  Calc_Z10( Times, Z20 );

  gsl_vector *term0 = gsl_vector_alloc(dn_size);
  gsl_vector *term1 = gsl_vector_alloc(dn_size);
  gsl_vector *term2 = gsl_vector_alloc(dn_size);
 
  gsl_vector_set_zero(term0);
  gsl_vector_set_zero(term1);
  gsl_vector_set_zero(term2);
 

  gsl_vector *par_i = gsl_vector_alloc(ne);
  gsl_vector *miu_i = gsl_vector_alloc(d_size);
  for (size_t i = 0; i < n_size; ++i) {
    //-- term 0 --//
    gsl_vector_view term0_sub = gsl_vector_subvector(term0, i*d_size, d_size);
    gsl_vector_const_view Pheno_i = gsl_matrix_const_row(Pheno, i);
    gsl_vector_add(&term0_sub.vector, &Pheno_i.vector);

    //-- term 1 --//
    gsl_vector_view term1_sub = gsl_vector_subvector(term1, i*d_size, d_size);

    gsl_vector_set_zero( par_i );
    gsl_vector_set_zero( miu_i );
    gsl_vector_const_view coefs = gsl_matrix_const_row(Qmat, i);
    gsl_blas_dgemv(CblasTrans, 1.0, Betas, &coefs.vector, 0.0, par_i);
    LC_get_mu( par_i, Times, curve_num, &term1_sub.vector );

   
  }
  gsl_blas_dgemv(CblasNoTrans, 1.0, W0, betas, 0.0, term2);

  gsl_vector_view y0 = gsl_matrix_column(Y0,0);
  gsl_vector_add(&y0.vector, term0);
  gsl_vector_sub(&y0.vector, term1);
  gsl_vector_add(&y0.vector, term2);



  //-- free --//
  gsl_vector_free( par );
  gsl_matrix_free( W_basic );
  gsl_matrix_free( Betas );

  gsl_matrix_free( Z10 );
  gsl_matrix_free( Z20 );

  gsl_vector_free(term0);
  gsl_vector_free(term1);
  gsl_vector_free(term2);
  
  gsl_vector_free( par_i );
  gsl_vector_free( miu_i );
  //-- end --//
}

void Buid_XY( const gsl_matrix *W0, const gsl_matrix *Y0, const gsl_matrix *Uk, 
              gsl_matrix *X, gsl_matrix *Y) 
{
  size_t dn_size = Y0->size1;
  size_t n_size = Uk->size1;
  size_t p_size = W0->size2;
  size_t d_size = dn_size / n_size;

  //-- X = UtW0 --//
  //-- Y = UtY0 --//
  gsl_matrix_set_zero(X);
  gsl_matrix_set_zero(Y);

  gsl_matrix *temp  = gsl_matrix_alloc(d_size, p_size);
  gsl_matrix *temp2 = gsl_matrix_alloc(d_size, 1);

  for (size_t j = 0; j < n_size; ++j) {
    gsl_matrix_view X_sub = gsl_matrix_submatrix(X, j*d_size, 0, d_size, p_size);
    gsl_matrix_view Y_sub = gsl_matrix_submatrix(Y, j*d_size, 0, d_size, 1);

    for(size_t i = 0; i < n_size; ++i) {
      double uij = gsl_matrix_get(Uk, i, j);
      gsl_matrix_const_view W0_sub = gsl_matrix_const_submatrix(W0, i*d_size, 0, d_size, p_size);
      gsl_matrix_const_view Y0_sub = gsl_matrix_const_submatrix(Y0, i*d_size, 0, d_size, 1);

      gsl_matrix_memcpy(temp, &W0_sub.matrix);
      gsl_matrix_scale(temp, uij);
      gsl_matrix_add(&X_sub.matrix, temp);  

      gsl_matrix_memcpy(temp2, &Y0_sub.matrix);
      gsl_matrix_scale(temp2, uij);
      gsl_matrix_add(&Y_sub.matrix, temp2);
    }
  }

  gsl_matrix_free( temp );
  gsl_matrix_free( temp2 );
  return;
}

int check_beta( const gsl_matrix *Qmat, const gsl_vector *beta, const size_t curve_num, const size_t nrf )
{
  size_t ne_size = 0;
 
  if(curve_num == 1){ ne_size = 3; }  //--gompertz  

  size_t n_size = Qmat->size1;
  size_t c_size = Qmat->size2;
  size_t cne_size = beta->size;

  gsl_matrix *beta_mat = gsl_matrix_alloc(c_size, ne_size);
  gsl_matrix_set_zero(beta_mat);
  for(size_t i = 0; i < c_size; i++){ 
  	for(size_t j = 0; j < ne_size; j++){
       double tmp = gsl_vector_get(beta, i*ne_size+j);
       gsl_matrix_set(beta_mat, i, j, tmp);
  	}
  }

  gsl_matrix *Qb = gsl_matrix_alloc(n_size, ne_size);
  gsl_matrix_set_zero( Qb );
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Qmat, beta_mat, 0.0, Qb);
  
  int prb = 0;
  for(size_t i = 0; i < n_size; i++){ 
    if(curve_num == 1){ 
      double a = gsl_matrix_get(Qb,i,0); 
      double b = gsl_matrix_get(Qb,i,1); 
      double k = gsl_matrix_get(Qb,i,2); 
      if(a <= 0 || b <= 0 || k <=0 || k > 1){ prb = 1; i = n_size; }
    }
  }

  //--
  gsl_matrix_free( beta_mat );
  gsl_matrix_free( Qb );

  return prb;
}



void iteration( const size_t nrf, const size_t maxiter, const double cc_par, const double cc_gra, 
                     const size_t npar_bsnp, const size_t curve_num, 
                    const gsl_matrix *Pheno, const gsl_matrix *Qmat, 
                    const gsl_matrix *Uk, const gsl_vector *eval, const gsl_vector *Time, 
            const size_t h0, gsl_vector *par_vg, gsl_vector *par_vp, gsl_vector *par_vr,
			gsl_matrix *X, gsl_vector *beta, gsl_vector *ai, gsl_vector *pi, 
            double &log_RL, double &pvalue,  gsl_vector *logRL_logL_2parts )
{
  size_t ne_size = 0;
  
  if(curve_num == 1){ ne_size = 3; }  //--gompertz  
  
  size_t n_size = eval->size;
  size_t d_size = Time->size;
  size_t dn_size = d_size * n_size;
  size_t p_size = Qmat->size2 * ne_size;
  size_t uniG_size = par_vg->size;
  size_t uniP_size = par_vp->size;
  size_t uniR_size = par_vr->size;
  size_t uniGP_size  = uniG_size + uniP_size;
  size_t uniGPR_size = uniG_size + uniP_size + uniR_size;
  size_t nr1_size = uniG_size - 1;
  size_t nr2_size = uniP_size - 1;

  //-- define variables --//
  gsl_matrix *Z10    = gsl_matrix_alloc(d_size, nr1_size+1);
  gsl_matrix *Z20    = gsl_matrix_alloc(d_size, nr2_size+1);
  gsl_matrix *G0     = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
  gsl_matrix *P0     = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
  gsl_matrix *R0     = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix *G0_inv = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
  gsl_matrix *P0_inv = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
  gsl_matrix *R0_inv = gsl_matrix_alloc(d_size, d_size);
  
  gsl_matrix *fd_GPR   = gsl_matrix_alloc(uniGPR_size, 1);
  gsl_matrix *IAmat    = gsl_matrix_alloc(uniGPR_size, uniGPR_size);
  gsl_matrix *IA_inv   = gsl_matrix_alloc(uniGPR_size, uniGPR_size);
 
  
  
  gsl_matrix *delta          = gsl_matrix_alloc(uniGPR_size, 1);
  gsl_matrix *var_com        = gsl_matrix_alloc(uniGPR_size, 1);
  gsl_matrix *var_com_update = gsl_matrix_alloc(uniGPR_size, 1);
  
  gsl_matrix_set_zero( Z10 );  gsl_matrix_set_zero( Z20 );
  gsl_matrix_set_zero( G0 );  gsl_matrix_set_zero( G0_inv );
  gsl_matrix_set_zero( P0 );  gsl_matrix_set_zero( P0_inv );
  gsl_matrix_set_zero( R0 );  gsl_matrix_set_zero( R0_inv );
  gsl_matrix_set_zero( fd_GPR );
  gsl_matrix_set_zero( IAmat );
  gsl_matrix_set_zero( IA_inv );
 
 
  gsl_matrix_set_zero( delta );
  gsl_matrix_set_zero( var_com );
  gsl_matrix_set_zero( var_com_update );

  for(size_t i = 0; i < uniG_size; ++i){  gsl_matrix_set(var_com, i, 0, gsl_vector_get(par_vg, i) );  }
  for(size_t i = 0; i < uniP_size; ++i){  gsl_matrix_set(var_com, i + uniG_size, 0, gsl_vector_get(par_vp, i) );  }
  for(size_t i = 0; i < uniR_size; ++i){  gsl_matrix_set(var_com, i + uniGP_size, 0, gsl_vector_get(par_vr, i) );  }

  gsl_matrix *TTTi = gsl_matrix_alloc(d_size, d_size); 
  gsl_vector *ei = gsl_vector_alloc(dn_size);
  gsl_matrix_set_zero( TTTi );
  gsl_vector_set_zero( ei );

  gsl_matrix *W0 = gsl_matrix_alloc(dn_size, p_size);
  gsl_matrix *Y0 = gsl_matrix_alloc(dn_size, 1);
  gsl_matrix *Y = gsl_matrix_alloc(dn_size, 1);

  //-- below are raw code --//
  size_t flag_pd = 1;   
  size_t iter_count = 0;
  
 
  double cc_par_val = 10000.0;
  double cc_gra_val = 10000.0;
  double cc_logl_val0 = 10000.0;
  double cc_logl_val = -1000000.0;

  double log_RL0 = 1.0;
  double pvalue0 = 1.0;
  gsl_vector *beta0 = gsl_vector_alloc(beta->size);
  gsl_vector *logRL_logL_2parts0 = gsl_vector_alloc(logRL_logL_2parts->size);
  gsl_vector_set_zero(beta0);
  gsl_vector_set_zero(logRL_logL_2parts0);

  while (iter_count < maxiter) {
    iter_count = iter_count + 1;
    cc_logl_val0 = cc_logl_val;

    log_RL0 = log_RL;
    pvalue0 = pvalue;
    gsl_vector_memcpy(beta0, beta);
    gsl_vector_memcpy(logRL_logL_2parts0, logRL_logL_2parts);
    printf("\n  iteration = %zu \n", iter_count);
    
   
   
    
      printf("   W0, Y0,         ");
      gsl_matrix_set_zero( W0 );
      gsl_matrix_set_zero( Y0 );
      Build_W0YO( nrf, beta, ai, pi, Pheno, Qmat, Time, curve_num,  W0, Y0 ); 
      time_t current_time = time(0);
      tm* local_time = localtime(&current_time);
      printf("    %s ", asctime(local_time) );

     
      printf("  X = UtW0, Y = UtY0, ");
      gsl_matrix_set_zero( X );
      gsl_matrix_set_zero( Y );
      Buid_XY( W0, Y0, Uk,   X, Y );     
      current_time = time(0);
      local_time = localtime(&current_time);
      printf("%s ", asctime(local_time) );  
    

    
    printf("  1.Calc fdGPR_IAmat \n");
    Calc_fdGPR_IAmat( par_vg, par_vp, par_vr, beta, ai, pi, Time, eval, Uk, X, Y, W0, Y0,
                      fd_GPR, IAmat, TTTi, ei );

   
    printf("  2.Calc IA_inv \n");
    gsl_matrix *IAmat_cpy = gsl_matrix_alloc(uniGPR_size, uniGPR_size);
    gsl_matrix_set_zero( IAmat_cpy );
    gsl_matrix_memcpy( IAmat_cpy, IAmat );
    int sig;
    gsl_permutation *pmt = gsl_permutation_alloc( uniGPR_size );
    LUDecomp(IAmat_cpy, pmt, &sig);
    LUInvert(IAmat_cpy, pmt, IA_inv);
    gsl_permutation_free( pmt );
    gsl_matrix_free( IAmat_cpy );

   
    printf("  3.delta = IA_inv * fd_GPR \n      "); 
    gsl_matrix_set_zero( delta );
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, IA_inv, fd_GPR, 0.0, delta); 
    for(size_t i = 0; i < uniGPR_size; ++i){ 
      printf("%8.5f ", gsl_matrix_get(delta, i, 0) );
      if(i == uniG_size-1 || i == uniGP_size-1){ printf("\n      "); }
      if(i == uniGPR_size-1){ printf("\n"); }
    }
    printf("    var_com = \n      ");
    for(size_t i = 0; i < uniGPR_size; ++i){ 
      printf("%8.5f ", gsl_matrix_get(var_com, i, 0) );
      if(i == uniG_size-1 || i == uniGP_size-1){ printf("\n      "); }
      if(i == uniGPR_size-1){ printf("\n"); }
    }

    
    
    double step = 1.0;
    int max_backtrack = 10;      
    gsl_matrix *scaled_delta = gsl_matrix_alloc(uniGPR_size, 1);

    do {
      gsl_matrix_memcpy(var_com_update, var_com);
      gsl_matrix_memcpy(scaled_delta, delta);
      gsl_matrix_scale(scaled_delta, step);          
      gsl_matrix_sub(var_com_update, scaled_delta);  

      Update_par_vgvpvr(var_com_update, uniG_size, uniP_size, uniR_size,
                      par_vg, par_vp, par_vr);

     flag_pd = 1;
     Calc_flagpd(par_vg, par_vp, par_vr, flag_pd);

     if (flag_pd == 1) break;   

     step /= 2.0;               
     printf("   Step too large, halved to %f\n", step);
    } while (step > 1e-8 && max_backtrack-- > 0);

     gsl_matrix_free(scaled_delta);

    if (flag_pd == 0) {
      printf("   Cannot find a positive definite update, stopping.\n");
     break;  
     }





    printf("  4.var_com_update = var_com - delta \n      ");
    for(size_t i = 0; i < uniGPR_size; ++i){ 
      printf("%8.5f ", gsl_matrix_get(var_com_update, i, 0) );
      if(i == uniG_size-1 || i == uniGP_size-1){ printf("\n      "); }
      if(i == uniGPR_size-1){ printf("\n"); }
    }
    
   
    Calc_aipi2( par_vg, par_vp, par_vr, beta, eval, Uk, Time, X, Y, 
                ai, pi );

    Calc2_logRL_beta_pval( eval, Time, X, Y, par_vg, par_vp, par_vr, npar_bsnp, 
                           log_RL, beta, pvalue, logRL_logL_2parts );
    cc_logl_val = log_RL;
    printf("    beta: ");
    for(size_t i=0; i<p_size; i++){
      printf("%.5f ", gsl_vector_get(beta,i) );
    }
    printf("\n");



    gsl_matrix *actual_delta = gsl_matrix_alloc(uniGPR_size, 1);
    gsl_matrix_memcpy(actual_delta, delta);
    gsl_matrix_scale(actual_delta, step);      
   
    cc_par_val = Calc_ccparval( actual_delta, var_com_update );
    gsl_matrix_free(actual_delta);


    cc_gra_val = Calc_ccgraval( fd_GPR, var_com );
    gsl_matrix_memcpy(var_com, var_com_update);

    
    printf("\n  8.cc_par_val = %7.8f, cc_gra_val = %7.5f, cc_logl_val = %7.5f, cc_logl_val.diff =%7.6f \n", 
           cc_par_val, cc_gra_val, cc_logl_val, fabs(cc_logl_val-cc_logl_val0) );
    printf("    pvalue = %8.5le \n", pvalue);

    int prb = 0; 
    prb = check_beta( Qmat, beta, curve_num, nrf );
    printf("    prb_beta = %d \n\n", prb );
   
   
    if(h0 == 1){
      if( cc_par_val < cc_par && fabs(cc_logl_val-cc_logl_val0) < 1e-04 ){ break; }      
    } else {
      if( (cc_par_val < cc_par && fabs(cc_logl_val-cc_logl_val0) < 1e-03) || iter_count >= 10){ break; }
      
      if( prb == 1 ){ 
        if( iter_count > 1){
          log_RL = log_RL0;
          pvalue = pvalue0;  
          gsl_vector_memcpy(beta, beta0);  
          gsl_vector_memcpy(logRL_logL_2parts, logRL_logL_2parts0); 
        } else {
           pvalue = 1.0; 
           
        }        
        break; 
      }
    }
  } 

  

  

  
  gsl_vector_free(beta0);
  gsl_vector_free(logRL_logL_2parts0);

  gsl_matrix_free( Z10 );
  gsl_matrix_free( Z20 );
  gsl_matrix_free( G0 );
  gsl_matrix_free( P0 );
  gsl_matrix_free( R0 );
  gsl_matrix_free( G0_inv );
  gsl_matrix_free( P0_inv );
  gsl_matrix_free( R0_inv );

  gsl_matrix_free( fd_GPR );
  gsl_matrix_free( IAmat );
  gsl_matrix_free( IA_inv );
  

  gsl_matrix_free( delta );
  gsl_matrix_free( var_com );
  gsl_matrix_free( var_com_update );

  gsl_matrix_free( TTTi );
  gsl_vector_free( ei );

  gsl_matrix_free( W0 );
  gsl_matrix_free( Y0 );
  gsl_matrix_free( Y );

  return;
}



double get_mean( const gsl_vector *phe){
  size_t n_size = phe->size;
  double mean = 0.0;

  for(size_t i = 0; i < n_size; ++i){
    mean += gsl_vector_get( phe, i );
  }
  mean = mean / n_size;

  return mean;
}


double get_var( const gsl_vector *phe){
  size_t n_size = phe->size;
  double mean = get_mean(phe);

  double var = 0.0;
  for(size_t i = 0; i < n_size; ++i){
    double tmp = (gsl_vector_get(phe,i) - mean);
    var += (tmp * tmp);
  }
  var = var / n_size;

  return var;
}



double get_mean_rm_outliers(const gsl_vector *phe){
  size_t n_size = phe->size;
  double mean = 0.0;

  double tmp1 = get_mean( phe );
  double tmp2 = get_var( phe );

  double upper = tmp1 + 2*sqrt(tmp2);
  double lower = tmp1 - 2*sqrt(tmp2);

  size_t count = 0;
  for(size_t i = 0; i < n_size; ++i){
    double tmp = gsl_vector_get( phe, i );
    
    if((tmp >= upper) || (tmp <= lower)){
      mean += 0;  
      count = count + 0;  
    }else{
      mean += tmp;  
      count = count + 1;  
    }
  }
  mean = mean / count;

  printf("mean=%.4f, var=%.4f; upper = %.5f, lower = %.5f; count = %zu \n", tmp1, tmp2, upper, lower, count);

  return mean;
}


void get_ne_size( const size_t curve_type, size_t &ne_size ) {
  size_t nrf = 3;

  //--- "gompertz" --//
  if(curve_type == 1){ ne_size = 3; }
  return;
}

void get_lop_ne_size( const size_t curve_type, const size_t nrf, size_t &ne_size ) {
  
  //--- "gompertz" --//
  if(curve_type == 1){ ne_size = 3; }
  return;
}

double sum_least_squares( const gsl_vector *v, void *params )
{
    double *ctyy = (double *)params;

    size_t curve_num = ctyy[0];
    size_t d_size = ctyy[1];
   


    gsl_vector *Times = gsl_vector_alloc(d_size);
    gsl_vector *y     = gsl_vector_alloc(d_size);
    for(size_t i = 1; i <= d_size; ++i){
       gsl_vector_set( Times, i-1, ctyy[i+1] );
       gsl_vector_set( y, i-1, ctyy[i+d_size+1] );
    }
    gsl_vector *miu = gsl_vector_alloc(d_size); 
    LC_get_mu( v, Times, curve_num, miu );

    gsl_vector *cha = gsl_vector_alloc(d_size); 
    gsl_vector_set_zero(cha);
    gsl_vector_add(cha, y);
    gsl_vector_sub(cha, miu);

    double sum_sds = 0.0;
    double tmp = 0.0;
    for(size_t i = 0; i < d_size; ++i){  
      tmp = gsl_vector_get(cha, i);
      sum_sds = sum_sds +  tmp * tmp;
    }

    gsl_vector_free( Times );
    gsl_vector_free( y );
    gsl_vector_free( miu );
    gsl_vector_free( cha );

    return sqrt(sum_sds);
}

void optim_curvefit( const size_t curve_num, const size_t ne_size, 
                     const gsl_vector *Times, const gsl_vector *yk, 
                     gsl_vector *parameters )
{
    size_t d_size = Times->size;
    double cty[2*d_size+2];

    cty[0] = curve_num;
    cty[1] = d_size;
    for(size_t i = 1; i <= d_size; ++i){  
      double tmp1 = gsl_vector_get(Times, i-1);
      double tmp2 = gsl_vector_get(yk, i-1);
      cty[1+i] = tmp1;
      cty[1+i+d_size] = tmp2;
    }
    
    gsl_multimin_function minex_func;
    minex_func.n = ne_size;  
    minex_func.f = &sum_least_squares;
    minex_func.params = (void *)cty;


    int iter = 0, status; 

    //-- set initial step size to 0.5
    gsl_vector *x = gsl_vector_alloc(ne_size);
    gsl_vector_set_all(x, 0.5);

    gsl_vector *ss = gsl_vector_alloc(ne_size);
    gsl_vector_set_all(ss, 0.01);

    const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2; 
    gsl_multimin_fminimizer *s = gsl_multimin_fminimizer_alloc(T, ne_size);
    gsl_multimin_fminimizer_set(s, &minex_func, x, ss);                         

    double size;
    do{
        iter++;
        status = gsl_multimin_fminimizer_iterate( s );

        if( status ) break;

        size = gsl_multimin_fminimizer_size (s);
        status = gsl_multimin_test_size( size, 1E-3);

      
    }
    while ( status == GSL_CONTINUE && iter < 5000 );

    for(size_t i = 0; i < ne_size; ++i){  
      double tmp = gsl_vector_get( s->x, i);
      printf(" %.5f", tmp);
    }
    printf("; f() = %.3f; size = %.5f\n", s->fval, size);

    for(size_t i = 0; i < ne_size; ++i){  
      double tmp = gsl_vector_get( s->x, i);
      gsl_vector_set(parameters, i, tmp );
    }
    

    gsl_vector_free(x);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free(s);

    //return status;
}



int check_curve_pars(const size_t curve_num, const gsl_vector *v){
  
  
  gsl_vector *par = gsl_vector_alloc(v->size);
  gsl_vector_memcpy(par, v);

  int res;
  
  //-- gompertz --//
  if(curve_num == 1){ 
    double a = gsl_vector_get(par, 0);
    double b = gsl_vector_get(par, 1);  
    double k = gsl_vector_get(par, 2);

    
    if(b <= 0)  { res = 1; }
    if(k <= 0)  { res = 1; }
  }
 
  return res;
}

int expb_f( const gsl_vector *v, void *params,
            gsl_vector *f )
{
    double *ctyy = (double *)params;

    size_t curve_num = ctyy[0];
    size_t d_size = ctyy[1];
    
      gsl_vector *Times = gsl_vector_alloc(d_size);
      gsl_vector *y     = gsl_vector_alloc(d_size);
      for(size_t i = 1; i <= d_size; ++i){
        gsl_vector_set( Times, i-1, ctyy[i+1] );
        gsl_vector_set( y, i-1, ctyy[i+d_size+1] );
      }
      gsl_vector *miu = gsl_vector_alloc(d_size); 
      LC_get_mu( v, Times, curve_num, miu );

      gsl_vector *cha = gsl_vector_alloc(d_size); 
      gsl_vector_set_zero(cha);
      gsl_vector_add(cha, miu);
      gsl_vector_sub(cha, y);
      gsl_vector_memcpy(f, cha);

      gsl_vector_free( Times );
      gsl_vector_free( y );
      gsl_vector_free( miu );
      gsl_vector_free( cha );

      return GSL_SUCCESS;
  
}

int expb_df( const gsl_vector *v, void *params,
            gsl_matrix *J )
{
    double *ctyy = (double *)params;

    size_t curve_num = ctyy[0];
    size_t d_size = ctyy[1];
   
      gsl_vector *Times = gsl_vector_alloc(d_size);
      gsl_vector *y     = gsl_vector_alloc(d_size);
      for(size_t i = 1; i <= d_size; ++i){
        gsl_vector_set( Times, i-1, ctyy[i+1] );
        gsl_vector_set( y, i-1, ctyy[i+d_size+1] );
      }

     
      LC_get_derv( v, Times, curve_num, J );

 
      gsl_vector_free( Times );
      gsl_vector_free( y );

      return GSL_SUCCESS;
    
}

void callback( const size_t iter, void *params, 
               const gsl_multifit_nlinear_workspace *w )
{
  gsl_vector *f = gsl_multifit_nlinear_residual(w);
  gsl_vector *x = gsl_multifit_nlinear_position(w);
  double rcond;

  gsl_multifit_nlinear_rcond(&rcond, w);
 
}

void nonlinear_least_squares( const gsl_matrix *Pheno, const gsl_vector *Times, 
                              const size_t curve_num, gsl_matrix *parameters )
{
    
    size_t d_size = Times->size;
    size_t n_size = Pheno->size1;
    size_t ne_size = 0.0;
    ne_size = parameters->size2;

    gsl_vector *param = gsl_vector_alloc(ne_size);


    for(size_t i = 0; i < n_size; ++i){  
      gsl_vector_const_view yk = gsl_matrix_const_row(Pheno, i);


      const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
      gsl_multifit_nlinear_workspace *w;
      gsl_multifit_nlinear_fdf fdf;
      gsl_multifit_nlinear_parameters fdf_params =
          gsl_multifit_nlinear_default_parameters();
      const size_t n = d_size;
      const size_t p = ne_size;

      gsl_vector *f;
      gsl_matrix *J;
      gsl_matrix *covar = gsl_matrix_alloc(p,p);
      
      double cty[2*d_size+2];
      cty[0] = curve_num;
      cty[1] = d_size;
      for(size_t i = 1; i <= d_size; ++i){  
        double tmp1 = gsl_vector_get(Times, i-1);
        double tmp2 = gsl_vector_get(&yk.vector, i-1);
        cty[1+i] = tmp1;
        cty[1+i+d_size] = tmp2;
      }

      double x_init[p]; 
      for(size_t i=0; i<p; i++){ x_init[i] = 0.5; }
      double weights[n];
      for(size_t i=0; i<n; i++){ weights[i] = 1.0; } 

      gsl_vector_view x = gsl_vector_view_array(x_init, p);
    
      gsl_rng *r;
      double chisq, chisq0;
      int status, info;

      const double xtol = 1e-5;
      const double gtol = 1e-5;
      const double ftol = 1e-5; 

      gsl_rng_env_setup();
      r = gsl_rng_alloc( gsl_rng_default );

      fdf.f  = expb_f;
      fdf.df = expb_df;
      fdf.fvv = NULL;
      fdf.n = n;
      fdf.p = p;
      fdf.params = &cty;

     
      for(size_t i = 0; i < n; i++){
          double yi = gsl_vector_get( &yk.vector, i ); 
          double si = 0.1 * yi; 
          weights[i] = 1.0 / (si * si);
      }
      gsl_vector_view wts = gsl_vector_view_array(weights, n);

    
      w = gsl_multifit_nlinear_alloc(T, &fdf_params, n, p);

    
      gsl_multifit_nlinear_winit(&x.vector, &wts.vector, &fdf, w);

     
      f = gsl_multifit_nlinear_residual(w);
      gsl_blas_ddot(f, f, &chisq0);

      
      status = gsl_multifit_nlinear_driver(200, xtol, gtol, ftol,
                                           callback, NULL, &info, w);

     


      gsl_vector_view param_i = gsl_matrix_row(parameters, i);
      gsl_vector_memcpy(&param_i.vector, w->x); 

      gsl_multifit_nlinear_free(w);
      gsl_matrix_free(covar);
      gsl_rng_free(r);


    }

    gsl_vector_free( param );
    return;
}



void SAD10( const gsl_vector *par_vr, const gsl_vector *Time, 
            gsl_matrix *R0, gsl_matrix *R0inv, double &detR0 )
{
  size_t d_size = Time->size;

  gsl_matrix_set_zero(R0);
  gsl_matrix_set_zero(R0inv);

  size_t nr = par_vr->size;
  double phi = gsl_vector_get(par_vr, nr-1);

  //-- Lambda --//
  gsl_matrix *Lambda = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix_set_zero( Lambda );
  for (size_t i = 0; i < d_size; ++i) {
    double Time_i = gsl_vector_get(Time, i);
    for (size_t j = 0; j <= i; ++j) {
      double Time_j = gsl_vector_get(Time, j);
      double tmp = pow(phi, fabs(Time_i - Time_j));   
      gsl_matrix_set(Lambda, i, j, tmp);
    }
  }

  //-- middle (D = diag(par_vr[0..d_size-1])) --//
  gsl_matrix *middle = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix_set_zero( middle );
  for (size_t i = 0; i < d_size; ++i) {
    double middle_i = gsl_vector_get(par_vr, i);
    gsl_matrix_set(middle, i, i, middle_i);
  }

  //-- R0 = Lambda * D * Lambda^T --//
  gsl_matrix *Lambda_middle = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix_set_zero( Lambda_middle );
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Lambda, middle, 0.0, Lambda_middle);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Lambda_middle, Lambda, 0.0, R0);
  
  gsl_matrix_free( Lambda );
  gsl_matrix_free( middle );
  gsl_matrix_free( Lambda_middle );

  //-- R0_inv via LU decomposition --//
  gsl_matrix *R0_cpy = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix_memcpy(R0_cpy, R0);
  gsl_permutation *p = gsl_permutation_alloc(d_size);
  int signum;
  LUDecomp(R0_cpy, p, &signum);
  LUInvert(R0_cpy, p, R0inv);          

  gsl_matrix_free(R0_cpy);
  gsl_permutation_free(p);

  //-- determinant of R0 --//
  detR0 = 1.0;
  for (size_t i = 0; i < d_size; ++i) {
    detR0 *= gsl_vector_get(par_vr, i);
  }
}


void SAD1( const gsl_vector *par_vr, const gsl_vector *Time, 
           gsl_matrix *R0, gsl_matrix *R0inv, double &detR0 )
{
  size_t d_size = Time->size;

  gsl_matrix_set_zero(R0);
  gsl_matrix_set_zero(R0inv);

  double phi = gsl_vector_get(par_vr, 1);
  double sigma2 = gsl_vector_get(par_vr, 0);

  //-- Lambda --//
  gsl_matrix *Lambda = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix_set_zero( Lambda );
  for (size_t i = 0; i < d_size; ++i) {
    double Time_i = gsl_vector_get(Time, i);
    for (size_t j = 0; j <= i; ++j) {
      double Time_j = gsl_vector_get(Time, j);
      double tmp = pow(phi, fabs(Time_i - Time_j));
      gsl_matrix_set(Lambda, i, j, tmp);
    }
  }

  //-- middle = sigma2 * I --//
  gsl_matrix *middle = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix_set_identity( middle );
  gsl_matrix_scale(middle, sigma2);

  //-- R0 = Lambda * (sigma2 * I) * Lambda^T = sigma2 * Lambda * Lambda^T --//
  gsl_matrix *Lambda_middle = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix_set_zero( Lambda_middle );
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Lambda, middle, 0.0, Lambda_middle);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Lambda_middle, Lambda, 0.0, R0);
  
  gsl_matrix_free( Lambda );
  gsl_matrix_free( middle );
  gsl_matrix_free( Lambda_middle );

  //-- R0_inv via LU decomposition --//
  gsl_matrix *R0_cpy = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix_memcpy(R0_cpy, R0);
  gsl_permutation *p = gsl_permutation_alloc(d_size);
  int signum;
  LUDecomp(R0_cpy, p, &signum);
  LUInvert(R0_cpy, p, R0inv);

  gsl_matrix_free(R0_cpy);
  gsl_permutation_free(p);

  //-- determinant of R0 --//
  detR0 = 1.0;
  for (size_t i = 0; i < d_size; ++i) {
    detR0 *= sigma2;
  }
}


double logfy_multinorm( const gsl_matrix *Pheno, const gsl_vector *miu, 
                     const gsl_matrix *R0inv, const double detR0 )
{
    size_t n_size = Pheno->size1;
    size_t d_size = Pheno->size2;
    size_t nd_size = n_size * d_size;

    double t1 = 0.0;
    double t2 = 0.0;
    t1 = 0.0 - nd_size * 0.50 * log(2.0 * M_PI);
    t2 = 0.0 - n_size * 0.50 * log(detR0);
    
    double t3 = 0.0;
    gsl_vector *ymiu = gsl_vector_alloc(d_size);
    gsl_vector *sgm_ymiu = gsl_vector_alloc(d_size);
    for (size_t i = 0; i < n_size; ++i) {
        double t3_tmp = 0.0;
        gsl_vector_const_view Y_k = gsl_matrix_const_row(Pheno, i);

        gsl_vector_set_zero(ymiu);
        gsl_vector_set_zero(sgm_ymiu);

        gsl_vector_add(ymiu, &Y_k.vector);
        gsl_vector_sub(ymiu, miu);

        gsl_blas_dgemv(CblasNoTrans, 1.0, R0inv, ymiu, 0.0, sgm_ymiu);
        gsl_blas_ddot(ymiu, sgm_ymiu, &t3_tmp);
        t3 += -1.0/2.0 * t3_tmp;
    }

    double logfy = 0.0;
    logfy = t1 + t2 + t3;


    gsl_vector_free(ymiu);
    gsl_vector_free(sgm_ymiu);

    return logfy;
}

double mle1( const gsl_vector *v, void *params )
{
    double *ctyy = (double *)params;

    size_t curve_num = ctyy[0];
    size_t n_size = ctyy[1];
    size_t d_size = ctyy[2];
    size_t ne_size = ctyy[3];
    

    double *param1 = &ctyy[4];
    double *param2 = &ctyy[4+d_size];
    double *param3 = &ctyy[4+d_size+(n_size*d_size)];

    gsl_vector_const_view Times = gsl_vector_const_view_array(param1, d_size);
    gsl_matrix_const_view Pheno = gsl_matrix_const_view_array(param2, n_size, d_size);
    gsl_vector_const_view par2 = gsl_vector_const_view_array(param3, ne_size);

    //----//
    gsl_matrix *R0 = gsl_matrix_alloc(d_size, d_size);
    gsl_matrix *R0inv = gsl_matrix_alloc(d_size, d_size);
    gsl_matrix_set_zero(R0);
    gsl_matrix_set_zero(R0inv);
    double detR0 = 0.0;
    SAD1(v, &Times.vector, R0, R0inv, detR0);
    
    gsl_vector *miu = gsl_vector_alloc(d_size);
    gsl_vector_set_zero(miu);
    LC_get_mu( &par2.vector, &Times.vector, curve_num, miu );

    double logfy = -1.0 * logfy_multinorm( &Pheno.matrix, miu, R0inv, detR0 );

    
    if( gsl_vector_get(v,0) <= 0 ){ logfy = GSL_NAN; } 
    if( gsl_vector_get(v,1) <= 0 ){ logfy = GSL_NAN; }     
    if( gsl_vector_get(v,1) >= 1 ){ logfy = GSL_NAN; } 
    

    gsl_matrix_free(R0);
    gsl_matrix_free(R0inv);
    gsl_vector_free(miu);

    return logfy;
}

double mle2( const gsl_vector *v, void *params )
{
    double *ctyy = (double *)params;

    size_t curve_num = ctyy[0];
    size_t n_size = ctyy[1];
    size_t d_size = ctyy[2];
    

    double *param1 = &ctyy[3]; 
    double *param2 = &ctyy[3+d_size];
    double *param3 = &ctyy[3+d_size+(n_size*d_size)];

    gsl_vector_const_view Times = gsl_vector_const_view_array(param1, d_size);
    gsl_matrix_const_view Pheno = gsl_matrix_const_view_array(param2, n_size, d_size);
    gsl_vector_const_view par2 = gsl_vector_const_view_array(param3, 2);


    //----//
    double detR0 = 0.0;

    gsl_matrix *R0 = gsl_matrix_alloc(d_size, d_size);
    gsl_matrix *R0inv = gsl_matrix_alloc(d_size, d_size);
    SAD1(&par2.vector, &Times.vector, R0, R0inv, detR0);

    gsl_vector *miu = gsl_vector_alloc(d_size);
    gsl_vector_set_zero(miu);
    LC_get_mu( v, &Times.vector, curve_num, miu );

    double logfy = -1.0 * logfy_multinorm( &Pheno.matrix, miu, R0inv, detR0 );

    gsl_matrix_free(R0);
    gsl_matrix_free(R0inv);
    gsl_vector_free(miu);

    return logfy;
}

double mle3( const gsl_vector *v, void *params )
{
    double *ctyy = (double *)params;

    size_t curve_num = ctyy[0];
    size_t n_size = ctyy[1];
    size_t d_size = ctyy[2];
    size_t ne_size = 0.0;
  
    ne_size = v->size - 2;

    double *param1 = &ctyy[3]; 
    double *param2 = &ctyy[3+d_size];

    gsl_vector_const_view Times = gsl_vector_const_view_array(param1, d_size);
    gsl_matrix_const_view Pheno = gsl_matrix_const_view_array(param2, n_size, d_size);
    

    gsl_vector_const_view par1 = gsl_vector_const_subvector(v, 0, ne_size);
    gsl_vector_const_view par2 = gsl_vector_const_subvector(v, ne_size, 2); 

    //----//
    double detR0 = 0.0;

    gsl_matrix *R0 = gsl_matrix_alloc(d_size, d_size);
    gsl_matrix *R0inv = gsl_matrix_alloc(d_size, d_size);
    SAD1(&par2.vector, &Times.vector, R0, R0inv, detR0);

    gsl_vector *miu = gsl_vector_alloc(d_size);
    gsl_vector_set_zero(miu);
    LC_get_mu( &par1.vector, &Times.vector, curve_num, miu );

    double logfy = -1.0 * logfy_multinorm( &Pheno.matrix, miu, R0inv, detR0 );

    if( gsl_vector_get(v,0+ne_size) <= 0 ){ logfy = GSL_NAN; }
    if( gsl_vector_get(v,1+ne_size) <= 0 ){ logfy = GSL_NAN; }
    if( gsl_vector_get(v,1+ne_size) >= 1 ){ logfy = GSL_NAN; }

    gsl_matrix_free(R0);
    gsl_matrix_free(R0inv);
    gsl_vector_free(miu);

    return logfy;
}

double optim_mle1( gsl_vector *par1, const gsl_vector *par2, 
                 const gsl_vector *Times, const gsl_matrix *Pheno, 
                 const size_t curve_num, const size_t ne_size )
{
    size_t n_size = Pheno->size1;
    size_t d_size = Pheno->size2;

    double *cty = new double[4+d_size+(n_size*d_size)+ne_size];
   
    cty[0] = curve_num;
    cty[1] = n_size;
    cty[2] = d_size;
    cty[3] = ne_size;

    size_t c = 1;
    for(size_t i = 1; i <= d_size; ++i){  
      double tmp1 = gsl_vector_get(Times, i-1);
      cty[3+c] = tmp1;
      c++;
    }
    for(size_t i = 1; i <= n_size; ++i){  
      for(size_t j = 1; j <= d_size; ++j){  
        double tmp2 = gsl_matrix_get(Pheno, i-1, j-1);
        cty[3+c] = tmp2;
        c++;
      }
    }
    for(size_t j = 0; j < ne_size; ++j){  
      double tmp3 = gsl_vector_get(par2, j);
      cty[3+c] = tmp3;
      c++;
    }


    gsl_multimin_function minex_func;
    minex_func.n = 2;
    minex_func.f = &mle1;
    minex_func.params = (void *)cty;

    int iter = 0, status; 

    //-- set initial step size to 0.5
    gsl_vector *x = gsl_vector_alloc(2);
    gsl_vector_set_zero(x);
    gsl_vector_memcpy(x, par1);

    gsl_vector *ss = gsl_vector_alloc(2);
    gsl_vector_set_zero(ss);
    gsl_vector_set_all(ss, 0.0001);

    const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer *s = gsl_multimin_fminimizer_alloc(T, 2);
    gsl_multimin_fminimizer_set(s, &minex_func, x, ss);

    int size = 0.0;
    do{
        iter++;
        status = gsl_multimin_fminimizer_iterate( s );
       

        if( status ) break;

        size = gsl_multimin_fminimizer_size (s);
        status = gsl_multimin_test_size( size, 1E-3);

      
    }
    while ( status == GSL_CONTINUE && iter < 200 );

   


    gsl_vector_memcpy(par1, s->x );
    double res = s->fval;

    gsl_vector_free(x);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free(s);

    delete[] cty;
    cty = nullptr;

    return res;
}

double optim_mle2( gsl_vector *par1, const gsl_vector *par2, 
                 const gsl_vector *Times, const gsl_matrix *Pheno, 
                 const size_t curve_num, const size_t ne_size )
{
    size_t d_size = Times->size;
    size_t n_size = Pheno->size1;

    double *cty = new double[3+d_size+(n_size*d_size)+d_size+1];
   
    cty[0] = curve_num;
    cty[1] = n_size;
    cty[2] = d_size;

    size_t c = 1;
    for(size_t i = 1; i <= d_size; ++i){  
      double tmp1 = gsl_vector_get(Times, i-1);
      cty[2+c] = tmp1;
      c++;
    }
    for(size_t i = 1; i <= n_size; ++i){  
      for(size_t j = 1; j <= d_size; ++j){  
        double tmp2 = gsl_matrix_get(Pheno, i-1, j-1);
        cty[2+c] = tmp2;
        c++;
      }
    }
    for(size_t j = 0; j < 2; ++j){  
      double tmp3 = gsl_vector_get(par2, j);
      cty[2+c] = tmp3;
      c++;
    }


    gsl_multimin_function minex_func;
    minex_func.n = ne_size;
    minex_func.f = &mle2;
    minex_func.params = (void *)cty;


    int iter = 0, status; 

    //-- set initial step size to 0.5
    gsl_vector *x = gsl_vector_alloc(ne_size);
    gsl_vector_memcpy(x, par1);

    gsl_vector *ss = gsl_vector_alloc(ne_size);
    gsl_vector_set_all(ss, 0.5); 

    const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer *s = gsl_multimin_fminimizer_alloc(T, ne_size);
    gsl_multimin_fminimizer_set(s, &minex_func, x, ss);

    double size = 0.0;
    do{
        iter++;
        status = gsl_multimin_fminimizer_iterate( s );

        if( status ) break;

        size = gsl_multimin_fminimizer_size (s);
        status = gsl_multimin_test_size( size, 1E-3);

        if( status == GSL_SUCCESS ){
         
        }
        
    }
    while ( status == GSL_CONTINUE && iter < 200 );
    

    gsl_vector_memcpy(par1, s->x );
    double res = s->fval;

    gsl_vector_free(x);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free(s);

    delete[] cty;
    cty = nullptr;

    return res;
}

double optim_mle3( gsl_vector *par1, 
                 const gsl_vector *Times, const gsl_matrix *Pheno, 
                 const size_t curve_num, const size_t ne_size )
{
    size_t d_size = Times->size;
    size_t n_size = Pheno->size1;

    double *cty = new double[3+d_size+(n_size*d_size)];
   
    cty[0] = curve_num;
    cty[1] = n_size;
    cty[2] = d_size;

    size_t c = 1;
    for(size_t i = 1; i <= d_size; ++i){  
      double tmp1 = gsl_vector_get(Times, i-1);
      cty[2+c] = tmp1;
      c++;
    }
    for(size_t i = 1; i <= n_size; ++i){  
      for(size_t j = 1; j <= d_size; ++j){  
        double tmp2 = gsl_matrix_get(Pheno, i-1, j-1);
        cty[2+c] = tmp2;
        c++;
      }
    }


    gsl_multimin_function minex_func;
    minex_func.n = 2+ne_size;
    minex_func.f = mle3;
    minex_func.params = (void *)cty;


    int iter = 0, status; 

    //-- set initial step size to 0.5
    gsl_vector *x = gsl_vector_alloc(2+ne_size);
    gsl_vector_memcpy(x, par1);

    gsl_vector *ss = gsl_vector_alloc(2+ne_size);
    gsl_vector_set_all(ss, 0.0005);

    const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer *s = gsl_multimin_fminimizer_alloc(T, 2+ne_size);
    gsl_multimin_fminimizer_set(s, &minex_func, x, ss);

    double size = 0.0;
    do{
        iter++;
        status = gsl_multimin_fminimizer_iterate( s );

        if( status ) break;

        size = gsl_multimin_fminimizer_size (s);
        status = gsl_multimin_test_size( size, 1E-4);

    }
    while ( status == GSL_CONTINUE && iter < 200 );

   

    gsl_vector_memcpy(par1, s->x );
    double res = s->fval;

    gsl_vector_free(x);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free(s);

    delete[] cty;
    cty = nullptr;

    return res;
}

void MLE( gsl_vector *par, const gsl_matrix *Pheno, 
          const gsl_vector *Times, const size_t curve_num, const size_t ne_size )
{
   


   gsl_vector_view par1 = gsl_vector_subvector(par, ne_size, 2);
   gsl_vector_view par2 = gsl_vector_subvector(par, 0, ne_size);
   double res1 = optim_mle1( &par1.vector, &par2.vector, Times, Pheno, curve_num, ne_size );
   printf("optim1:");
   for(size_t i = 0; i < (ne_size+2); ++i){ 
     printf(" %.5f", gsl_vector_get(par,i) );
   }
   printf("; -logL = %.5f \n", res1 );


   gsl_vector_view par11 = gsl_vector_subvector(par, 0, ne_size);
   gsl_vector_view par22 = gsl_vector_subvector(par, ne_size, 2);
   double res2 = optim_mle2( &par11.vector, &par22.vector, Times, Pheno, curve_num, ne_size );
   printf("optim2:");
   for(size_t i = 0; i < (ne_size+2); ++i){ 
     printf(" %.5f", gsl_vector_get(par,i) );
   }
   printf("; -logL = %.5f \n", res2 );


   double res3 = optim_mle3( par, Times, Pheno, curve_num, ne_size );
   printf("optim3:");
   for(size_t i = 0; i < (ne_size+2); ++i){ 
     printf(" %.5f", gsl_vector_get(par,i) );
   }
   printf("; -logL = %.5f \n", res3 );

   return;
}


double mle1_last( const gsl_vector *v, void *params )
{
    double *ctyy = (double *)params;

    size_t curve_num = ctyy[0];
    size_t n_size = ctyy[1];
    size_t d_size = ctyy[2];
    size_t ne_size = ctyy[3];
   

    double *param1 = &ctyy[4];
    double *param2 = &ctyy[4+d_size];
    double *param3 = &ctyy[4+d_size+(n_size*d_size)];

    gsl_vector_const_view Times = gsl_vector_const_view_array(param1, d_size);
    gsl_matrix_const_view Pheno = gsl_matrix_const_view_array(param2, n_size, d_size);
    gsl_vector_const_view par2 = gsl_vector_const_view_array(param3, ne_size);

    //----//
    gsl_matrix *R0 = gsl_matrix_alloc(d_size, d_size);
    gsl_matrix *R0inv = gsl_matrix_alloc(d_size, d_size);
    gsl_matrix_set_zero(R0);
    gsl_matrix_set_zero(R0inv);
    double detR0 = 0.0;
    SAD10(v, &Times.vector, R0, R0inv, detR0);
    
    gsl_vector *miu = gsl_vector_alloc(d_size);
    gsl_vector_set_zero(miu);
    LC_get_mu( &par2.vector, &Times.vector, curve_num, miu );

    double logfy = -1.0 * logfy_multinorm( &Pheno.matrix, miu, R0inv, detR0 );
    if( gsl_vector_get(v,d_size) <= 0 ){ logfy = GSL_NAN; }     
   

    gsl_matrix_free(R0);
    gsl_matrix_free(R0inv);
    gsl_vector_free(miu);

    return logfy;
}
double optim_mle1_last( gsl_vector *par1, const gsl_vector *par2, 
                 const gsl_vector *Times, const gsl_matrix *Pheno, 
                 const size_t curve_num, const size_t ne_size )
{
    size_t n_size = Pheno->size1;
    size_t d_size = Pheno->size2;

    double *cty = new double[4+d_size+(n_size*d_size)+ne_size];
   
    cty[0] = curve_num;
    cty[1] = n_size;
    cty[2] = d_size;
    cty[3] = ne_size;

    size_t c = 1;
    for(size_t i = 1; i <= d_size; ++i){  
      double tmp1 = gsl_vector_get(Times, i-1);
      cty[3+c] = tmp1;
      c++;
    }
    for(size_t i = 1; i <= n_size; ++i){  
      for(size_t j = 1; j <= d_size; ++j){  
        double tmp2 = gsl_matrix_get(Pheno, i-1, j-1);
        cty[3+c] = tmp2;
        c++;
      }
    }
    for(size_t j = 0; j < ne_size; ++j){  
      double tmp3 = gsl_vector_get(par2, j);
      cty[3+c] = tmp3;
      c++;
    }


    gsl_multimin_function minex_func;
    minex_func.n = d_size+1;
    minex_func.f = &mle1_last;
    minex_func.params = (void *)cty;

    int iter = 0, status; 

    //-- set initial step size to 0.5
    gsl_vector *x = gsl_vector_alloc(d_size+1);
    gsl_vector_set_zero(x);
    gsl_vector_memcpy(x, par1);

    gsl_vector *ss = gsl_vector_alloc(d_size+1);
    gsl_vector_set_zero(ss);
    gsl_vector_set_all(ss, 0.00001);

    const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer *s = gsl_multimin_fminimizer_alloc(T, d_size+1);
    gsl_multimin_fminimizer_set(s, &minex_func, x, ss);

    int size = 0.0;
    do{
        iter++;
        status = gsl_multimin_fminimizer_iterate( s );
        

        if( status ) break;

        size = gsl_multimin_fminimizer_size (s);
        status = gsl_multimin_test_size( size, 1E-3);

       
    }
    while ( status == GSL_CONTINUE && iter < 200 );

    


    gsl_vector_memcpy(par1, s->x );
    double res = s->fval;

    gsl_vector_free(x);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free(s);

    delete[] cty;
    cty = nullptr;

    return res;
}
void MLE_last( gsl_vector *par, const gsl_matrix *Pheno, 
          const gsl_vector *Times, const size_t curve_num, const size_t ne_size )
{
   
   size_t d_size = Times->size;
  


   gsl_vector_view par1 = gsl_vector_subvector(par, ne_size, d_size+1);
   gsl_vector_view par2 = gsl_vector_subvector(par, 0, ne_size);
   double res1 = optim_mle1_last( &par1.vector, &par2.vector, Times, Pheno, curve_num, ne_size );
   printf("optim1:");
   for(size_t i = 0; i < (ne_size+d_size+1); ++i){ 
     printf(" %.5f", gsl_vector_get(par,i) );
   }
   printf("; -logL = %.5f \n", res1 );

   return;
}
void Funmap(const gsl_matrix *Pheno, const gsl_vector *Times, 
            const size_t curve_num, const size_t nrf, gsl_vector *par2 )
{
   size_t n_size = Pheno->size1;
   size_t d_size = Times->size;
   size_t ne_size = 0;
 
   get_lop_ne_size(curve_num, nrf, ne_size);
   printf("Funmap: nrf=%zu, ne_size=%zu \n", nrf, ne_size);

   gsl_matrix *parameters = gsl_matrix_alloc(n_size, ne_size); 
   gsl_matrix_set_zero( parameters );
   nonlinear_least_squares( Pheno, Times, curve_num, parameters );

   gsl_vector *par = gsl_vector_alloc(ne_size+2);
   gsl_vector_set_zero(par);
   for(size_t j = 0; j < ne_size; ++j){ 
     gsl_vector_view colj = gsl_matrix_column(parameters, j);
  
     double tmp3 = get_mean_rm_outliers( &colj.vector );
     gsl_vector_set(par, j, tmp3);
   }
  

   
     gsl_vector_const_view colj = gsl_matrix_const_column(Pheno, 0); 
     double tmp1 = get_var( &colj.vector );
     gsl_vector_set(par, ne_size+0, sqrt(tmp1)/2.0 );
     gsl_vector_set(par, ne_size+1, 0.35);

   printf("NLS: \n");
   for(size_t j = 0; j < ne_size; ++j){ 
     double parj = gsl_vector_get(par, j);
     printf("%.5f ", parj);
   }
   printf("\n");
   for(size_t j = ne_size; j < (ne_size+2); ++j){ 
     double parj = gsl_vector_get(par, j);
     printf("%.5f ", parj);
   }
   printf("\n");


   printf("MLE1: \n");
   MLE( par, Pheno, Times, curve_num, ne_size );
   printf("MLE2: \n");
   MLE( par, Pheno, Times, curve_num, ne_size );
   printf("MLE3: \n");
   MLE( par, Pheno, Times, curve_num, ne_size );

   
   gsl_vector_set_zero(par2);
   gsl_vector_view par2_part1 = gsl_vector_subvector(par2, 0, ne_size);
   gsl_vector_view par_part1  = gsl_vector_subvector(par, 0, ne_size);
   gsl_vector_memcpy(&par2_part1.vector, &par_part1.vector);

   double par_v2  = gsl_vector_get(par, ne_size);
   double par_phi = gsl_vector_get(par, ne_size+1);
   for(size_t j = 1; j <= d_size; ++j){
      gsl_vector_set(par2, ne_size+j-1, par_v2);
   }
   gsl_vector_set(par2, ne_size+d_size, par_phi);
   
   printf("MLE4: \n");
   MLE_last( par2, Pheno, Times, curve_num, ne_size );


   gsl_vector_free( par );
   gsl_matrix_free( parameters );
   return;
}


void set_initials(const gsl_matrix *Pheno, const gsl_vector *Times, 
                  const gsl_matrix *xbl, const size_t curve_num, const size_t nrf,
                  const gsl_matrix *Uk, const gsl_vector *eval, 
            gsl_vector *par_vg, gsl_vector *par_vp, gsl_vector *par_vr, 
            gsl_vector *beta, gsl_vector *ai, gsl_vector *pi )
{
   size_t nr1_size = par_vg->size - 1;
   size_t nr2_size = par_vp->size - 1;

   size_t n_size = Pheno->size1;
   size_t d_size = Times->size;
   size_t nd_size = n_size * d_size;
   size_t c_size = xbl->size2;
   size_t ne_size = 0;
  
   get_lop_ne_size(curve_num, nrf, ne_size);
 

   gsl_vector_set_zero( par_vg );
   gsl_vector_set_zero( par_vp );
   gsl_vector_set_zero( par_vr );
   gsl_vector_set_zero( beta );
   gsl_vector_set_zero( ai );
   gsl_vector_set_zero( pi );

   //------ step1: beta_star, u_star ------//
   gsl_vector *par = gsl_vector_alloc(ne_size + d_size + 1);
   Funmap( Pheno, Times, curve_num, nrf, par );
   gsl_vector_view par_fix = gsl_vector_subvector(par, 0, ne_size);

   gsl_vector *beta_star = gsl_vector_alloc( c_size * ne_size);
   gsl_vector *a0_star = gsl_vector_alloc( n_size * (nr1_size+1) );
   gsl_vector *p0_star = gsl_vector_alloc( n_size * (nr2_size+1) );
   gsl_vector_set_zero( beta_star );
   gsl_vector_set_zero(a0_star);
   gsl_vector_set_zero(p0_star);
   
   
   gsl_vector_view beta_part = gsl_vector_subvector(beta_star, 0, ne_size);
   gsl_vector_memcpy(&beta_part.vector, &par_fix.vector);


   //------ step2: W_star, Z10_star, Z20_star ------//
   size_t p_size = c_size * ne_size;
   gsl_matrix *Qmat = gsl_matrix_alloc(n_size, c_size);
   gsl_matrix_memcpy(Qmat, xbl);

   gsl_matrix *W0_star = gsl_matrix_alloc(nd_size, p_size);
   gsl_matrix *Y0_star = gsl_matrix_alloc(nd_size, 1);
   Build_W0YO( nrf, beta_star, a0_star, p0_star, Pheno, Qmat, Times, curve_num, W0_star, Y0_star );

   gsl_matrix *Z10_star = gsl_matrix_alloc(d_size, nr1_size+1);
   gsl_matrix *Z20_star = gsl_matrix_alloc(d_size, nr2_size+1);
   Calc_Z10( Times, Z10_star );
   Calc_Z10( Times, Z20_star );

   //------ step3: betas,a0,p0 ------//
   gsl_matrix *Q = gsl_matrix_alloc(p_size, p_size);
   gsl_vector *J = gsl_vector_alloc(p_size);
   gsl_matrix_set_zero(Q);
   gsl_vector_set_zero(J);
   for(size_t i = 0; i < n_size; ++i){
     size_t start = i*d_size;
     gsl_matrix_view Wi = gsl_matrix_submatrix(W0_star, start,0, d_size, p_size);
     gsl_vector_view Yi = gsl_matrix_subcolumn(Y0_star, 0, i*d_size, d_size);

     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &Wi.matrix, &Wi.matrix, 1.0, Q);
     gsl_blas_dgemv(CblasTrans, 1.0, &Wi.matrix, &Yi.vector, 1.0, J);
   }
   
   gsl_matrix *Qi = gsl_matrix_alloc(p_size, p_size);
   gsl_matrix_set_zero(Qi);
   int sig;
   gsl_permutation *pmt = gsl_permutation_alloc(p_size);
   LUDecomp(Q, pmt, &sig);
   LUInvert(Q, pmt, Qi);
   gsl_matrix_free( Q );
   gsl_permutation_free(pmt);

   //----//--- betas
   gsl_vector *betas = gsl_vector_alloc(p_size);
   gsl_vector_set_zero( betas );
   gsl_blas_dgemv(CblasNoTrans, 1.0, Qi, J, 1.0, betas);   

   //----//--- a0,p0
   gsl_matrix *ZZ1 = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
   gsl_matrix *ZZ1inv = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
   gsl_matrix_set_zero(ZZ1);
   gsl_matrix_set_zero(ZZ1inv);
   gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Z10_star, Z10_star, 0.0, ZZ1);
   int sig1;
   gsl_permutation *pmt1 = gsl_permutation_alloc(nr1_size+1);
   LUDecomp(ZZ1, pmt1, &sig1);
   LUInvert(ZZ1, pmt1, ZZ1inv);
   gsl_matrix_free( ZZ1 );
   gsl_permutation_free( pmt1 );

   gsl_matrix *ZZ2 = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
   gsl_matrix *ZZ2inv = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
   gsl_matrix_set_zero(ZZ2);
   gsl_matrix_set_zero(ZZ2inv);
   gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Z20_star, Z20_star, 0.0, ZZ2);
   int sig2;
   gsl_permutation *pmt2 = gsl_permutation_alloc(nr2_size+1);
   LUDecomp(ZZ2, pmt2, &sig2);
   LUInvert(ZZ2, pmt2, ZZ2inv);
   gsl_matrix_free( ZZ2 );
   gsl_permutation_free( pmt2 );

   gsl_matrix *ZitZi  = gsl_matrix_alloc(nr1_size+1, d_size);
   gsl_matrix *ZitZi2 = gsl_matrix_alloc(nr2_size+1, d_size);
   gsl_matrix_set_zero( ZitZi );
   gsl_matrix_set_zero( ZitZi2 );
   gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, ZZ1inv, Z10_star, 0.0, ZitZi);
   gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, ZZ2inv, Z20_star, 0.0, ZitZi2);

   gsl_vector *a0 = gsl_vector_alloc(  n_size*(nr1_size+1) );
   gsl_vector *p0 = gsl_vector_alloc(  n_size*(nr2_size+1) );
   gsl_vector_set_zero(a0);
   gsl_vector_set_zero(p0);
   gsl_vector *Wbi = gsl_vector_alloc(d_size);
   gsl_vector *YWbi = gsl_vector_alloc(d_size);
   for(size_t i = 0; i < n_size; ++i){
     size_t start = i*d_size;

     gsl_vector_set_zero( Wbi );
     gsl_vector_view Yi = gsl_matrix_subcolumn(Y0_star, 0, start, d_size);
     gsl_matrix_view Wi = gsl_matrix_submatrix(W0_star, start,0, d_size, p_size);
     gsl_blas_dgemv(CblasNoTrans, 1.0, &Wi.matrix, betas, 0.0, Wbi);

     gsl_vector_set_zero( YWbi );
     gsl_vector_add(YWbi, &Yi.vector);
     gsl_vector_sub(YWbi, Wbi);

     gsl_vector_view a0_part = gsl_vector_subvector(a0, i*(nr1_size+1), nr1_size+1);
     gsl_vector_view p0_part = gsl_vector_subvector(p0, i*(nr2_size+1), nr2_size+1);
     gsl_vector_set_zero( &a0_part.vector );
     gsl_vector_set_zero( &p0_part.vector );
     gsl_blas_dgemv(CblasNoTrans, 1.0, ZitZi, YWbi, 0.0, &a0_part.vector);
     gsl_blas_dgemv(CblasNoTrans, 1.0, ZitZi2, YWbi, 0.0, &p0_part.vector);
   }

   //------ step4: sgm_vg, sgm_vp, sgm_vr ------//
   gsl_matrix *sgm_vg = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
   gsl_matrix *sgm_vp = gsl_matrix_alloc(nr2_size+1, nr2_size+1);
   gsl_matrix_set_zero( sgm_vg );
   gsl_matrix_set_zero( sgm_vp );
   for(size_t i = 0; i < n_size; ++i){
      gsl_vector_view a0_i = gsl_vector_subvector(a0, i*(nr1_size+1), nr1_size+1);
      gsl_vector_view p0_i = gsl_vector_subvector(p0, i*(nr2_size+1), nr2_size+1);

      gsl_blas_dger(1.0, &a0_i.vector, &a0_i.vector, sgm_vg );
      gsl_blas_dger(1.0, &p0_i.vector, &p0_i.vector, sgm_vp );
   }
   if(1 == 1){
     gsl_matrix_scale(sgm_vg, 1.0/n_size/2.0);
     gsl_matrix_scale(sgm_vp, 1.0/n_size/2.0);
     gsl_vector_view sgm_vg_diag = gsl_matrix_diagonal( sgm_vg ); 
     gsl_vector_view sgm_vp_diag = gsl_matrix_diagonal( sgm_vp ); 
     gsl_vector_memcpy( par_vg, &sgm_vg_diag.vector );
     gsl_vector_memcpy( par_vp, &sgm_vp_diag.vector );
     gsl_vector_view par_last = gsl_vector_subvector(par, ne_size, d_size+1);
     gsl_vector_memcpy( par_vr, &par_last.vector );
     gsl_vector_set_all( par_vg, gsl_vector_get(par_vr,0)/20 );
     gsl_vector_set_all( par_vp, gsl_vector_get(par_vr,0)/20 );
   }else{
     gsl_matrix_scale(sgm_vg, 1.0/n_size/2.0);
     gsl_matrix_scale(sgm_vp, 1.0/n_size/2.0);
     gsl_vector_view sgm_vg_diag = gsl_matrix_diagonal( sgm_vg ); 
     gsl_vector_view sgm_vp_diag = gsl_matrix_diagonal( sgm_vp );
     gsl_vector_memcpy( par_vg, &sgm_vg_diag.vector );
     gsl_vector_memcpy( par_vp, &sgm_vp_diag.vector );

     gsl_matrix *Z10G    = gsl_matrix_alloc(d_size, nr1_size+1);
     gsl_matrix *Z10GZ10 = gsl_matrix_alloc(d_size, d_size);
     gsl_matrix *Z20P    = gsl_matrix_alloc(d_size, nr2_size+1);
     gsl_matrix *Z20PZ20 = gsl_matrix_alloc(d_size, d_size);
     gsl_matrix_set_zero( Z10G );
     gsl_matrix_set_zero( Z10GZ10 );
     gsl_matrix_set_zero( Z20P );
     gsl_matrix_set_zero( Z20PZ20 );

     gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Z10_star, sgm_vg, 1.0, Z10G);
     gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Z20_star, sgm_vp, 1.0, Z20P);
     gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Z10G, Z10_star, 1.0, Z10GZ10);
     gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Z20P, Z20_star, 1.0, Z20PZ20);

     gsl_vector_view Z10GZ10_diag = gsl_matrix_diagonal( Z10GZ10 ); 
     gsl_vector_view Z20PZ20_diag = gsl_matrix_diagonal( Z20PZ20 ); 

     printf("\nZ10GZ10_diag: ");
     for(size_t i = 0; i < d_size; ++i){
      printf( "%.4f ", gsl_vector_get(&Z10GZ10_diag.vector,i));
     }
     printf("\nZ20PZ20_diag: ");
     for(size_t i = 0; i < d_size; ++i){
      printf( "%.4f ", gsl_vector_get(&Z20PZ20_diag.vector,i));
     }

     gsl_vector_view par_last = gsl_vector_subvector(par, ne_size, d_size+1);
     printf("\npar_vr0:      ");
     for(size_t i = 0; i <= d_size; ++i){
      printf( "%.4f ", gsl_vector_get(&par_last.vector,i) );
     }

     gsl_vector_view par_vr1 = gsl_vector_subvector(par, ne_size, d_size);   
     gsl_vector_sub( &par_vr1.vector, &Z10GZ10_diag.vector );
     gsl_vector_sub( &par_vr1.vector, &Z20PZ20_diag.vector );
     printf("\npar_vr1:      ");
     for(size_t i = 0; i < d_size; ++i){
      printf( "%.4f ", gsl_vector_get(&par_vr1.vector,i));
     }
     printf("\n");
     double phi = gsl_vector_get(par, ne_size+d_size);
   
     gsl_vector_view par_vr_front = gsl_vector_subvector(par_vr, 0, d_size);
     gsl_vector_memcpy( &par_vr_front.vector, &par_vr1.vector );
     gsl_vector_set( par_vr, d_size, phi );
   
     for(size_t i = 0; i < d_size; ++i){
      double tmp = gsl_vector_get(par_vr,i);
      if(tmp <= 0){ gsl_vector_set(par_vr,i, 0.01 ); }
     }
    

     gsl_matrix_free(Z10G);
     gsl_matrix_free(Z10GZ10);
     gsl_matrix_free(Z20P);
     gsl_matrix_free(Z20PZ20);
   }
   


   gsl_vector_memcpy( beta, betas ); 
   gsl_vector_memcpy( ai, a0);
   gsl_vector_memcpy( pi, p0);


   //------ free -----//
   gsl_vector_free( par );
   gsl_vector_free( beta_star );
   gsl_vector_free( a0_star );
   gsl_vector_free( p0_star );

   gsl_matrix_free( W0_star );
   gsl_matrix_free( Y0_star );
   gsl_matrix_free( Z10_star );
   gsl_matrix_free( Z20_star );

   gsl_vector_free( J );
   gsl_matrix_free( Qi );

   gsl_vector_free( betas );
   gsl_vector_free( a0 );
   gsl_vector_free( p0 );

   gsl_matrix_free( ZZ1inv );
   gsl_matrix_free( ZZ2inv );
   gsl_matrix_free( ZitZi );
   gsl_matrix_free( ZitZi2 );

   gsl_vector_free( Wbi );
   gsl_vector_free( YWbi );

   gsl_matrix_free( Qmat );
   gsl_matrix_free( sgm_vg );
   gsl_matrix_free( sgm_vp );
   //------ end ------//
   return;
}


void Calc_Bij( const gsl_matrix *X, const gsl_matrix *F1, const gsl_matrix *M, const gsl_vector *eval, 
               const gsl_matrix *ZGi, const size_t i, const size_t j, 
         gsl_matrix *Bij)
{
  size_t     dn_size = X->size1;
  size_t      p_size = X->size2;
  size_t      n_size = eval->size;
  size_t      d_size = dn_size / n_size;
  size_t    nr1_size = ZGi->size1-1;
  
  gsl_matrix *XiF1    = gsl_matrix_alloc(d_size, p_size); 
  gsl_matrix *XiF1Xj  = gsl_matrix_alloc(d_size, d_size);
  gsl_matrix *MXiF1Xj = gsl_matrix_alloc(nr1_size+1, d_size);
  
  gsl_matrix_set_zero( XiF1 );
  gsl_matrix_set_zero( XiF1Xj );
  gsl_matrix_set_zero( MXiF1Xj );
  gsl_matrix_set_zero(Bij);
  

  gsl_matrix_const_view ZGi_sub = gsl_matrix_const_submatrix(ZGi, 0, i*(nr1_size+1), nr1_size+1, nr1_size+1);
  gsl_matrix_const_view Mi_sub  = gsl_matrix_const_submatrix(M, 0, i*(d_size), nr1_size+1, d_size);
  gsl_matrix_const_view X_i     = gsl_matrix_const_submatrix(X, i*d_size, 0, d_size, p_size);
  
  if (i == j ) { 
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,   &X_i.matrix,              F1, 0.0, XiF1 );
  gsl_blas_dgemm(CblasNoTrans,   CblasTrans, 1.0,          XiF1,     &X_i.matrix, 0.0, XiF1Xj );
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &Mi_sub.matrix,         XiF1Xj, 0.0, MXiF1Xj );
  gsl_blas_dgemm(CblasNoTrans,   CblasTrans, 1.0,        MXiF1Xj, &Mi_sub.matrix, 0.0, Bij );
  gsl_matrix_add(Bij, &ZGi_sub.matrix);
  
  }else {
  gsl_matrix_const_view Mj_sub = gsl_matrix_const_submatrix(M, 0, j*(d_size), nr1_size+1, d_size);
  gsl_matrix_const_view X_j    = gsl_matrix_const_submatrix(X, j*d_size, 0, d_size, p_size);
  
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,      &X_i.matrix,                  F1, 0.0, XiF1 );
    gsl_blas_dgemm(CblasNoTrans,    CblasTrans, 1.0,              XiF1,       &X_j.matrix, 0.0, XiF1Xj );
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,  &Mi_sub.matrix,            XiF1Xj, 0.0, MXiF1Xj );
    gsl_blas_dgemm(CblasNoTrans,    CblasTrans, 1.0,           MXiF1Xj, &Mj_sub.matrix, 0.0, Bij );
  
  }
    gsl_matrix_free(XiF1);
  gsl_matrix_free(XiF1Xj);
  gsl_matrix_free(MXiF1Xj);
  
}

void calc_chisq(  const gsl_matrix *X, const gsl_matrix *Uk, const gsl_vector *eval, const gsl_matrix *S, 
                    const gsl_vector *par_vg,  const gsl_matrix *F1, const gsl_matrix *M,
          const gsl_vector *ai, const gsl_matrix *ZGi, 
          gsl_vector *ui, gsl_vector *chisq, gsl_vector *pval  ) 
{ 
  size_t n_size    = eval->size;
  size_t dn_size   = X->size1;
  size_t p_size    = X->size2;
  size_t d_size    = S->size2;
  size_t uniG_size = par_vg->size;
  size_t nr1_size  = uniG_size - 1;

  gsl_matrix *Ki = gsl_matrix_alloc(n_size, n_size);
  gsl_matrix *Dk = gsl_matrix_alloc(n_size, n_size);
  gsl_matrix *temp = gsl_matrix_alloc(n_size, n_size);
  gsl_matrix *par_vg_diag = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
  
  printf("Uk: \n");
        for (size_t row1 = 0; row1 < 5 ; ++row1) {
          for (size_t col1 = 0; col1 < 5 ; ++col1) {
           double val1 = gsl_matrix_get(Uk, row1, col1);
           printf("%lf ", val1);
          }
         printf("\n");
    }


  gsl_matrix_set_zero( Dk );
  gsl_matrix_set_zero( Ki );
  gsl_matrix_set_zero( temp );
  gsl_matrix_set_zero(par_vg_diag );  
  printf("chisq... 1 \n");


//-- Compute the Ki--//

  for (size_t i = 0; i < n_size; ++i) {
    double delta = 1.0 / gsl_vector_get(eval, i);
    gsl_matrix_set(Dk, i,i, delta);
  }
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Uk, Dk, 0.0, temp);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, temp, Uk, 0.0, Ki);
  
  gsl_matrix_free(Dk);
  gsl_matrix_free(temp);

  printf("chisq... 2 \n");


  printf("Ki within calc_chisq: \n");
        for (size_t row1 = 0; row1 < 10 ; ++row1) {
          for (size_t col1 = 0; col1 < 10 ; ++col1) {
           double val1 = gsl_matrix_get(Ki, row1, col1);
           printf("%lf ", val1);
          }
         printf("\n");
    }




  printf("chisq... 3 \n");

   for (size_t j = 0; j < uniG_size; ++j) {
        double value = gsl_vector_get(par_vg, j);
        gsl_matrix_set(par_vg_diag, j, j, value);
    }
    printf("chisq... 4 \n");
  //--Var_u--part1--//
  printf("dsize = %zu \n", d_size);
  

  gsl_matrix *SKi  = gsl_matrix_alloc(d_size, n_size);
  gsl_vector *SKiS = gsl_vector_alloc(d_size);
  gsl_matrix_set_zero( SKi );
  gsl_vector_set_zero( SKiS );
  gsl_blas_dgemm(CblasTrans,   CblasNoTrans, 1.0,   S, Ki, 0.0, SKi);
  

  for (size_t iii = 0; iii < d_size; ++iii) { 
    double result = 0.0;

    gsl_vector_const_view row_SKi = gsl_matrix_const_row(SKi, iii);
    gsl_vector_const_view col_S = gsl_matrix_const_column(S, iii);

    gsl_blas_ddot(&row_SKi.vector, &col_S.vector, &result);
    gsl_vector_set(SKiS, iii, result);

}
  

  printf("chisq....7 SKiS: \n");
  for (size_t jj = 0; jj < 100; ++jj) {
      double val = gsl_vector_get(SKiS, jj);
      printf("%d   %lf \n", jj, val);
  } 
  



  gsl_matrix *partone  = gsl_matrix_alloc(nr1_size+1, (nr1_size+1)*d_size);
  gsl_matrix_set_zero( partone );
  printf("chisq... 8 \n");

   for (size_t i = 0; i < d_size; ++i) {
   double SKiS_ii = gsl_vector_get(SKiS, i);
     gsl_matrix_view partone_sub = gsl_matrix_submatrix(partone, 0, i*(nr1_size+1), nr1_size+1, nr1_size+1);  
    gsl_matrix_add(&partone_sub.matrix, par_vg_diag );
    gsl_matrix_scale(&partone_sub.matrix, SKiS_ii);   

   
   }
    
    gsl_vector_free(SKiS);
    printf("chisq... 9 \n");
  
  gsl_matrix *parttwo  = gsl_matrix_alloc(nr1_size+1, (nr1_size+1)*d_size);
  gsl_matrix_set_zero( parttwo );
 
  printf("chisq... 10 \n");

 
  for (size_t k = 0; k < d_size; ++k) {
    gsl_matrix *Yij = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
    gsl_matrix *Bij = gsl_matrix_alloc(nr1_size+1, nr1_size+1);

    gsl_vector_const_view Sk_row  = gsl_matrix_const_column(S, k);
    gsl_vector_const_view Sk_col  = gsl_matrix_const_column(S, k); 
    gsl_matrix_view parttwo_sub = gsl_matrix_submatrix(parttwo, 0, k*(nr1_size+1), nr1_size+1, nr1_size+1);     
    
    time_t current_time = time(0);
    tm* local_time = localtime(&current_time);
    printf("chisq... 11, snp_%d", k); 
    printf("    %s", asctime(local_time) );

    for (size_t i = 0; i < n_size; ++i) {
      gsl_vector_const_view Uki_col = gsl_matrix_const_column(Uk, i);
      double delta_i  = gsl_vector_get(eval, i);
      double tmp2 = 0.0;
      gsl_blas_ddot( &Uki_col.vector, &Sk_row.vector, &tmp2);

      for (size_t j = 0; j < n_size; ++j) {         
         gsl_matrix_set_zero( Yij );
        
         Calc_Bij( X, F1, M,  eval, ZGi, i, j,    Bij);
       

        double delta_j  = gsl_vector_get(eval, j);
        double delta_ij = 1.0 / (delta_i * delta_j);

        double tmp1 = 0.0;        
        double tmp  = 0.0;
    
        gsl_matrix_add(Yij, Bij);
        gsl_matrix_scale(Yij, delta_ij);
    
        
        gsl_vector_const_view Ukj_row = gsl_matrix_const_column(Uk, j);     
        gsl_blas_ddot( &Ukj_row.vector, &Sk_col.vector, &tmp1);
        
        tmp = tmp1 * tmp2;    
        gsl_matrix_scale(Yij, tmp);
        gsl_matrix_add(&parttwo_sub.matrix, Yij);
        
      }

      }

      printf("k = %d \n", k);
     
    gsl_matrix_free(Bij);
    gsl_matrix_free(Yij);
  }
  
 
   
    printf("test... 13 \n"); 
  
    gsl_matrix *Var_u_diag  = gsl_matrix_alloc(nr1_size+1, (nr1_size+1)*d_size);
    gsl_matrix_set_zero( Var_u_diag );
  
    for (size_t i = 0; i < d_size; ++i) {
      gsl_matrix_view Var_u_diag_sub = gsl_matrix_submatrix(Var_u_diag, 0, i*(nr1_size+1), nr1_size+1, nr1_size+1); 
      gsl_matrix_const_view partone_sub = gsl_matrix_const_submatrix(partone, 0, i*(nr1_size+1), nr1_size+1, nr1_size+1); 
      gsl_matrix_const_view parttwo_sub = gsl_matrix_const_submatrix(parttwo, 0, i*(nr1_size+1), nr1_size+1, nr1_size+1); 
      gsl_matrix_add( &Var_u_diag_sub.matrix, &partone_sub.matrix );
      gsl_matrix_sub( &Var_u_diag_sub.matrix, &parttwo_sub.matrix );

  }

    gsl_matrix_free(partone);
    gsl_matrix_free(parttwo);
    printf("test... 14\n"); 
  
 
  gsl_matrix *I    = gsl_matrix_alloc(nr1_size+1, nr1_size+1);
  gsl_vector *temp11 = gsl_vector_alloc(nr1_size+1);
  gsl_vector_set_zero( ui );
  gsl_matrix_set_identity(I);
  gsl_vector_set_zero( temp11 );
  printf("test... 15\n"); 
  
  for (size_t i = 0; i < d_size; ++i) {
    gsl_vector_view u_sub   = gsl_vector_subvector(ui, i*(nr1_size+1), nr1_size+1);
  
    for(size_t j = 0; j < n_size; ++j){
      gsl_vector_const_view ai_sub   = gsl_vector_const_subvector(ai, j*(nr1_size+1), nr1_size+1);
      double SKij = gsl_matrix_get(SKi, i, j);
      gsl_blas_dgemv(CblasNoTrans, SKij, I, &ai_sub.vector,  0, temp11 );
      gsl_vector_add(&u_sub.vector, temp11);    
    }

    printf("ui_%d : ", i );
    for (size_t ii = 0; ii < nr1_size+1; ii++) {
        double value = gsl_vector_get(&u_sub.vector, ii);
        printf(" %f\n",  ii, value);
    }
    printf("\n");
  }

  printf("test_16 ...\n"); 
  gsl_matrix_free(I);
  gsl_vector_free(temp11);
  gsl_matrix_free(SKi);
  
  

   gsl_matrix *Var_u_inv = gsl_matrix_alloc( nr1_size+1, (nr1_size+1)*d_size );
   gsl_vector *Varu_u    = gsl_vector_alloc(nr1_size+1);
   
   gsl_permutation *pmt1 = gsl_permutation_alloc( nr1_size+1 );
   printf("test_17...\n"); 
   gsl_matrix_set_zero( Var_u_inv );
   gsl_vector_set_zero( Varu_u);
   gsl_vector_set_zero( chisq );
   printf("chisq-- start \n");

   for (size_t i = 0; i < d_size; ++i) {
     gsl_matrix_view Var_u_diag_sub      = gsl_matrix_submatrix(Var_u_diag, 0, i*(nr1_size+1), nr1_size+1, nr1_size+1);  
     gsl_matrix_view Var_u_inv_sub        = gsl_matrix_submatrix(Var_u_inv,  0, i*(nr1_size+1), nr1_size+1, nr1_size+1);
     gsl_vector_const_view u_sub          = gsl_vector_const_subvector(ui, i*(nr1_size+1), nr1_size+1);

     double chisq_sub=0.0;     
   
     printf("Var_u_diag_%d: \n", i);
     for (size_t row = 0; row < nr1_size + 1; ++row) {
       for (size_t col = 0; col < nr1_size + 1; ++col) {
            double val = gsl_matrix_get(&Var_u_diag_sub.matrix, row, col);
            printf("%lf ", val);
        }
        printf("\n");
     }

     int sig;
     LUDecomp(&Var_u_diag_sub.matrix, pmt1, &sig);
     LUInvert(&Var_u_diag_sub.matrix, pmt1, &Var_u_inv_sub.matrix);
      
     gsl_blas_dgemv(CblasNoTrans, 1.0, &Var_u_inv_sub.matrix, &u_sub.vector, 0, Varu_u );
     gsl_blas_ddot( &u_sub.vector, Varu_u, &chisq_sub);

     gsl_vector_set(chisq, i, chisq_sub );               
   }
   printf("chisq-- finished \n");
    

    
    for (size_t i = 0; i < d_size; ++i) {
    
      double pval_i  = 0.0;
      double chisq_i = gsl_vector_get(chisq, i);
      pval_i = gsl_cdf_chisq_Q(chisq_i, (double)nr1_size+1 );
      gsl_vector_set(pval, i, pval_i );
  }

  printf("chisq_printf_chisq_pval: \n");
  for (size_t row = 0; row < d_size; ++row) {
    double val1 = gsl_vector_get(chisq, row);
    double val2 = gsl_vector_get(pval, row);
    printf("%d:    %lf \n", row, val1, val2);
  }

  
  //-- free --//
  
  gsl_permutation_free( pmt1 );
  gsl_vector_free(Varu_u);
  gsl_matrix_free(Var_u_diag); 

  return;
}

void step1_step2S( const gsl_matrix *Pheno, const gsl_vector *Times, const gsl_matrix *xbl, 
                    const size_t curve_num, const gsl_matrix *Uk, const gsl_vector *eval, const gsl_matrix *S, 
                    const size_t nrf, const size_t max_iter, const double cc_par, const double cc_gra, 
              gsl_vector *par_vg, gsl_vector *par_vp, gsl_vector *par_vr, 
              gsl_vector *beta, gsl_vector *ai, gsl_vector *pi,
              double &log_RL, double &pvalue, gsl_vector *logRL_logL_2parts,
            gsl_vector *chisq, gsl_vector *pval, gsl_vector *ui )
{
    size_t n_size = Pheno->size1;
    size_t d_size = Times->size;
    size_t c_size = xbl->size2;
    size_t ne_size = 0;
    get_lop_ne_size(curve_num, nrf, ne_size);
    size_t npar_bsnp = ne_size; 

   
    gsl_matrix *X = gsl_matrix_alloc(d_size * n_size, c_size * ne_size);
    gsl_matrix_set_zero( X );
    gsl_vector_set_zero( par_vg );
    gsl_vector_set_zero( par_vp );
    gsl_vector_set_zero( par_vr );
    gsl_vector_set_zero( beta );
    gsl_vector_set_zero( ai );
    gsl_vector_set_zero( pi );

    gsl_matrix *Qmat = gsl_matrix_alloc(n_size, c_size);
    gsl_matrix_memcpy(Qmat, xbl);
    log_RL = 0.0;
    pvalue = 1.0;
    gsl_vector_set_zero( logRL_logL_2parts );
    
    printf("begin \n");

    //--- step1: set initial values for IAE_iteration --//
    set_initials( Pheno, Times, xbl, curve_num, nrf, Uk, eval, 
                  par_vg, par_vp, par_vr, beta, ai, pi );

     printf("end \n");
    printf("beta: \n");
    for(size_t i = 0; i < c_size; ++i){ 
      for(size_t j = 0; j < ne_size; ++j){ 
        printf("%.5f ", gsl_vector_get(beta,i*ne_size+j)); 
      }
      printf("\n");
    }

    printf("par_vg: ");
    for(size_t i = 0; i < par_vg->size; ++i){ printf("%.5f ", gsl_vector_get(par_vg,i)); }
    printf("\n");

    printf("par_vp: ");
    for(size_t i = 0; i < par_vp->size; ++i){ printf("%.5f ", gsl_vector_get(par_vp,i)); }
    printf("\n");

    printf("par_vr: ");
    for(size_t i = 0; i < (d_size+1); ++i){ printf("%.5f ", gsl_vector_get(par_vr,i)); }
    printf("\n");    
    printf("\n");
    
    size_t h0 = 1.0;
   
    gsl_vector_set(par_vr, d_size, 0.001);
    iteration( nrf, max_iter, cc_par, cc_gra, npar_bsnp, 
                   curve_num, Pheno, Qmat, Uk, eval, Times,
                   h0, par_vg, par_vp, par_vr, X, beta, ai, pi, 
                   log_RL, pvalue, logRL_logL_2parts );
    
    printf("pi: ");
       for (size_t i = 0; i < pi->size; ++i) {
    printf("%.5f ", gsl_vector_get(pi, i)); 
              }
    printf("\n");

    gsl_matrix_free( Qmat ); 
    
    printf("test step2 \n");
  
    gsl_matrix *Z10    = gsl_matrix_alloc(d_size, par_vg->size);
    gsl_matrix *Z20    = gsl_matrix_alloc(d_size, par_vp->size);
    gsl_matrix *G0     = gsl_matrix_alloc(par_vg->size, par_vg->size);
    gsl_matrix *P0     = gsl_matrix_alloc(par_vp->size, par_vp->size);
    gsl_matrix *R0     = gsl_matrix_alloc(d_size, d_size);
    gsl_matrix *G0_inv = gsl_matrix_alloc(par_vg->size, par_vg->size);
    gsl_matrix *P0_inv = gsl_matrix_alloc(par_vp->size, par_vp->size);
    gsl_matrix *R0_inv = gsl_matrix_alloc(d_size, d_size);
    gsl_matrix *ZGi    = gsl_matrix_alloc(par_vg->size, (par_vg->size) * n_size);
    gsl_matrix *p22p22p12ZGip12p22 = gsl_matrix_alloc(par_vp->size, (par_vp->size) * n_size);
    gsl_matrix *final  = gsl_matrix_alloc(d_size, d_size * n_size);
    gsl_matrix *M      = gsl_matrix_alloc(par_vg->size, d_size * n_size);
    gsl_matrix *N      = gsl_matrix_alloc(d_size, (par_vp->size) * n_size);
    gsl_matrix *F1     = gsl_matrix_alloc(c_size * ne_size, c_size * ne_size);
    size_t i = 0;
    size_t j = 0;

    Calc_Z10_Z20_G0_G0inv_P0_P0inv_R0_R0inv( par_vg, par_vp, par_vr, Times, 
                                             Z10, Z20, G0, P0, R0, G0_inv, P0_inv, R0_inv );
                       
    Calc_alltemp( eval, Z10, Z20, G0, G0_inv, R0_inv, P0_inv, 
             ZGi, p22p22p12ZGip12p22, final, M, N );
       
    Calc_F1( X, R0_inv, final, F1 );  
    
    printf("tset step3 \n");
    calc_chisq( X, Uk, eval, S, par_vg,  F1, M, ai, ZGi, 
               ui, chisq, pval );

    printf("tset chisq \n");
  
    //-- free --//
    gsl_matrix_free( Z10 );
    gsl_matrix_free( Z20 );
    gsl_matrix_free( G0 ); 
    gsl_matrix_free( P0 ); 
    gsl_matrix_free( R0 ); 
    gsl_matrix_free( G0_inv );
    gsl_matrix_free( P0_inv );
    gsl_matrix_free( R0_inv );

    gsl_matrix_free( ZGi ); 
    gsl_matrix_free( p22p22p12ZGip12p22 );
    gsl_matrix_free( final );
    gsl_matrix_free( M );
    gsl_matrix_free( N );

    gsl_matrix_free( F1 );
    gsl_matrix_free( X );
   
    
    return;
} 


void MVLMM::AnalyzePlink_0S(const gsl_matrix *U, const gsl_vector *eval, const gsl_matrix *S, 
                           const gsl_matrix *Y, const gsl_vector *Times, 
                           const gsl_matrix *W, const size_t curve_num ) 
{
  size_t n_size = Y->size1, d_size = Y->size2, c_size = W->size2;
  size_t ne_size = 0;
  get_lop_ne_size( curve_num, nrf, ne_size );

 
  gsl_matrix *Dk = gsl_matrix_alloc(n_size, n_size);
  gsl_matrix *temp = gsl_matrix_alloc(n_size, n_size);
  gsl_matrix *Ki = gsl_matrix_alloc(n_size, n_size);

  gsl_matrix_set_zero( Dk );
  gsl_matrix_set_zero( temp );
  gsl_matrix_set_zero( Ki );

//----Compute the Ki---//
  for (size_t i = 0; i < n_size; ++i) {
    double delta = 1.0 / gsl_vector_get(eval, i);
    gsl_matrix_set(Dk, i,i, delta);
  }
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, U, Dk, 0.0, temp);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, temp, U, 0.0, Ki);
  
  gsl_matrix_free(Dk);
  gsl_matrix_free(temp);


  printf("AnalyzePlink_0S\n");
  printf("AnalyzePlink_0S, Times: ");
  for(size_t i=0; i<d_size; i++){
    printf("%.4f ", gsl_vector_get(Times,i) );
  }
  printf("\n");
  
  gsl_vector *par_vg_null = gsl_vector_alloc(nr1_size+1); 
  gsl_vector *par_vp_null = gsl_vector_alloc(nr2_size+1); 
  gsl_vector *par_vr_null = gsl_vector_alloc(d_size+1); 
  gsl_vector *beta_null = gsl_vector_alloc(c_size * ne_size); 
  gsl_vector *ai_null = gsl_vector_alloc(n_size * (nr1_size+1)); 
  gsl_vector *pi_null = gsl_vector_alloc(n_size * (nr2_size+1)); 
  double log_RL_null = 0.0, pvalue_null = 0.0;
  gsl_vector *logRL_logL_2parts_null = gsl_vector_alloc(4); 
  double log_L_null = 0.0;

  gsl_matrix *Pheno = gsl_matrix_alloc( Y->size1, Y->size2 );
  gsl_matrix *xbl = gsl_matrix_alloc( W->size1, W->size2 );
  
  gsl_matrix_memcpy(Pheno, Y);
  gsl_matrix_memcpy(xbl, W);
  
  

  size_t snp_n = S->size2;
  gsl_vector *chisq = gsl_vector_alloc( snp_n );
  gsl_vector_set_zero( chisq );

  gsl_vector *pval = gsl_vector_alloc( snp_n );
  gsl_vector_set_zero( pval );
        

  gsl_vector *ui    = gsl_vector_alloc( snp_n*(nr1_size+1) );   
  gsl_vector_set_zero( ui );
  
  printf("AnalyzePlink_0S, nr1_size = %d\n", nr1_size);
  printf("AnalyzePlink_0S, snp_n = %d\n", snp_n);  
  
   
  step1_step2S( Pheno, Times, xbl,  curve_num, U, eval, S, 
                    nrf, max_iter, cc_par, cc_gra, 
               par_vg_null, par_vp_null, par_vr_null, beta_null, ai_null, pi_null, 
               log_RL_null, pvalue_null,  logRL_logL_2parts_null,
               chisq, pval, ui );
  log_L_null = gsl_vector_get(logRL_logL_2parts_null, 2) + gsl_vector_get(logRL_logL_2parts_null, 3);
  
  printf("step1_step2S-----END");
  
  // Prepare
  debug_msg("entering");
  string file_bed = file_bfile + ".bed";
  ifstream infile(file_bed.c_str(), ios::binary);
  if (!infile) {
    cout << "error reading bed file:" << file_bed << endl;
    return;
  }

  clock_t time_start = clock();
  time_opt = 0;

  char ch[1];
  bitset<8> b;

  // Create a large matrix.
  size_t msize = S->size2;
  gsl_matrix *Xlarge = gsl_matrix_alloc(U->size1, msize);
  gsl_matrix_set_zero(Xlarge);

  int n_bit, n_miss, ci_total, ci_test;
  double geno, x_mean;
  gsl_vector *x      = gsl_vector_alloc(n_size); 


  // Start reading genotypes and analyze.
  if (ni_total % 4 == 0) {
    n_bit = ni_total / 4;
  } else {
    n_bit = ni_total / 4 + 1;
  }

  for (int i = 0; i < 3; ++i) {
    infile.read(ch, 1);
    b = ch[0];
  }

  size_t csnp = 0, t_last = 0;
  for (size_t t = 0; t < indicator_snp.size(); ++t) {
    if (indicator_snp[t] == 0) {
      continue;
    }
  
    t_last++;
  }


 
  for (vector<SNPINFO>::size_type t = 0; t < snpInfo.size(); ++t) {
    if (t % d_pace == 0 || t == snpInfo.size() - 1) {
  
    }
    if (indicator_snp[t] == 0) {
      continue;
    }

    infile.seekg(t * n_bit + 3);

    x_mean = 0.0;
    n_miss = 0;
    ci_total = 0;
    ci_test = 0;
    for (int i = 0; i < n_bit; ++i) {
      infile.read(ch, 1);
      b = ch[0];

      for (size_t j = 0; j < 4; ++j) {
        if ((i == (n_bit - 1)) && ci_total == (int)ni_total) {
          break;
        }
        if (indicator_idv[ci_total] == 0) {
          ci_total++;
          continue;
        }

        if (b[2 * j] == 0) {
          if (b[2 * j + 1] == 0) {
            gsl_vector_set(x, ci_test, 2);
            x_mean += 2.0;
          } else {
            gsl_vector_set(x, ci_test, 1);
            x_mean += 1.0;
          }
        } else {
          if (b[2 * j + 1] == 1) {
            gsl_vector_set(x, ci_test, 0);
          } else {
            gsl_vector_set(x, ci_test, -9);
            n_miss++;
          }
        }

        ci_total++;
        ci_test++;
      }
    }

    csnp++;
    if (csnp % msize == 0 || csnp == t_last) {    
  
      for (size_t i = 0; i < t_last; i++) {
       
          vector<double> v_ui_i;

          for (size_t ii = 0; ii <= nr1_size; ii++) { v_ui_i.push_back( gsl_vector_get(ui, i*(nr1_size+1)+ii) ); }
          double v_chisq = gsl_vector_get( chisq, i );
          double v_pval  = gsl_vector_get( pval, i );

         

          MPHSUMSTAT_yxx SNPsyxx = {v_chisq, v_pval, v_ui_i};
          sumStat0.push_back(SNPsyxx);
            
          
          v_ui_i.clear();
       
      }
    }
  }
  cout << endl;

  infile.close();
  infile.clear();



  
  Vg_remle_null.clear();
  Vp_remle_null.clear();
  Ve_remle_null.clear();
  beta_remle_null.clear();

  
  for (size_t i = 0; i <= nr1_size; i++) {
    Vg_remle_null.push_back(gsl_vector_get(par_vg_null, i));
  }
  for (size_t i = 0; i <= nr2_size; i++) {
    Vp_remle_null.push_back(gsl_vector_get(par_vp_null, i));
  }
  for (size_t i = 0; i <= d_size; i++) {
    Ve_remle_null.push_back(gsl_vector_get(par_vr_null, i));
  }
  for (size_t i = 0; i < (c_size*ne_size); i++) {
    beta_remle_null.push_back(gsl_vector_get(beta_null, i));
  }
  logl_mle_H0 = log_L_null;
  logl_remle_H0 = log_RL_null;


  cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  cout.precision(4);
  cout << endl;
  cout << "REMLE estimate for Vg in the null model: " << endl;
  for (size_t j = 0; j <= nr1_size; j++) {
    cout << gsl_vector_get(par_vg_null, j) << "\t";
  }
  cout << endl;

  cout << "REMLE estimate for Vp in the null model: " << endl;
  for (size_t j = 0; j <= nr2_size; j++) {
    cout << gsl_vector_get(par_vp_null, j) << "\t";
  }
  cout << endl;
  
  cout << "REMLE estimate for Vr in the null model: " << endl;
  for (size_t j = 0; j <= d_size; j++) {
    cout << gsl_vector_get(par_vr_null, j) << "\t";
  }
  cout << endl;

  cout << "REMLE estimate for beta in the null model: " << endl;
  for (size_t j = 0; j < (c_size*ne_size); j++) {
    cout << gsl_vector_get(beta_null, j) << "\t";
  }
  cout << endl;
  cout << "REMLE likelihood = " << log_RL_null << endl;
  cout << "MLE likelihood = " << log_L_null << endl;
  cout << endl;

  
  gsl_vector_free( par_vg_null ); 
  gsl_vector_free( par_vp_null ); 
  gsl_vector_free( par_vr_null ); 
  gsl_vector_free( beta_null ); 
  gsl_vector_free( ai_null ); 
  gsl_vector_free( pi_null ); 
  gsl_vector_free( logRL_logL_2parts_null ); 

  gsl_matrix_free( Pheno );
  gsl_matrix_free( xbl );

  gsl_vector_free( x ); 
  gsl_matrix_free( Xlarge );

  
  gsl_vector_free( chisq );
  gsl_vector_free( pval );
  gsl_vector_free( ui );
  
  return;
}











