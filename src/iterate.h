/*
    Dynamic Genomic Best linear Unbiased Prediction  (DynGBLUP)
    Copyright © 2026, Meixia Ye
    
*/

#ifndef __MVLMM_H__
#define __MVLMM_H__

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"

#include "io.h"
#include "param.h"

#define LMM_BATCH_SIZE 2300 

using namespace std;

class MVLMM {

public:
 
  int a_mode;    
  size_t d_pace; 

  string file_bfile;
  string file_geno;
  string file_pheno; 
  string file_anno;  
  string file_cvt;   
  string file_kin;   
  string file_S;
  string file_ku, file_kd; 
  string file_out;
  string path_out;

  
  double logl_mle_H0, logl_remle_H0;
  vector<double> Vg_remle_null, Vp_remle_null, Ve_remle_null;
  vector<double> beta_remle_null, ai_remle_null, pi_remle_null;
 
  size_t nrf, nr1_size, nr2_size;
  size_t max_iter;
  double cc_par, cc_gra;
  
  
  size_t curve_type;
  
  size_t ni_total, ni_test; 
  size_t ns_total, ns_test; 
  size_t n_cvt;
  size_t n_ph;
  double time_opt; 

  
  vector<int> indicator_idv;
  vector<int> indicator_snp;
  vector<SNPINFO> snpInfo;
  vector<MPHSUMSTAT> sumStat; 
  vector<MPHSUMSTAT_yxx> sumStat0; 
  

  vector<int> res_xh;

  void CopyFromParam(PARAM &cPar);
  void CopyToParam(PARAM &cPar);
  void AnalyzePlink_0S(const gsl_matrix *U, const gsl_vector *eval, const gsl_matrix *S, 
                           const gsl_matrix *Y, const gsl_vector *Times, 
                           const gsl_matrix *W, const size_t curve_num );


  void WriteFiles_yxx();
  void WriteFiles_h2();
};


void step1_step2S( const gsl_matrix *Pheno, const gsl_vector *Times, const gsl_matrix *xbl, 
                    const size_t curve_num, const gsl_matrix *Uk, const gsl_vector *eval, const gsl_matrix *S, 
                    const size_t nrf, const size_t max_iter, const double cc_par, const double cc_gra, 
              gsl_vector *par_vg, gsl_vector *par_vp, gsl_vector *par_vr, 
              gsl_vector *beta, gsl_vector *ai, gsl_vector *pi, 
              double &log_RL, double &pvalue, gsl_vector *logRL_logL_2parts,
            gsl_vector *chisq, gsl_vector *pval, gsl_vector *ui );

void iteration( const size_t nrf, const size_t maxiter, const double cc_par, const double cc_gra, 
                    const size_t npar_bsnp, const size_t curve_num, 
                    const gsl_matrix *Pheno, const gsl_matrix *Qmat, 
                    const gsl_matrix *Uk, const gsl_vector *eval, const gsl_vector *Time, 
            const size_t h0, gsl_vector *par_vg, gsl_vector *par_vp, gsl_vector *par_vr, gsl_matrix *X, 
            gsl_vector *beta, gsl_vector *ai, gsl_vector *pi, double &log_RL, double &pvalue, gsl_vector *logRL_logL_2parts);

#endif
