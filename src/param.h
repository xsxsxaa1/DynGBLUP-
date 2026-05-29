/*
    Dynamic Genomic Best linear Unbiased Prediction  (DynGBLUP)
    Copyright © 2026, Meixia Ye
    
*/

#ifndef __PARAM_H__
#define __PARAM_H__

#include "debug.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
#include <map>
#include <set>
#include <vector>

#define K_BATCH_SIZE 24949
#define DEFAULT_PACE 1000  

using namespace std;

class SNPINFO {
public:
  string chr;
  string rs_number;
  double cM;
  long int base_position;
  string a_minor;
  string a_major;
  size_t n_miss;
  double missingness;
  double maf;
  size_t n_idv;         
  size_t n_nb;          
  size_t file_position; 
};


class MPHSUMSTAT {
public:
  

  double p_wald;          
  double p_lr2;           
  double log_RL;               
  double log_L;                
  vector<double> v_Vg;    
  vector<double> v_Vp;    
  vector<double> v_Ve;    
  vector<double> v_beta;  
};

class MPHSUMSTAT_yxx {
public:
  
  
  double v_chisq;          
  double v_pval;           
  vector<double> v_ui_i;  
};



class HEADER {
public:
  size_t rs_col;
  size_t chr_col;
  size_t pos_col;
  size_t cm_col;
  size_t a1_col;
  size_t a0_col;
  size_t z_col;
  size_t beta_col;
  size_t chisq_col;
  size_t p_col;
  size_t n_col;
  size_t nmis_col;
  size_t nobs_col;
  size_t af_col;
  size_t coln; 
};

class PARAM {
public:

  uint a_mode; 
  int k_mode; 
  vector<size_t> p_column; 
  size_t d_pace = DEFAULT_PACE;   
  size_t curve_type;


  string file_bfile;
  string file_geno;
  string file_pheno;
  string file_anno; 
  string file_kin;
  string file_S;
  string file_ku, file_kd;
  string file_cvt;  
  string file_out;
  string path_out;

  
  double miss_level;
  double maf_level;
  double hwe_level;
  double r2_level;

  
  double logl_mle_H0, logl_remle_H0;
  vector<double> Vg_remle_null, Vp_remle_null, Ve_remle_null;
  vector<double> beta_remle_null, ai_remle_null, pi_remle_null;

  size_t max_iter;
  double cc_par, cc_gra;
 
  size_t nrf, nr1_size, nr2_size;
  double trace_G;
  bool error;

  
  size_t ni_total, ni_test, ni_cvt;
  size_t ni_max = 0; 

  
  size_t np_obs, np_miss;
  size_t ns_total, ns_test;
  
  size_t n_cvt;               
  size_t n_ph;                

  double time_total;          
  double time_G;              
                              
  double time_eigen;          
  double time_opt;

 
  gsl_matrix *U;          
  gsl_vector *eval;       
  gsl_matrix *Y;          
  gsl_matrix *W;          
  gsl_matrix *S;          
  gsl_vector *Times;      
  gsl_matrix *G;          

  vector<vector<double>> pheno;
  vector<vector<double>> cvt;
  vector<vector<int>> indicator_pheno;

  
  vector<int> indicator_idv;
  vector<int> indicator_snp;
  vector<int> indicator_cvt;

  map<string, int> mapID2num;             
  map<string, string> mapRS2chr;          
  map<string, long int> mapRS2bp;         
  map<string, double> mapRS2cM;           
  map<string, double> mapRS2est;          
   
  vector<SNPINFO> snpInfo;          
  set<string> setSnps;              
  set<string> setKSnps;             
  set<string> setGWASnps;           

  PARAM();

  void ReadFiles();
  void CheckParam();
  void CheckData();
  void PrintSummary();
  void ReadGenotypes(gsl_matrix *UtX, gsl_matrix *K, const bool calc_K);
  void ReadGenotypes(vector<vector<unsigned char>> &Xt, gsl_matrix *K,
                     const bool calc_K);
  void CheckCvt();
  void CopyCvt(gsl_matrix *W);
  void CopyA(size_t flag, gsl_matrix *A);
  
  void ProcessCvtPhen();
  void CopyCvtPhen(gsl_matrix *W, gsl_vector *y, size_t flag);
  void CopyCvtPhen(gsl_matrix *W, gsl_matrix *Y, size_t flag);
  void CalcKinS(gsl_matrix *matrix_kin, gsl_matrix *matrix_S);
  
  void WriteVector(const gsl_vector *q, const gsl_vector *s,
                   const size_t n_total, const string suffix);
  void WriteMatrix(const gsl_matrix *matrix_U, const string suffix);
  void WriteVector(const gsl_vector *vector_D, const string suffix);
 
  void UpdateSNPnZ(const map<string, double> &mapRS2wA,
                   const map<string, string> &mapRS2A1,
                   const map<string, double> &mapRS2z, gsl_vector *w,
                   gsl_vector *z, vector<size_t> &vec_cat);
  void UpdateSNP(const map<string, double> &mapRS2wA);
};

size_t GetabIndex(const size_t a, const size_t b, const size_t n_cvt);

#endif
