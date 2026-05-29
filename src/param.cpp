/*
    Dynamic Genomic Best linear Unbiased Prediction  (DynGBLUP)
    Copyright © 2026, Meixia Ye
    
*/

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sys/stat.h>

#include "gsl/gsl_blas.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_vector.h"

#include "eigenlib.h"
#include "io.h"
#include "mathfunc.h"
#include "param.h"

using namespace std;



void trim_individuals(vector<int> &idvs, size_t ni_max) {
  if (ni_max) {
    size_t count = 0;
    for (auto ind = idvs.begin(); ind != idvs.end(); ++ind) {
      if (*ind)
        count++;
      if (count >= ni_max)
        break;
    }
    if (count != idvs.size()) {
      if (is_debug_mode())
        cout << "**** TEST MODE: trim individuals from " << idvs.size()
             << " to " << count << endl;
      idvs.resize(count);
    }
  }
}



PARAM::PARAM(void)
    : a_mode(0), k_mode(1), d_pace(DEFAULT_PACE),
      file_out("result"), path_out("./output/"), 
      miss_level(0.95), maf_level(0.01), hwe_level(0), r2_level(0.9999), 
      nrf(3), nr1_size(3), nr2_size(3), U(nullptr), eval(nullptr), Y(nullptr), W(nullptr), S(nullptr), Times(nullptr), G(nullptr),
      max_iter(20), cc_par(0.00001), cc_gra(0.0005), 
      time_total(0.0), time_G(0.0), time_eigen(0.0),  
      time_opt(0.0) {}


void PARAM::ReadFiles(void) {
  string file_str;
  
  
  if (!file_cvt.empty()) {
    if (ReadFile_cvt(file_cvt, indicator_cvt, cvt, n_cvt) == false) {
      error = true;
    }
    if ((indicator_cvt).size() == 0) {
      n_cvt = 1;
    }
  } else {
    n_cvt = 1;
  }
  trim_individuals(indicator_cvt, ni_max);
  printf("  read_cvt...finished \n");

  
  if (!file_bfile.empty()) {
    file_str = file_bfile + ".bim";
    snpInfo.clear();
    if (ReadFile_bim(file_str, snpInfo) == false) {
      error = true;
    }
    printf("  read_bim...finished \n");

    file_str = file_bfile + ".fam";
    if (ReadFile_fam(file_str, indicator_pheno, pheno, mapID2num, p_column) == false) {
        error = true;
    }
    printf("  read_fam...finished \n");

    ProcessCvtPhen();

    auto W1 = gsl_matrix_safe_alloc(ni_test, n_cvt);
    CopyCvt(W1);
    

    file_str = file_bfile + ".bed";
    if (ReadFile_bed(file_str, setSnps, W1, indicator_idv, indicator_snp,
                     snpInfo, maf_level, miss_level, hwe_level, r2_level,
                     ns_test) == false) {
      error = true;
    }
    printf("  read_bed...finished \n");

    gsl_matrix_free(W1);
    ns_total = indicator_snp.size();
  }

  
 if (!file_geno.empty()) {

    ProcessCvtPhen();

    auto W2 = gsl_matrix_safe_alloc(ni_test, n_cvt);
    CopyCvt(W2);

    trim_individuals(indicator_idv, ni_max);
    trim_individuals(indicator_cvt, ni_max);
   
    gsl_matrix_free(W2);

    ns_total = indicator_snp.size();
  }
  printf("step1, finished \n");

  return;
}

void PARAM::CheckParam(void) {
  struct stat fileInfo;
  string str;

  
  if (k_mode != 1 && k_mode != 2) {
    cout << "error! unknown kinship/relatedness input mode: " << k_mode << endl;
    error = true;
  }
  if (a_mode != 1 && a_mode != 2 && a_mode != 3 && a_mode != 4 && a_mode != 5 &&
      a_mode != 11 && a_mode != 12 && a_mode != 13 && a_mode != 14 &&
      a_mode != 15 && a_mode != 21 && a_mode != 22 && a_mode != 25 &&
      a_mode != 26 && a_mode != 27 && a_mode != 28 && a_mode != 31 &&
      a_mode != 41 && a_mode != 42 && a_mode != 43 && a_mode != 51 &&
      a_mode != 52 && a_mode != 53 && a_mode != 54 && a_mode != 61 &&
      a_mode != 62 && a_mode != 63 && a_mode != 66 && a_mode != 67 &&
      a_mode != 71) {
    cout << "error! unknown analysis mode: " << a_mode
         << ". make sure -gk or -eigen or -lmm is sepcified correctly." << endl;
    error = true;
  }
  if (miss_level > 1) {
    cout << "error! missing level needs to be between 0 and 1. "
         << "current value = " << miss_level << endl;
    error = true;
  }
  if (maf_level > 0.5) {
    cout << "error! maf level needs to be between 0 and 0.5. "
         << "current value = " << maf_level << endl;
    error = true;
  }
  if (hwe_level > 1) {
    cout << "error! hwe level needs to be between 0 and 1. "
         << "current value = " << hwe_level << endl;
    error = true;
  }
  if (r2_level > 1) {
    cout << "error! r2 level needs to be between 0 and 1. "
         << "current value = " << r2_level << endl;
    error = true;
  }

  

  if ( (p_column.size() == 0) ) {
    
    p_column.push_back(1);
  } else {
    for (size_t i = 0; i < p_column.size(); i++) {
      for (size_t j = 0; j < i; j++) {
        if (p_column[i] == p_column[j]) {
          cout << "error! identical phenotype "
               << "columns: " << p_column[i] << endl;
          error = true;
        }
      }
    }
  }

  n_ph = p_column.size();

  if (a_mode <= 4) {
    if(n_ph > 1){
      if(curve_type == 0){
         cout << "error! the current analysis mode " << a_mode << endl;
         error = true;
      }
    }else{
      if(curve_type == 0){
         cout << "error! the current analysis mode " << a_mode << endl;
         cout << "use -n [num1 num2 ...] to specify multiple phenotypes !" << endl;
         error = true;
      }
    }

    
  }

 
  if (n_ph > 1 && a_mode != 1 && a_mode != 2 && a_mode != 3 && a_mode != 4 &&
      a_mode != 43) {
    cout << "error! the current analysis mode " << a_mode
         << " can not deal with multiple phenotypes." << endl;
    error = true;
  }


  
  if (!file_bfile.empty()) {
    str = file_bfile + ".bim";
    if (stat(str.c_str(), &fileInfo) == -1) {
      cout << "error! fail to open .bim file: " << str << endl;
      error = true;
    }
    str = file_bfile + ".bed";
    if (stat(str.c_str(), &fileInfo) == -1) {
      cout << "error! fail to open .bed file: " << str << endl;
      error = true;
    }
    str = file_bfile + ".fam";
    if (stat(str.c_str(), &fileInfo) == -1) {
      cout << "error! fail to open .fam file: " << str << endl;
      error = true;
    }
  }

  if ( !file_geno.empty() ) {
    str = file_pheno;
    if (stat(str.c_str(), &fileInfo) == -1) {
      cout << "error! fail to open phenotype file: " << str << endl;
      error = true;
    }
  }

  str = file_geno;
  if (!str.empty() && stat(str.c_str(), &fileInfo) == -1) {
    cout << "error! fail to open mean genotype file: " << str << endl;
    error = true;
  }


  size_t flag = 0;
  if (!file_bfile.empty()) {
    flag++;
  }
  if (!file_geno.empty()) {
    flag++;
  }

  if (flag != 1 && a_mode != 15 && a_mode != 27 && a_mode != 28 &&
      a_mode != 43 && a_mode != 5 && a_mode != 61 && a_mode != 62 &&
      a_mode != 63 && a_mode != 66 && a_mode != 67) {
    cout << "error! either plink binary files, or bimbam mean"
         << "genotype files, or gene expression files are required." << endl;
    error = true;
  }

  if (file_pheno.empty() && (a_mode == 43 || a_mode == 5)) {
    cout << "error! phenotype file is required." << endl;
    error = true;
  }


  if (a_mode == 63) {
    if (file_kin.empty() && (file_ku.empty() || file_kd.empty()) ) {
      cout << "error! missing relatedness file. " << endl;
      error = true;
    }
    if (file_pheno.empty()) {
      cout << "error! missing phenotype file." << endl;
      error = true;
    }
  }


  enforce_fexists(file_anno, "open file");
  enforce_fexists(file_kin, "open file");
  enforce_fexists(file_cvt, "open file");

 
  if (k_mode == 2 && !file_geno.empty()) {
    cout << "error! use \"-km 1\" when using bimbam mean genotype "
         << "file. " << endl;
    error = true;
  }

  if ((a_mode == 1 || a_mode == 2 || a_mode == 3 || a_mode == 4 ||
       a_mode == 5 || a_mode == 31) &&
      (file_kin.empty() && (file_ku.empty() || file_kd.empty()))) {
    cout << "error! missing relatedness file. " << endl;
    error = true;
  }

  if ((a_mode == 43) && file_kin.empty()) {
    cout << "error! missing relatedness file. -predict option requires "
         << "-k option to provide a relatedness file." << endl;
    error = true;
  }
  

  return;
}

void PARAM::CheckData(void) {

  if ((indicator_cvt).size() != 0 &&
      (indicator_cvt).size() != (indicator_idv).size()) {
    error = true;
    cout << "error! number of rows in the covariates file do not "
         << "match the number of individuals. " << indicator_cvt.size() << endl;
    return;
  }

  
  ni_total = (indicator_idv).size();

  ni_test = 0;
  for (vector<int>::size_type i = 0; i < (indicator_idv).size(); ++i) {
    if (indicator_idv[i] == 0) {
      continue;
    }
    ni_test++;
  }

  ni_cvt = 0;
  for (size_t i = 0; i < indicator_cvt.size(); i++) {
    if (indicator_cvt[i] == 0) {
      continue;
    }
    ni_cvt++;
  }

  np_obs = 0;
  np_miss = 0;
  for (size_t i = 0; i < indicator_pheno.size(); i++) {
    if (indicator_cvt.size() != 0) {
      if (indicator_cvt[i] == 0) {
        continue;
      }
    }

    for (size_t j = 0; j < indicator_pheno[i].size(); j++) {
      if (indicator_pheno[i][j] == 0) {
        np_miss++;
      } else {
        np_obs++;
      }
    }
  }

  if (ni_test == 0) {
    error = true;
    cout << "error! number of analyzed individuals equals 0. " << endl;
    return;
  }

  if (a_mode == 43) {
    if (ni_cvt == ni_test) {
      error = true;
      cout << "error! no individual has missing "
           << "phenotypes." << endl;
      return;
    }
    if ((np_obs + np_miss) != (ni_cvt * n_ph)) {
      error = true;
      cout << "error! number of phenotypes do not match the "
           << "summation of missing and observed phenotypes." << endl;
      return;
    }
  }


  if (a_mode != 15 && a_mode != 27 && a_mode != 28) {
    cout << "## number of total individuals = " << ni_total << endl;
    if (a_mode == 43) {
      cout << "## number of analyzed individuals = " << ni_cvt << endl;
      cout << "## number of individuals with full phenotypes = " << ni_test
           << endl;
    } else {
      cout << "## number of analyzed individuals = " << ni_test << endl;
    }
    cout << "## number of covariates = " << n_cvt << endl;
    cout << "## number of phenotypes = " << n_ph << endl;
    if (a_mode == 43) {
      cout << "## number of observed data = " << np_obs << endl;
      cout << "## number of missing data = " << np_miss << endl;
    }
  }
 

  return;
}

void PARAM::PrintSummary() {
  if (n_ph == 1) {
    
  } else {
  }
  return;
}


void PARAM::ReadGenotypes(gsl_matrix *UtX, gsl_matrix *K, const bool calc_K) {
  string file_str;

  if (!file_bfile.empty()) {
    file_str = file_bfile + ".bed";
    if (ReadFile_bed(file_str, indicator_idv, indicator_snp, UtX, K, calc_K) ==
        false) {
      error = true;
    }
  } else {
    
      error = true;
    }
  

  return;
}

void PARAM::ReadGenotypes(vector<vector<unsigned char>> &Xt, gsl_matrix *K,
                          const bool calc_K) 
{
  string file_str;

  if (!file_bfile.empty()) {
    file_str = file_bfile + ".bed";
    if (ReadFile_bed(file_str, indicator_idv, indicator_snp, Xt, K, calc_K,
                     ni_test, ns_test) == false) {
      error = true;
    }
  } else {
   
      error = true;
    }
  

  return;
}



void PARAM::CalcKinS(gsl_matrix *matrix_kin, gsl_matrix *matrix_S) {
    string file_str;

    gsl_matrix_set_zero(matrix_kin);
    gsl_matrix_set_zero(matrix_S);

   
    if (file_bfile.empty()) {
        error = true;          
        return;
    }

    file_str = file_bfile + ".bed";
    if (PlinkKinS(file_str, indicator_snp, a_mode - 20, d_pace, matrix_kin, matrix_S) == false) {
        error = true;
    }
}




void compAKtoS(const gsl_matrix *A, const gsl_matrix *K, const size_t n_cvt,
               gsl_matrix *S) 
{
  size_t n_vc = S->size1, ni_test = A->size1;
  double di, dj, tr_AK, sum_A, sum_K, s_A, s_K, sum_AK, tr_A, tr_K, d;

  for (size_t i = 0; i < n_vc; i++) {
    for (size_t j = 0; j < n_vc; j++) {
      tr_AK = 0;
      sum_A = 0;
      sum_K = 0;
      sum_AK = 0;
      tr_A = 0;
      tr_K = 0;
      for (size_t l = 0; l < ni_test; l++) {
        s_A = 0;
        s_K = 0;
        for (size_t k = 0; k < ni_test; k++) {
          di = gsl_matrix_get(A, l, k + ni_test * i);
          dj = gsl_matrix_get(K, l, k + ni_test * j);
          s_A += di;
          s_K += dj;

          tr_AK += di * dj;
          sum_A += di;
          sum_K += dj;
          if (l == k) {
            tr_A += di;
            tr_K += dj;
          }
        }
        sum_AK += s_A * s_K;
      }

      sum_A /= (double)ni_test;
      sum_K /= (double)ni_test;
      sum_AK /= (double)ni_test;
      tr_A -= sum_A;
      tr_K -= sum_K;
      d = tr_AK - 2 * sum_AK + sum_A * sum_K;

      if (tr_A == 0 || tr_K == 0) {
        d = 0;
      } else {
        d = d / (tr_A * tr_K) - 1 / (double)(ni_test - n_cvt);
      }

      gsl_matrix_set(S, i, j, d);
    }
  }

  return;
}


size_t GetabIndex(const size_t a, const size_t b, const size_t n_cvt) 
{
  if (a > n_cvt + 2 || b > n_cvt + 2 || a <= 0 || b <= 0) {
    cout << "error in GetabIndex." << endl;
    return 0;
  }
  size_t index;
  size_t l, h;
  if (b > a) {
    l = a;
    h = b;
  } else {
    l = b;
    h = a;
  }

  size_t n = n_cvt + 2;
  index = (2 * n - l + 2) * (l - 1) / 2 + h - l;

  return index;
}


void compKtoV(const gsl_matrix *G, gsl_matrix *V) {
  size_t n_vc = G->size2 / G->size1, ni_test = G->size1;

  gsl_matrix *KiKj =
      gsl_matrix_alloc(ni_test, (n_vc * (n_vc + 1)) / 2 * ni_test);
  gsl_vector *trKiKj = gsl_vector_alloc(n_vc * (n_vc + 1) / 2);
  gsl_vector *trKi = gsl_vector_alloc(n_vc);

  double d, tr, r = (double)ni_test / (double)(ni_test - 1);
  size_t t, t_il, t_jm, t_lm, t_im, t_jl, t_ij;

  
  t = 0;
  for (size_t i = 0; i < n_vc; i++) {
    gsl_matrix_const_view Ki =
        gsl_matrix_const_submatrix(G, 0, i * ni_test, ni_test, ni_test);
    for (size_t j = i; j < n_vc; j++) {
      gsl_matrix_const_view Kj =
          gsl_matrix_const_submatrix(G, 0, j * ni_test, ni_test, ni_test);
      gsl_matrix_view KiKj_sub =
          gsl_matrix_submatrix(KiKj, 0, t * ni_test, ni_test, ni_test);
      eigenlib_dgemm("N", "N", 1.0, &Ki.matrix, &Kj.matrix, 0.0,
                     &KiKj_sub.matrix);
      t++;
    }
  }

  
  t = 0;
  for (size_t i = 0; i < n_vc; i++) {
    for (size_t j = i; j < n_vc; j++) {
      tr = 0;
      for (size_t k = 0; k < ni_test; k++) {
        tr += gsl_matrix_get(KiKj, k, t * ni_test + k);
      }
      gsl_vector_set(trKiKj, t, tr);

      t++;
    }

    tr = 0;
    for (size_t k = 0; k < ni_test; k++) {
      tr += gsl_matrix_get(G, k, i * ni_test + k);
    }
    gsl_vector_set(trKi, i, tr);
  }

  
  for (size_t i = 0; i < n_vc; i++) {
    for (size_t j = i; j < n_vc; j++) {
      t_ij = GetabIndex(i + 1, j + 1, n_vc - 2);
      for (size_t l = 0; l < n_vc + 1; l++) {
        for (size_t m = 0; m < n_vc + 1; m++) {
          if (l != n_vc && m != n_vc) {
            t_il = GetabIndex(i + 1, l + 1, n_vc - 2);
            t_jm = GetabIndex(j + 1, m + 1, n_vc - 2);
            t_lm = GetabIndex(l + 1, m + 1, n_vc - 2);
            tr = 0;
            for (size_t k = 0; k < ni_test; k++) {
              gsl_vector_const_view KiKl_row =
                  gsl_matrix_const_subrow(KiKj, k, t_il * ni_test, ni_test);
              gsl_vector_const_view KiKl_col =
                  gsl_matrix_const_column(KiKj, t_il * ni_test + k);
              gsl_vector_const_view KjKm_row =
                  gsl_matrix_const_subrow(KiKj, k, t_jm * ni_test, ni_test);
              gsl_vector_const_view KjKm_col =
                  gsl_matrix_const_column(KiKj, t_jm * ni_test + k);

              gsl_vector_const_view Kl_row =
                  gsl_matrix_const_subrow(G, k, l * ni_test, ni_test);
              gsl_vector_const_view Km_row =
                  gsl_matrix_const_subrow(G, k, m * ni_test, ni_test);

              if (i <= l && j <= m) {
                gsl_blas_ddot(&KiKl_row.vector, &KjKm_col.vector, &d);
                tr += d;
                gsl_blas_ddot(&Km_row.vector, &KiKl_col.vector, &d);
                tr -= r * d;
                gsl_blas_ddot(&Kl_row.vector, &KjKm_col.vector, &d);
                tr -= r * d;
              } else if (i <= l && j > m) {
                gsl_blas_ddot(&KiKl_row.vector, &KjKm_row.vector, &d);
                tr += d;
                gsl_blas_ddot(&Km_row.vector, &KiKl_col.vector, &d);
                tr -= r * d;
                gsl_blas_ddot(&Kl_row.vector, &KjKm_row.vector, &d);
                tr -= r * d;
              } else if (i > l && j <= m) {
                gsl_blas_ddot(&KiKl_col.vector, &KjKm_col.vector, &d);
                tr += d;
                gsl_blas_ddot(&Km_row.vector, &KiKl_row.vector, &d);
                tr -= r * d;
                gsl_blas_ddot(&Kl_row.vector, &KjKm_col.vector, &d);
                tr -= r * d;
              } else {
                gsl_blas_ddot(&KiKl_col.vector, &KjKm_row.vector, &d);
                tr += d;
                gsl_blas_ddot(&Km_row.vector, &KiKl_row.vector, &d);
                tr -= r * d;
                gsl_blas_ddot(&Kl_row.vector, &KjKm_row.vector, &d);
                tr -= r * d;
              }
            }

            tr += r * r * gsl_vector_get(trKiKj, t_lm);
          } else if (l != n_vc && m == n_vc) {
            t_il = GetabIndex(i + 1, l + 1, n_vc - 2);
            t_jl = GetabIndex(j + 1, l + 1, n_vc - 2);
            tr = 0;
            for (size_t k = 0; k < ni_test; k++) {
              gsl_vector_const_view KiKl_row =
                  gsl_matrix_const_subrow(KiKj, k, t_il * ni_test, ni_test);
              gsl_vector_const_view KiKl_col =
                  gsl_matrix_const_column(KiKj, t_il * ni_test + k);
              gsl_vector_const_view Kj_row =
                  gsl_matrix_const_subrow(G, k, j * ni_test, ni_test);

              if (i <= l) {
                gsl_blas_ddot(&KiKl_row.vector, &Kj_row.vector, &d);
                tr += d;
              } else {
                gsl_blas_ddot(&KiKl_col.vector, &Kj_row.vector, &d);
                tr += d;
              }
            }
            tr += -r * gsl_vector_get(trKiKj, t_il) -
                  r * gsl_vector_get(trKiKj, t_jl) +
                  r * r * gsl_vector_get(trKi, l);
          } else if (l == n_vc && m != n_vc) {
            t_jm = GetabIndex(j + 1, m + 1, n_vc - 2);
            t_im = GetabIndex(i + 1, m + 1, n_vc - 2);
            tr = 0;
            for (size_t k = 0; k < ni_test; k++) {
              gsl_vector_const_view KjKm_row =
                  gsl_matrix_const_subrow(KiKj, k, t_jm * ni_test, ni_test);
              gsl_vector_const_view KjKm_col =
                  gsl_matrix_const_column(KiKj, t_jm * ni_test + k);
              gsl_vector_const_view Ki_row =
                  gsl_matrix_const_subrow(G, k, i * ni_test, ni_test);

              if (j <= m) {
                gsl_blas_ddot(&KjKm_row.vector, &Ki_row.vector, &d);
                tr += d;
              } else {
                gsl_blas_ddot(&KjKm_col.vector, &Ki_row.vector, &d);
                tr += d;
              }
            }
            tr += -r * gsl_vector_get(trKiKj, t_im) -
                  r * gsl_vector_get(trKiKj, t_jm) +
                  r * r * gsl_vector_get(trKi, m);
          } else {
            tr = gsl_vector_get(trKiKj, t_ij) - r * gsl_vector_get(trKi, i) -
                 r * gsl_vector_get(trKi, j) + r * r * (double)(ni_test - 1);
          }

          gsl_matrix_set(V, l, t_ij * (n_vc + 1) + m, tr);
        }
      }
    }
  }

  gsl_matrix_scale(V, 1.0 / pow((double)ni_test, 2));

  gsl_matrix_free(KiKj);
  gsl_vector_free(trKiKj);
  gsl_vector_free(trKi);

  return;
}


void JackknifeAKtoS(const gsl_matrix *W, const gsl_matrix *A,
                    const gsl_matrix *K, gsl_matrix *S, gsl_matrix *Svar) 
{
  size_t n_vc = Svar->size1, ni_test = A->size1, n_cvt = W->size2;

  vector<vector<vector<double>>> trAK, sumAK;
  vector<vector<double>> sumA, sumK, trA, trK, sA, sK;
  vector<double> vec_tmp;
  double di, dj, d, m, v;

  
  for (size_t i = 0; i < ni_test; i++) {
    vec_tmp.push_back(0);
  }

  for (size_t i = 0; i < n_vc; i++) {
    sumA.push_back(vec_tmp);
    sumK.push_back(vec_tmp);
    trA.push_back(vec_tmp);
    trK.push_back(vec_tmp);
    sA.push_back(vec_tmp);
    sK.push_back(vec_tmp);
  }

  for (size_t i = 0; i < n_vc; i++) {
    trAK.push_back(sumK);
    sumAK.push_back(sumK);
  }

  
  for (size_t i = 0; i < n_vc; i++) {
    for (size_t l = 0; l < ni_test; l++) {
      for (size_t k = 0; k < ni_test; k++) {
        di = gsl_matrix_get(A, l, k + ni_test * i);
        dj = gsl_matrix_get(K, l, k + ni_test * i);

        for (size_t t = 0; t < ni_test; t++) {
          if (t == l || t == k) {
            continue;
          }
          sumA[i][t] += di;
          sumK[i][t] += dj;
          if (l == k) {
            trA[i][t] += di;
            trK[i][t] += dj;
          }
        }
        sA[i][l] += di;
        sK[i][l] += dj;
      }
    }

    for (size_t t = 0; t < ni_test; t++) {
      sumA[i][t] /= (double)(ni_test - 1);
      sumK[i][t] /= (double)(ni_test - 1);
    }
  }

  for (size_t i = 0; i < n_vc; i++) {
    for (size_t j = 0; j < n_vc; j++) {
      for (size_t l = 0; l < ni_test; l++) {
        for (size_t k = 0; k < ni_test; k++) {
          di = gsl_matrix_get(A, l, k + ni_test * i);
          dj = gsl_matrix_get(K, l, k + ni_test * j);
          d = di * dj;

          for (size_t t = 0; t < ni_test; t++) {
            if (t == l || t == k) {
              continue;
            }
            trAK[i][j][t] += d;
          }
        }

        for (size_t t = 0; t < ni_test; t++) {
          if (t == l) {
            continue;
          }
          di = gsl_matrix_get(A, l, t + ni_test * i);
          dj = gsl_matrix_get(K, l, t + ni_test * j);

          sumAK[i][j][t] += (sA[i][l] - di) * (sK[j][l] - dj);
        }
      }

      for (size_t t = 0; t < ni_test; t++) {
        sumAK[i][j][t] /= (double)(ni_test - 1);
      }

      m = 0;
      v = 0;
      for (size_t t = 0; t < ni_test; t++) {
        d = trAK[i][j][t] - 2 * sumAK[i][j][t] + sumA[i][t] * sumK[j][t];
        if ((trA[i][t] - sumA[i][t]) == 0 || (trK[j][t] - sumK[j][t]) == 0) {
          d = 0;
        } else {
          d /= (trA[i][t] - sumA[i][t]) * (trK[j][t] - sumK[j][t]);
          d -= 1 / (double)(ni_test - n_cvt - 1);
        }
        m += d;
        v += d * d;
      }
      m /= (double)ni_test;
      v /= (double)ni_test;
      v -= m * m;
      v *= (double)(ni_test - 1);
      gsl_matrix_set(Svar, i, j, v);
      if (n_cvt == 1) {
        d = gsl_matrix_get(S, i, j);
        d = (double)ni_test * d - (double)(ni_test - 1) * m;
        gsl_matrix_set(S, i, j, d);
      }
    }
  }

  return;
}






void PARAM::WriteVector(const gsl_vector *q, const gsl_vector *s,
                        const size_t n_total, const string suffix) 
{
  string file_str;
  file_str = path_out + "/" + file_out;
  file_str += ".";
  file_str += suffix;
  file_str += ".txt";

  ofstream outfile(file_str.c_str(), ofstream::out);
  if (!outfile) {
    cout << "error writing file: " << file_str.c_str() << endl;
    return;
  }

  outfile.precision(10);

  for (size_t i = 0; i < q->size; ++i) {
    outfile << gsl_vector_get(q, i) << endl;
  }

  for (size_t i = 0; i < s->size; ++i) {
    outfile << gsl_vector_get(s, i) << endl;
  }

  outfile << n_total << endl;

  outfile.close();
  outfile.clear();
  return;
}

void PARAM::WriteMatrix(const gsl_matrix *matrix_U, const string suffix) {
  string file_str;
  file_str = path_out + "/" + file_out;
  file_str += ".";
  file_str += suffix;
  file_str += ".txt";

  ofstream outfile(file_str.c_str(), ofstream::out);
  if (!outfile) {
    cout << "error writing file: " << file_str.c_str() << endl;
    return;
  }

  outfile.precision(10);

  for (size_t i = 0; i < matrix_U->size1; ++i) {
    for (size_t j = 0; j < matrix_U->size2; ++j) {
      outfile << tab(j) << gsl_matrix_get(matrix_U, i, j);
    }
    outfile << endl;
  }

  outfile.close();
  outfile.clear();
  return;
}

void PARAM::WriteVector(const gsl_vector *vector_D, const string suffix) {
  string file_str;
  file_str = path_out + "/" + file_out;
  file_str += ".";
  file_str += suffix;
  file_str += ".txt";

  ofstream outfile(file_str.c_str(), ofstream::out);
  if (!outfile) {
    cout << "error writing file: " << file_str.c_str() << endl;
    return;
  }

  outfile.precision(10);

  for (size_t i = 0; i < vector_D->size; ++i) {
    outfile << gsl_vector_get(vector_D, i) << endl;
  }

  outfile.close();
  outfile.clear();
  return;
}





void PARAM::CheckCvt() {
  if (indicator_cvt.size() == 0) {
    return;
  }

  size_t ci_test = 0;

  gsl_matrix *W = gsl_matrix_alloc(ni_test, n_cvt);

  for (vector<int>::size_type i = 0; i < indicator_idv.size(); ++i) {
    if (indicator_idv[i] == 0 || indicator_cvt[i] == 0) {
      continue;
    }
    for (size_t j = 0; j < n_cvt; ++j) {
      gsl_matrix_set(W, ci_test, j, (cvt)[i][j]);
    }
    ci_test++;
  }

  size_t flag_ipt = 0;
  double v_min, v_max;
  set<size_t> set_remove;

 
  for (size_t i = 0; i < W->size2; i++) {
    gsl_vector_view w_col = gsl_matrix_column(W, i);
    gsl_vector_minmax(&w_col.vector, &v_min, &v_max);
    if (v_min == v_max) {
      flag_ipt = 1;
      set_remove.insert(i);
    }
  }

  
  if (n_cvt == set_remove.size()) {
    indicator_cvt.clear();
    n_cvt = 1;
  } else if (flag_ipt == 0) {
    cout << "no intercept term is found in the cvt file. " << endl;
    
    for (vector<int>::size_type i = 0; i < indicator_idv.size(); ++i) {
      if (indicator_idv[i] == 0 || indicator_cvt[i] == 0) {
        continue;
      }
    
    }

  
  } else {
  }

  gsl_matrix_free(W);

  return;
}


void PARAM::ProcessCvtPhen() {

  
  int k = 1;
  indicator_idv.clear();
  for (size_t i = 0; i < indicator_pheno.size(); i++) {
    k = 1;
    for (size_t j = 0; j < indicator_pheno[i].size(); j++) {
      if (indicator_pheno[i][j] == 0) {
        k = 0;
      }
    }
    indicator_idv.push_back(k);
  }

  
  if ((indicator_cvt).size() != 0) {
    for (vector<int>::size_type i = 0; i < (indicator_idv).size(); ++i) {
      indicator_idv[i] *= indicator_cvt[i];
    }
  }

  
  ni_test = 0;
  for (vector<int>::size_type i = 0; i < (indicator_idv).size(); ++i) {
    if (indicator_idv[i] == 0) {
      continue;
    }
    ni_test++;
  }


  
  if (ni_test == 0 && a_mode != 15) {
    error = true;
    cout << "error! number of analyzed individuals equals 0. " << endl;
    return;
  }

  
  if (indicator_cvt.size() != 0) {
    CheckCvt();
  } else {
    vector<double> cvt_row;
    cvt_row.push_back(1);

    for (vector<int>::size_type i = 0; i < (indicator_idv).size(); ++i) {
      indicator_cvt.push_back(1);
      cvt.push_back(cvt_row);
    }
  }

  return;
}

void PARAM::CopyCvt(gsl_matrix *W) {
  size_t ci_test = 0;

  for (vector<int>::size_type i = 0; i < indicator_idv.size(); ++i) {
    if (indicator_idv[i] == 0 || indicator_cvt[i] == 0) {
      continue;
    }
    for (size_t j = 0; j < n_cvt; ++j) {
      gsl_matrix_set(W, ci_test, j, (cvt)[i][j]);
    }
    ci_test++;
  }

  return;
}





void PARAM::CopyCvtPhen(gsl_matrix *W, gsl_vector *y, size_t flag) {
  size_t ci_test = 0;

  for (vector<int>::size_type i = 0; i < indicator_idv.size(); ++i) {
    if (flag == 0) {
      if (indicator_idv[i] == 0) {
        continue;
      }
    } else {
      if (indicator_cvt[i] == 0) {
        continue;
      }
    }

    gsl_vector_set(y, ci_test, (pheno)[i][0]);

    for (size_t j = 0; j < n_cvt; ++j) {
      gsl_matrix_set(W, ci_test, j, (cvt)[i][j]);
    }
    ci_test++;
  }

  return;
}


void PARAM::CopyCvtPhen(gsl_matrix *W, gsl_matrix *Y, size_t flag) {
  size_t ci_test = 0;

  for (vector<int>::size_type i = 0; i < indicator_idv.size(); ++i) {
    if (flag == 0) {
      if (indicator_idv[i] == 0) {
        continue;
      }
    } else {
      if (indicator_cvt[i] == 0) {
        continue;
      }
    }

    for (size_t j = 0; j < n_ph; ++j) {
      gsl_matrix_set(Y, ci_test, j, (pheno)[i][j]);
    }
    for (size_t j = 0; j < n_cvt; ++j) {
      gsl_matrix_set(W, ci_test, j, (cvt)[i][j]);
    }

    ci_test++;
  }

  return;
}





