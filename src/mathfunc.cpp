/*
    Dynamic Genomic Best linear Unbiased Prediction  (DynGBLUP)
    Copyright © 2026, Meixia Ye
    
*/

#include <bitset>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <map>
#include <set>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <tuple>
#include <vector>


#include "gsl/gsl_version.h"

#if GSL_MAJOR_VERSION < 2
#pragma message "GSL version " GSL_VERSION
#endif

#include "gsl/gsl_sys.h" 
#include "gsl/gsl_blas.h"
#include "gsl/gsl_cdf.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_eigen.h"

#include "debug.h"
#include "fastblas.h"
#include "lapack.h"
#include "mathfunc.h"

using namespace std;

bool has_nan(const vector<double> v) {
  for (const auto& e: v) {
    if (is_nan(e))
      return true;
  }
  return false;
}

bool has_nan(const gsl_vector *v) {
  for (size_t i = 0; i < v->size; ++i)
    if (gsl_isnan(gsl_vector_get(v,i))) return true;
  return false;
}
bool has_inf(const gsl_vector *v) {
  for (size_t i = 0; i < v->size; ++i) {
    auto value = gsl_vector_get(v,i);
    if (gsl_isinf(value) != 0) return true;
  }
  return false;
}
bool has_nan(const gsl_matrix *m) {
  for (size_t i = 0; i < m->size1; ++i)
    for (size_t j = 0; j < m->size2; ++j)
      if (gsl_isnan(gsl_matrix_get(m,i,j))) return true;
  return false;
}
bool has_inf(const gsl_matrix *m) {
  for (size_t i = 0; i < m->size1; ++i)
    for (size_t j = 0; j < m->size2; ++j) {
      auto value = gsl_matrix_get(m,i,j);
      if (gsl_isinf(value) != 0) return true;
    }
  return false;
}

double VectorVar(const gsl_vector *v) {
  double d, m = 0.0, m2 = 0.0;
  for (size_t i = 0; i < v->size; ++i) {
    d = gsl_vector_get(v, i);
    m += d;
    m2 += d * d;
  }
  m /= (double)v->size;
  m2 /= (double)v->size;
  return m2 - m * m;
}


void CenterMatrix(gsl_matrix *G) {
  double d;
  gsl_vector *w = gsl_vector_safe_alloc(G->size1);
  gsl_vector *Gw = gsl_vector_safe_alloc(G->size1);
  gsl_vector_set_all(w, 1.0);

  gsl_blas_dgemv(CblasNoTrans, 1.0, G, w, 0.0, Gw);
  gsl_blas_dsyr2(CblasUpper, -1.0 / (double)G->size1, Gw, w, G);
  gsl_blas_ddot(w, Gw, &d);
  gsl_blas_dsyr(CblasUpper, d / ((double)G->size1 * (double)G->size1), w, G);

  for (size_t i = 0; i < G->size1; ++i) {
    for (size_t j = 0; j < i; ++j) {
      d = gsl_matrix_get(G, j, i);
      gsl_matrix_set(G, i, j, d);
    }
  }

  gsl_vector_safe_free(w);
  gsl_vector_safe_free(Gw);

  return;
}

void CenterMatrix(gsl_matrix *G, const gsl_vector *w) {
  double d, wtw;
  gsl_vector *Gw = gsl_vector_safe_alloc(G->size1);

  gsl_blas_ddot(w, w, &wtw);
  gsl_blas_dgemv(CblasNoTrans, 1.0, G, w, 0.0, Gw);
  gsl_blas_dsyr2(CblasUpper, -1.0 / wtw, Gw, w, G);
  gsl_blas_ddot(w, Gw, &d);
  gsl_blas_dsyr(CblasUpper, d / (wtw * wtw), w, G);

  for (size_t i = 0; i < G->size1; ++i) {
    for (size_t j = 0; j < i; ++j) {
      d = gsl_matrix_get(G, j, i);
      gsl_matrix_set(G, i, j, d);
    }
  }

  gsl_vector_safe_free(Gw);

  return;
}

void CenterMatrix(gsl_matrix *G, const gsl_matrix *W) {
  gsl_matrix *WtW = gsl_matrix_safe_alloc(W->size2, W->size2);
  gsl_matrix *WtWi = gsl_matrix_safe_alloc(W->size2, W->size2);
  gsl_matrix *WtWiWt = gsl_matrix_safe_alloc(W->size2, G->size1);
  gsl_matrix *GW = gsl_matrix_safe_alloc(G->size1, W->size2);
  gsl_matrix *WtGW = gsl_matrix_safe_alloc(W->size2, W->size2);
  gsl_matrix *Gtmp = gsl_matrix_safe_alloc(G->size1, G->size1);

  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, W, W, 0.0, WtW);

  int sig;
  gsl_permutation *pmt = gsl_permutation_alloc(W->size2);
  LUDecomp(WtW, pmt, &sig);
  LUInvert(WtW, pmt, WtWi);

  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, WtWi, W, 0.0, WtWiWt);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, G, W, 0.0, GW);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, GW, WtWiWt, 0.0, Gtmp);

  gsl_matrix_sub(G, Gtmp);
  gsl_matrix_transpose(Gtmp);
  gsl_matrix_sub(G, Gtmp);

  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, W, GW, 0.0, WtGW);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, WtWiWt, WtGW, 0.0, GW);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, GW, WtWiWt, 0.0, Gtmp);

  gsl_matrix_add(G, Gtmp);

  gsl_matrix_safe_free(WtW);
  gsl_matrix_safe_free(WtWi);
  gsl_matrix_safe_free(WtWiWt);
  gsl_matrix_safe_free(GW);
  gsl_matrix_safe_free(WtGW);
  gsl_matrix_safe_free(Gtmp);

  return;
}

void StandardizeMatrix(gsl_matrix *G) {
  double d = 0.0;
  vector<double> vec_d;

  for (size_t i = 0; i < G->size1; ++i) {
    vec_d.push_back(gsl_matrix_get(G, i, i));
  }
  for (size_t i = 0; i < G->size1; ++i) {
    for (size_t j = i; j < G->size2; ++j) {
      if (j == i) {
        gsl_matrix_set(G, i, j, 1);
      } else {
        d = gsl_matrix_get(G, i, j);
        d /= sqrt(vec_d[i] * vec_d[j]);
        gsl_matrix_set(G, i, j, d);
        gsl_matrix_set(G, j, i, d);
      }
    }
  }

  return;
}

double ScaleMatrix(gsl_matrix *G) {
  double d = 0.0;

  for (size_t i = 0; i < G->size1; ++i) {
    d += gsl_matrix_get(G, i, i);
  }
  d /= (double)G->size1;

  if (d != 0) {
    gsl_matrix_scale(G, 1.0 / d);
  }

  return d;
}

bool isMatrixSymmetric(const gsl_matrix *G) {
  enforce(G->size1 == G->size2);
  auto m = G->data;
  auto size = G->size1;
  for(size_t c = 0; c < size; c++) {
    for(size_t r = 0; r < c; r++) {
      int x1 = c, y1 = r, x2 = r, y2 = c;
      auto idx1 = y1*size+x1, idx2 = y2*size+x2;
      if(m[idx1] != m[idx2]) {
        cout << "Mismatch coordinates (" << c << "," << r << ")" << m[idx1] << ":" << m[idx2] << "!" << endl;
        return false;
      }
    }
  }
  return true;
}

bool isMatrixPositiveDefinite(const gsl_matrix *G) {
  enforce(G->size1 == G->size2);
  auto G2 = gsl_matrix_safe_alloc(G->size1, G->size2);
  enforce_gsl(gsl_matrix_safe_memcpy(G2,G));
  auto handler = gsl_set_error_handler_off();
#if GSL_MAJOR_VERSION >= 2 && GSL_MINOR_VERSION >= 3
  auto s = gsl_linalg_cholesky_decomp(G2); 
#else
  auto s = gsl_linalg_cholesky_decomp(G2);
#endif
  gsl_set_error_handler(handler);
  if (s == GSL_SUCCESS) {
    gsl_matrix_safe_free(G2);
    return true;
  }
  gsl_matrix_free(G2);
  return (false);
}

gsl_vector *getEigenValues(const gsl_matrix *G) {
  enforce(G->size1 == G->size2);
  auto G2 = gsl_matrix_safe_alloc(G->size1, G->size2);
  enforce_gsl(gsl_matrix_safe_memcpy(G2,G));
  auto eworkspace = gsl_eigen_symm_alloc(G->size1);
  enforce(eworkspace);
  gsl_vector *eigenvalues = gsl_vector_safe_alloc(G->size1);
  enforce_gsl(gsl_eigen_symm(G2, eigenvalues, eworkspace));
  gsl_eigen_symm_free(eworkspace);
  gsl_matrix_safe_free(G2);
  return eigenvalues;
}



tuple<double, double, double> minmax(const gsl_vector *v) {
  auto min  = v->data[0];
  auto min1 = min;
  auto max  = min;
  for (size_t i=1; i<v->size; i++) {
    auto value = std::abs(v->data[i]);
    if (value < min) {
      min1 = min;
      min = value;
    }
    if (value > max)
      max = value;
  }
  return std::make_tuple(min, min1, max);
}

tuple<double, double, double> abs_minmax(const gsl_vector *v) {
  auto min  = std::abs(v->data[0]);
  auto min1 = min;
  auto max  = min;
  for (size_t i=1; i<v->size; i++) {
    auto value = std::abs(v->data[i]);
    if (value < min) {
      min1 = min;
      min = value;
    }
    if (value > max)
      max = value;
  }
  return std::make_tuple(min, min1, max);
}


bool has_negative_values_but_one(const gsl_vector *v) {
  bool one_skipped = false;
  for (size_t i=0; i<v->size; i++) {
    if (v->data[i] < 0.0) {
      if (one_skipped)
        return true;
      one_skipped = true;
    }
  }
  return false;
}

uint count_abs_small_values(const gsl_vector *v, double min) {
  uint count = 0;
  for (size_t i=0; i<v->size; i++) {
    if (std::abs(v->data[i]) < min) {
      count += 1;
    }
  }
  return count;
}


bool isMatrixIllConditioned(const gsl_vector *eigenvalues, double max_ratio) {
  auto t = abs_minmax(eigenvalues);
  auto absmin = get<0>(t);
  auto absmin1 = get<1>(t);
  auto absmax = get<2>(t);
  if (absmax/absmin1 > max_ratio) {
    #if !NDEBUG
    cerr << "**** DEBUG: Ratio |eigenmax|/|eigenmin| suggests matrix is ill conditioned for double precision" << endl;
    auto t = minmax(eigenvalues);
    auto min = get<0>(t);
    auto min1 = get<1>(t);
    auto max = get<2>(t);
    cerr << "**** DEBUG: Abs eigenvalues [Min " << absmin << ", " << absmin1 << " ... " << absmax << " Max] Ratio (" << absmax << "/" << absmin1 << ") = " << absmax/absmin1 << endl;
    cerr << "**** DEBUG: Eigenvalues [Min " << min << ", " << min1 << " ... " << max << " Max]" << endl;
    #endif
    return true;
  }
  return false;
}

double sum(const double *m, size_t rows, size_t cols) {
  double sum = 0.0;
  for (size_t i = 0; i<rows*cols; i++)
    sum += m[i];
  return sum;
}

double SumVector(const gsl_vector *v) {
  double sum = 0;
  for (size_t i = 0; i < v->size; i++ ) {
    sum += gsl_vector_get(v, i);
  }
  return( sum );
}


double CenterVector(gsl_vector *y) {
  double d = 0.0;

  for (size_t i = 0; i < y->size; ++i) {
    d += gsl_vector_get(y, i);
  }
  d /= (double)y->size;

  gsl_vector_add_constant(y, -1.0 * d);

  return d;
}

void CenterVector(gsl_vector *y, const gsl_matrix *W) {
  gsl_matrix *WtW = gsl_matrix_safe_alloc(W->size2, W->size2);
  gsl_vector *Wty = gsl_vector_safe_alloc(W->size2);
  gsl_vector *WtWiWty = gsl_vector_safe_alloc(W->size2);

  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, W, W, 0.0, WtW);
  gsl_blas_dgemv(CblasTrans, 1.0, W, y, 0.0, Wty);

  int sig;
  gsl_permutation *pmt = gsl_permutation_alloc(W->size2);
  LUDecomp(WtW, pmt, &sig);
  LUSolve(WtW, pmt, Wty, WtWiWty);

  gsl_blas_dgemv(CblasNoTrans, -1.0, W, WtWiWty, 1.0, y);

  gsl_matrix_safe_free(WtW);
  gsl_vector_safe_free(Wty);
  gsl_vector_safe_free(WtWiWty);

  return;
}

void StandardizeVector(gsl_vector *y) {
  double d = 0.0, m = 0.0, v = 0.0;

  for (size_t i = 0; i < y->size; ++i) {
    d = gsl_vector_get(y, i);
    m += d;
    v += d * d;
  }
  m /= (double)y->size;
  v /= (double)y->size;
  v -= m * m;

  gsl_vector_add_constant(y, -1.0 * m);
  gsl_vector_scale(y, 1.0 / sqrt(v));

  return;
}

void CalcUtX(const gsl_matrix *U, gsl_matrix *UtX) {
  gsl_matrix *X = gsl_matrix_safe_alloc(UtX->size1, UtX->size2);
  gsl_matrix_safe_memcpy(X, UtX);
  fast_dgemm("T", "N", 1.0, U, X, 0.0, UtX);
  gsl_matrix_safe_free(X);
}

void CalcUtX(const gsl_matrix *U, const gsl_matrix *X, gsl_matrix *UtX) {
  fast_dgemm("T", "N", 1.0, U, X, 0.0, UtX);
}

void CalcUtX(const gsl_matrix *U, const gsl_vector *x, gsl_vector *Utx) {
  gsl_blas_dgemv(CblasTrans, 1.0, U, x, 0.0, Utx);
}

void Kronecker(const gsl_matrix *K, const gsl_matrix *V, gsl_matrix *H) {
  for (size_t i = 0; i < K->size1; i++) {
    for (size_t j = 0; j < K->size2; j++) {
      gsl_matrix_view H_sub = gsl_matrix_submatrix(
          H, i * V->size1, j * V->size2, V->size1, V->size2);
      gsl_matrix_safe_memcpy(&H_sub.matrix, V);
      gsl_matrix_scale(&H_sub.matrix, gsl_matrix_get(K, i, j));
    }
  }
  return;
}

void KroneckerSym(const gsl_matrix *K, const gsl_matrix *V, gsl_matrix *H) {
  for (size_t i = 0; i < K->size1; i++) {
    for (size_t j = i; j < K->size2; j++) {
      gsl_matrix_view H_sub = gsl_matrix_submatrix(
          H, i * V->size1, j * V->size2, V->size1, V->size2);
      gsl_matrix_safe_memcpy(&H_sub.matrix, V);
      gsl_matrix_scale(&H_sub.matrix, gsl_matrix_get(K, i, j));

      if (i != j) {
        gsl_matrix_view H_sub_sym = gsl_matrix_submatrix(
            H, j * V->size1, i * V->size2, V->size1, V->size2);
        gsl_matrix_safe_memcpy(&H_sub_sym.matrix, &H_sub.matrix);
      }
    }
  }
  return;
}


double CalcHWE(const size_t n_hom1, const size_t n_hom2, const size_t n_ab) {
  if ((n_hom1 + n_hom2 + n_ab) == 0) {
    return 1;
  }

  int n_aa = n_hom1 < n_hom2 ? n_hom1 : n_hom2;
  int n_bb = n_hom1 < n_hom2 ? n_hom2 : n_hom1;

  int rare_copies = 2 * n_aa + n_ab;
  int genotypes = n_ab + n_bb + n_aa;

  double *het_probs = (double *)malloc((rare_copies + 1) * sizeof(double));
  if (het_probs == NULL)
    cout << "Internal error: SNP-HWE: Unable to allocate array" << endl;

  int i;
  for (i = 0; i <= rare_copies; i++)
    het_probs[i] = 0.0;

  
  int mid = ((long int)rare_copies *
             (2 * (long int)genotypes - (long int)rare_copies)) /
            (2 * (long int)genotypes);

  
  if ((rare_copies & 1) ^ (mid & 1))
    mid++;

  int curr_hets = mid;
  int curr_homr = (rare_copies - mid) / 2;
  int curr_homc = genotypes - curr_hets - curr_homr;

  het_probs[mid] = 1.0;
  double sum = het_probs[mid];
  for (curr_hets = mid; curr_hets > 1; curr_hets -= 2) {
    het_probs[curr_hets - 2] = het_probs[curr_hets] * curr_hets *
                               (curr_hets - 1.0) /
                               (4.0 * (curr_homr + 1.0) * (curr_homc + 1.0));
    sum += het_probs[curr_hets - 2];

    
    curr_homr++;
    curr_homc++;
  }

  curr_hets = mid;
  curr_homr = (rare_copies - mid) / 2;
  curr_homc = genotypes - curr_hets - curr_homr;
  for (curr_hets = mid; curr_hets <= rare_copies - 2; curr_hets += 2) {
    het_probs[curr_hets + 2] = het_probs[curr_hets] * 4.0 * curr_homr *
                               curr_homc /
                               ((curr_hets + 2.0) * (curr_hets + 1.0));
    sum += het_probs[curr_hets + 2];

   
    curr_homr--;
    curr_homc--;
  }

  for (i = 0; i <= rare_copies; i++)
    het_probs[i] /= sum;

  double p_hwe = 0.0;

  
  for (i = 0; i <= rare_copies; i++) {
    if (het_probs[i] > het_probs[n_ab])
      continue;
    p_hwe += het_probs[i];
  }

  p_hwe = p_hwe > 1.0 ? 1.0 : p_hwe;

  free(het_probs);

  return p_hwe;
}

double UcharToDouble02(const unsigned char c) { return (double)c * 0.01; }

unsigned char Double02ToUchar(const double dosage) {
  return (int)(dosage * 100);
}


