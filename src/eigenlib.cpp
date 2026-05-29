/*
    Dynamic Genomic Best linear Unbiased Prediction  (DynGBLUP)
    Copyright © 2026, Meixia Ye
    
*/

#include "Eigen/Dense"
#include "gsl/gsl_matrix.h"
#include <cmath>
#include <iostream>
#include <vector>


using namespace std;
using namespace Eigen;


void eigenlib_dgemm(const char *TransA, const char *TransB, const double alpha,
                    const gsl_matrix *A, const gsl_matrix *B, const double beta,
                    gsl_matrix *C) {
  Map<Matrix<double, Dynamic, Dynamic, RowMajor>, 0, OuterStride<Dynamic>>
      A_mat(A->data, A->size1, A->size2, OuterStride<Dynamic>(A->tda));
  Map<Matrix<double, Dynamic, Dynamic, RowMajor>, 0, OuterStride<Dynamic>>
      B_mat(B->data, B->size1, B->size2, OuterStride<Dynamic>(B->tda));
  Map<Matrix<double, Dynamic, Dynamic, RowMajor>, 0, OuterStride<Dynamic>>
      C_mat(C->data, C->size1, C->size2, OuterStride<Dynamic>(C->tda));

  if (*TransA == 'N' || *TransA == 'n') {
    if (*TransB == 'N' || *TransB == 'n') {
      C_mat = alpha * A_mat * B_mat + beta * C_mat;
    } else {
      C_mat = alpha * A_mat * B_mat.transpose() + beta * C_mat;
    }
  } else {
    if (*TransB == 'N' || *TransB == 'n') {
      C_mat = alpha * A_mat.transpose() * B_mat + beta * C_mat;
    } else {
      C_mat = alpha * A_mat.transpose() * B_mat.transpose() + beta * C_mat;
    }
  }
}

void eigenlib_dgemv(const char *TransA, const double alpha, const gsl_matrix *A,
                    const gsl_vector *x, const double beta, gsl_vector *y) {
  Map<Matrix<double, Dynamic, Dynamic, RowMajor>, 0, OuterStride<Dynamic>>
      A_mat(A->data, A->size1, A->size2, OuterStride<Dynamic>(A->tda));
  Map<Matrix<double, Dynamic, 1>, 0, InnerStride<Dynamic>> x_vec(
      x->data, x->size, InnerStride<Dynamic>(x->stride));
  Map<Matrix<double, Dynamic, 1>, 0, InnerStride<Dynamic>> y_vec(
      y->data, y->size, InnerStride<Dynamic>(y->stride));

  if (*TransA == 'N' || *TransA == 'n') {
    y_vec = alpha * A_mat * x_vec + beta * y_vec;
  } else {
    y_vec = alpha * A_mat.transpose() * x_vec + beta * y_vec;
  }
}

void eigenlib_invert(gsl_matrix *A) {
  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> A_mat(A->data, A->size1,
                                                        A->size2);
  A_mat = A_mat.inverse();
}

void eigenlib_dsyr(const double alpha, const gsl_vector *b, gsl_matrix *A) {
  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> A_mat(A->data, A->size1,
                                                        A->size2);
  Map<Matrix<double, Dynamic, 1>, 0, OuterStride<Dynamic>> b_vec(
      b->data, b->size, OuterStride<Dynamic>(b->stride));
  A_mat = alpha * b_vec * b_vec.transpose() + A_mat;
}

void eigenlib_eigensymm(const gsl_matrix *G, gsl_matrix *U, gsl_vector *eval) {
  Map<Matrix<double, Dynamic, Dynamic, RowMajor>, 0, OuterStride<Dynamic>>
      G_mat(G->data, G->size1, G->size2, OuterStride<Dynamic>(G->tda));
  Map<Matrix<double, Dynamic, Dynamic, RowMajor>, 0, OuterStride<Dynamic>>
      U_mat(U->data, U->size1, U->size2, OuterStride<Dynamic>(U->tda));
  Map<Matrix<double, Dynamic, 1>, 0, OuterStride<Dynamic>> eval_vec(
      eval->data, eval->size, OuterStride<Dynamic>(eval->stride));

  SelfAdjointEigenSolver<MatrixXd> es(G_mat);
  if (es.info() != Success)
    abort();
  eval_vec = es.eigenvalues();
  U_mat = es.eigenvectors();
}
