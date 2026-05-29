/*
    Dynamic Genomic Best linear Unbiased Prediction  (DynGBLUP)
    Copyright © 2026, Meixia Ye
    
*/

#ifndef __EIGENLIB_H__
#define __EIGENLIB_H__


void eigenlib_dgemm(const char *TransA, const char *TransB, const double alpha,
                    const gsl_matrix *A, const gsl_matrix *B, const double beta,
                    gsl_matrix *C);
void eigenlib_dgemv(const char *TransA, const double alpha, const gsl_matrix *A,
                    const gsl_vector *x, const double beta, gsl_vector *y);
void eigenlib_invert(gsl_matrix *A);
void eigenlib_dsyr(const double alpha, const gsl_vector *b, gsl_matrix *A);
void eigenlib_eigensymm(const gsl_matrix *G, gsl_matrix *U, gsl_vector *eval);

#endif
