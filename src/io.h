/*
    Dynamic Genomic Best linear Unbiased Prediction  (DynGBLUP)
    Copyright © 2026, Meixia Ye
    
*/

#ifndef __IO_H__
#define __IO_H__

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
#include <algorithm>
#include <map>
#include <vector>

#include "gzstream.h"
#include "param.h"

#define tab(col) ( col ? "\t" : "")

using namespace std;

void ProgressBar(string str, double p, double total, double ratio = -1.0);

std::istream &safeGetline(std::istream &is, std::string &t);


bool ReadFile_bim(const string &file_bim, vector<SNPINFO> &snpInfo);
bool ReadFile_fam(const string &file_fam, vector<vector<int>> &indicator_pheno,
                  vector<vector<double>> &pheno, map<string, int> &mapID2num,
                  const vector<size_t> &p_column);

bool ReadFile_cvt(const string &file_cvt, vector<int> &indicator_cvt,
                  vector<vector<double>> &cvt, size_t &n_cvt);

bool ReadFile_bed(const string &file_bed, const set<string> &setSnps,
                  const gsl_matrix *W, vector<int> &indicator_idv,
                  vector<int> &indicator_snp, vector<SNPINFO> &snpInfo,
                  const double &maf_level, const double &miss_level,
                  const double &hwe_level, const double &r2_level,
                  size_t &ns_test);

void ReadFile_kin(const string &file_kin, vector<int> &indicator_idv,
                  map<string, int> &mapID2num, const size_t k_mode, bool &error,
                  gsl_matrix *G);

void ReadFile_S(const string &file_S, vector<int> &indicator_idv,vector<int> &indicator_snp,
                  bool &error,
                  gsl_matrix *S);

void ReadFile_eigenU(const string &file_u, bool &error, gsl_matrix *U);
void ReadFile_eigenD(const string &file_d, bool &error, gsl_vector *eval);




bool ReadFile_bed(const string &file_bed, vector<int> &indicator_idv,
                  vector<int> &indicator_snp, gsl_matrix *UtX, gsl_matrix *K,
                  const bool calc_K);

bool ReadFile_bed(const string &file_bed, vector<int> &indicator_idv,
                  vector<int> &indicator_snp, vector<vector<unsigned char>> &Xt,
                  gsl_matrix *K, const bool calc_K, const size_t ni_test,
                  const size_t ns_test);

bool CountFileLines(const string &file_input, size_t &n_lines);


bool ReadHeader_io(const string &line, HEADER &header);

bool PlinkKinS(const string &file_bed, vector<int> &indicator_snp,
              const int k_mode, const int display_pace,
              gsl_matrix *matrix_kin, gsl_matrix *matrix_S );


#endif
