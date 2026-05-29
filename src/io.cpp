/*
    Dynamic Genomic Best linear Unbiased Prediction  (DynGBLUP)
    Copyright © 2026, Meixia Ye
    
*/

#include <assert.h>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "gsl/gsl_blas.h"
#include "gsl/gsl_cdf.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"

#include "debug.h"
#include "fastblas.h"
#include "gzstream.h"
#include "io.h"
#include "lapack.h"
#include "mathfunc.h"

using namespace std;


void ProgressBar(string str, double p, double total, double ratio) {
  assert(p<=total);
  assert(p>=0);
  if (total <= 0.0) return;
  const double progress = (100.0 * p / total);
  const uint barsize = (int)(progress / 2.0); 
  
  assert(barsize < 101); 
  if (barsize > 0) {
    cout << std::string(barsize,'=');
  }
  cout << std::string(50-barsize,' ');
  cout << setprecision(0) << fixed << " " << progress << "%";
  if (ratio != -1.0)
    cout << setprecision(2) << "    " << ratio;
  cout << "\r" << flush;
}

bool isBlankLine(char const *line) {
  for (char const *cp = line; *cp; ++cp) {
    if (!isspace(*cp))
      return false;
  }
  return true;
}

bool isBlankLine(std::string const &line) { return isBlankLine(line.c_str()); }


std::istream &safeGetline(std::istream &is, std::string &t) {
  t.clear();

  
  std::istream::sentry se(is, true);
  std::streambuf *sb = is.rdbuf();

  for (;;) {
    int c = sb->sbumpc();
    switch (c) {
    case '\n':
      return is;
    case '\r':
      if (sb->sgetc() == '\n')
        sb->sbumpc();
      return is;
    case EOF:

      
      if (t.empty())
        is.setstate(std::ios::eofbit);
      return is;
    default:
      t += (char)c;
    }
  }
}




bool ReadFile_column(const string &file_pheno, vector<int> &indicator_idv,
                     vector<double> &pheno, const int &p_column) 
{
  debug_msg("entered");
  indicator_idv.clear();
  pheno.clear();

  igzstream infile(file_pheno.c_str(), igzstream::in);
  if (!infile) {
    cout << "error! fail to open phenotype file: " << file_pheno << endl;
    return false;
  }

  string line;
  char *ch_ptr;

  string id;
  double p;
  while (!safeGetline(infile, line).eof()) {
    ch_ptr = strtok_safe((char *)line.c_str(), " , \t");
    for (int i = 0; i < (p_column - 1); ++i) {
      ch_ptr = strtok(NULL, " , \t");
    }
    enforce_msg(ch_ptr,"Problem reading PHENO column");
    if (strcmp(ch_ptr, "NA") == 0) {
      indicator_idv.push_back(0);
      pheno.push_back(-9);
    } else {

      p = atof(ch_ptr);
      indicator_idv.push_back(1);
      pheno.push_back(p);
    }
  }

  infile.close();
  infile.clear();

  return true;
}




bool ReadFile_cvt(const string &file_cvt, vector<int> &indicator_cvt,
                  vector<vector<double>> &cvt, size_t &n_cvt) 
{
  debug_msg("entered");
  indicator_cvt.clear();

  ifstream infile(file_cvt.c_str(), ifstream::in);
  if (!infile) {
    cout << "error! fail to open covariates file: " << file_cvt << endl;
    return false;
  }

  string line;
  char *ch_ptr;
  double d;

  int flag_na = 0;

  while (!safeGetline(infile, line).eof()) {
    vector<double> v_d;
    flag_na = 0;
    ch_ptr = strtok((char *)line.c_str(), " , \t");
    while (ch_ptr != NULL) {
      if (strcmp(ch_ptr, "NA") == 0) {
        flag_na = 1;
        d = -9;
      } else {
        d = atof(ch_ptr);
      }

      v_d.push_back(d);
      ch_ptr = strtok(NULL, " , \t");
    }

    if (flag_na == 0) {
      indicator_cvt.push_back(1);
    } else {
      indicator_cvt.push_back(0);
    }
    cvt.push_back(v_d);
  }

  if (indicator_cvt.empty()) {
    n_cvt = 0;
  } else {
    flag_na = 0;
    for (vector<int>::size_type i = 0; i < indicator_cvt.size(); ++i) {
      if (indicator_cvt[i] == 0) {
        continue;
      }

      if (flag_na == 0) {
        flag_na = 1;
        n_cvt = cvt[i].size();
      }
      if (flag_na != 0 && n_cvt != cvt[i].size()) {
        cout << "error! number of covariates in row " << i
             << " do not match other rows." << endl;
        return false;
      }
    }
  }

 
  infile.close();
  infile.clear();

  return true;
}


bool ReadFile_bim(const string &file_bim, vector<SNPINFO> &snpInfo) 
{
  debug_msg("entered");
  snpInfo.clear();

  ifstream infile(file_bim.c_str(), ifstream::in);
  if (!infile) {
    cout << "error opening .bim file: " << file_bim << endl;
    return false;
  }

  string line;
  char *ch_ptr;

  string rs;
  long int b_pos;
  string chr;
  double cM;
  string major;
  string minor;

  while (getline(infile, line)) {
    ch_ptr = strtok_safe((char *)line.c_str(), " \t");
    chr = ch_ptr;
    ch_ptr = strtok_safe(NULL, " \t");
    rs = ch_ptr;
    ch_ptr = strtok_safe(NULL, " \t");
    cM = atof(ch_ptr);
    ch_ptr = strtok_safe(NULL, " \t");
    b_pos = atol(ch_ptr);
    ch_ptr = strtok_safe(NULL, " \t");
    minor = ch_ptr;
    ch_ptr = strtok_safe(NULL, " \t");
    major = ch_ptr;

    SNPINFO sInfo = {chr, rs, cM, b_pos, minor, major, 0, -9, -9, 0, 0, 0};
    snpInfo.push_back(sInfo);
  }

  infile.close();
  infile.clear();
  return true;
}


bool ReadFile_fam(const string &file_fam, vector<vector<int>> &indicator_pheno,
                  vector<vector<double>> &pheno, map<string, int> &mapID2num,
                  const vector<size_t> &p_column) 
{
  debug_msg("entered");
  indicator_pheno.clear();
  pheno.clear();
  mapID2num.clear();

  igzstream infile(file_fam.c_str(), igzstream::in);
  if (!infile) {
    cout << "error opening .fam file: " << file_fam << endl;
    return false;
  }

  string line;
  char *ch_ptr;

  string id;
  int c = 0;
  double p;

  vector<double> pheno_row;
  vector<int> ind_pheno_row;

  size_t p_max = *max_element(p_column.begin(), p_column.end());
  map<size_t, size_t> mapP2c;
  for (size_t i = 0; i < p_column.size(); i++) {
    mapP2c[p_column[i]] = i;
    pheno_row.push_back(-9);
    ind_pheno_row.push_back(0);
  }

  while (!safeGetline(infile, line).eof()) {
    ch_ptr = strtok_safe((char *)line.c_str(), " \t");
    ch_ptr = strtok_safe(NULL, " \t");
    id = ch_ptr;
    ch_ptr = strtok_safe(NULL, " \t");
    ch_ptr = strtok_safe(NULL, " \t");
    ch_ptr = strtok_safe(NULL, " \t");
    ch_ptr = strtok(NULL, " \t");

    size_t i = 0;
    while (i < p_max) {
      if (mapP2c.count(i + 1) != 0) {
        enforce_msg(ch_ptr,"Problem reading FAM file (phenotypes out of range)");

        if (strcmp(ch_ptr, "NA") == 0) {
          ind_pheno_row[mapP2c[i + 1]] = 0;
          pheno_row[mapP2c[i + 1]] = -9;
        } else {
          p = atof(ch_ptr);

          if (p == -9) {
            ind_pheno_row[mapP2c[i + 1]] = 0;
            pheno_row[mapP2c[i + 1]] = -9;
          } else {
            ind_pheno_row[mapP2c[i + 1]] = 1;
            pheno_row[mapP2c[i + 1]] = p;
          }
        }
      }
      i++;
      ch_ptr = strtok(NULL, " , \t");
    }

    indicator_pheno.push_back(ind_pheno_row);
    pheno.push_back(pheno_row);

    mapID2num[id] = c;
    c++;
  }

  infile.close();
  infile.clear();
  return true;
}





bool ReadFile_bed(const string &file_bed, const set<string> &setSnps,
                  const gsl_matrix *W, vector<int> &indicator_idv,
                  vector<int> &indicator_snp, vector<SNPINFO> &snpInfo,
                  const double &maf_level, const double &miss_level,
                  const double &hwe_level, const double &r2_level,
                  size_t &ns_test) 
{
  debug_msg("entered");
  indicator_snp.clear();
  size_t ns_total = snpInfo.size();

 
  ofstream snp_filter_log("snp_filter_log.txt");
  if (!snp_filter_log) {
    cout << "error creating snp filter log file" << endl;
  } else {
    snp_filter_log << "RS\tCHR\tBP\tMAF\tMISS_RATE\tN0\tN1\tN2\tFILTER_REASON\tSTATUS" << endl;
    cout << "DEBUG: Created snp_filter_log.txt" << endl;
  }

  ifstream infile(file_bed.c_str(), ios::binary);
  if (!infile) {
    cout << "error reading bed file:" << file_bed << endl;
    if (snp_filter_log) snp_filter_log.close();
    return false;
  }

  gsl_vector *genotype = gsl_vector_safe_alloc(W->size1);
  gsl_vector *genotype_miss = gsl_vector_safe_alloc(W->size1);
  gsl_matrix *WtW = gsl_matrix_safe_alloc(W->size2, W->size2);
  gsl_matrix *WtWi = gsl_matrix_safe_alloc(W->size2, W->size2);
  gsl_vector *Wtx = gsl_vector_safe_alloc(W->size2);
  gsl_vector *WtWiWtx = gsl_vector_safe_alloc(W->size2);
  gsl_permutation *pmt = gsl_permutation_alloc(W->size2);

  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, W, W, 0.0, WtW);
  int sig;
  LUDecomp(WtW, pmt, &sig);
  LUInvert(WtW, pmt, WtWi);

  double v_x, v_w, geno;
  size_t c_idv = 0;

  char ch[1];
  bitset<8> b;

  size_t ni_total = indicator_idv.size();
  size_t ni_test = 0;
  for (size_t i = 0; i < ni_total; ++i) {
    ni_test += indicator_idv[i];
  }
  ns_test = 0;


  size_t n_bit;
  if (ni_total % 4 == 0) {
    n_bit = ni_total / 4;
  } else {
    n_bit = ni_total / 4 + 1;
  }


  for (int i = 0; i < 3; ++i) {
    infile.read(ch, 1);
    b = ch[0];
  }

  double maf;
  size_t n_miss;
  size_t n_0, n_1, n_2, c;


  for (size_t t = 0; t < ns_total; ++t) {
    string filter_reason = "PASS"; 


    infile.seekg(t * n_bit + 3);

    if (setSnps.size() != 0 && setSnps.count(snpInfo[t].rs_number) == 0) {
      snpInfo[t].n_miss = -9;
      snpInfo[t].missingness = -9;
      snpInfo[t].maf = -9;
      snpInfo[t].file_position = t;
      indicator_snp.push_back(0);
      

      if (snp_filter_log) {
        snp_filter_log << snpInfo[t].rs_number << "\t" 
                       << snpInfo[t].chr << "\t" 
                       << snpInfo[t].base_position << "\t"
                       << "-9\t-9\t-9\t-9\t-9\t"
                       << "NOT_IN_SNP_SET\tFILTERED" << endl;
      }
      continue;
    }

    c = 0;
    maf = 0.0;
    n_miss = 0;
    n_0 = 0;
    n_1 = 0;
    n_2 = 0;
    c_idv = 0;
    gsl_vector_set_zero(genotype_miss);
    for (size_t i = 0; i < n_bit; ++i) {
      infile.read(ch, 1);
      b = ch[0];

      for (size_t j = 0; j < 4; ++j) {
        if ((i == (n_bit - 1)) && c == ni_total) {
          break;
        }
        if (indicator_idv[c] == 0) {
          c++;
          continue;
        }
        c++;

        if (b[2 * j] == 0) {
          if (b[2 * j + 1] == 0) {
            gsl_vector_set(genotype, c_idv, 2.0);
            maf += 2.0;
            n_2++;
          } else {
            gsl_vector_set(genotype, c_idv, 1.0);
            maf += 1.0;
            n_1++;
          }
        } else {
          if (b[2 * j + 1] == 1) {
            gsl_vector_set(genotype, c_idv, 0.0);
            maf += 0.0;
            n_0++;
          } else {
            gsl_vector_set(genotype_miss, c_idv, 1);
            n_miss++;
          }
        }
        c_idv++;
      }
    }

    maf /= 2.0 * (double)(ni_test - n_miss);

    snpInfo[t].n_miss = n_miss;
    snpInfo[t].missingness = (double)n_miss / (double)ni_test;
    snpInfo[t].maf = maf;
    snpInfo[t].n_idv = ni_test - n_miss;
    snpInfo[t].n_nb = 0;
    snpInfo[t].file_position = t;

    if ((double)n_miss / (double)ni_test > miss_level) {
      filter_reason = "MISSING_RATE";
      indicator_snp.push_back(0);
    }
    else if ((maf < maf_level || maf > (1.0 - maf_level)) && maf_level != -1) {
      filter_reason = "MAF";
      indicator_snp.push_back(0);
    }
    else if ((n_0 + n_1) <= 1 || (n_1 + n_2) <= 1 || (n_2 + n_0) <= 1) {
      filter_reason = "GENOTYPE_COUNT";
      indicator_snp.push_back(0);
    }
    else if (hwe_level != 0 && maf_level != -1) {
      double hwe_p = CalcHWE(n_0, n_2, n_1);
      if (hwe_p < hwe_level) {
        filter_reason = "HWE";
        indicator_snp.push_back(0);
      }
    }
    else {
      for (size_t i = 0; i < genotype->size; ++i) {
        if (gsl_vector_get(genotype_miss, i) == 1) {
          geno = maf * 2.0;
          gsl_vector_set(genotype, i, geno);
        }
      }

      gsl_blas_dgemv(CblasTrans, 1.0, W, genotype, 0.0, Wtx);
      gsl_blas_dgemv(CblasNoTrans, 1.0, WtWi, Wtx, 0.0, WtWiWtx);
      gsl_blas_ddot(genotype, genotype, &v_x);
      gsl_blas_ddot(Wtx, WtWiWtx, &v_w);

      if (W->size2 != 1 && v_w / v_x > r2_level) {
        filter_reason = "COVARIATE_R2";
        indicator_snp.push_back(0);
      }
    }

    
    if (snp_filter_log) {
      string status = (filter_reason == "PASS") ? "KEPT" : "FILTERED";
      snp_filter_log << snpInfo[t].rs_number << "\t" 
                     << snpInfo[t].chr << "\t" 
                     << snpInfo[t].base_position << "\t"
                     << maf << "\t" 
                     << (double)n_miss / (double)ni_test << "\t"
                     << n_0 << "\t" << n_1 << "\t" << n_2 << "\t"
                     << filter_reason << "\t" << status << endl;
    }

   
    if (filter_reason == "PASS") {
      indicator_snp.push_back(1);
      ns_test++;
    }
  }

  gsl_vector_free(genotype);
  gsl_vector_free(genotype_miss);
  gsl_matrix_free(WtW);
  gsl_matrix_free(WtWi);
  gsl_vector_free(Wtx);
  gsl_vector_free(WtWiWtx);
  gsl_permutation_free(pmt);

  infile.close();
  infile.clear();
  
  if (snp_filter_log) {
    snp_filter_log.close();
    cout << "DEBUG: Closed snp_filter_log.txt" << endl;
  }

  return true;
}

void ReadFile_kin(const string &file_kin, vector<int> &indicator_idv,
                  map<string, int> &mapID2num, const size_t k_mode, bool &error,
                  gsl_matrix *G) 
{
  debug_msg("entered");
  igzstream infile(file_kin.c_str(), igzstream::in);
  if (!infile) {
    cout << "error! fail to open kinship file: " << file_kin << endl;
    error = true;
    return;
  }

  size_t ni_total = indicator_idv.size();

  gsl_matrix_set_zero(G);

  string line;
  char *ch_ptr;
  double d;

  if (k_mode == 1) {
    size_t i_test = 0, i_total = 0, j_test = 0, j_total = 0;
    while (getline(infile, line)) {
      if (i_total == ni_total) {
        fail_msg("number of rows in the kinship file is larger than the number of phentypes");
      }

      if (indicator_idv[i_total] == 0) {
        i_total++;
        continue;
      }

      j_total = 0;
      j_test = 0;
      ch_ptr = strtok((char *)line.c_str(), " , \t");
      while (ch_ptr != NULL) {
        if (j_total == ni_total) {
          fail_msg(string("number of columns in the kinship file is larger than the number of individuals for row = ")+to_string(i_total));
        }

        d = atof(ch_ptr);
        if (indicator_idv[j_total] == 1) {
          gsl_matrix_set(G, i_test, j_test, d);
          j_test++;
        }
        j_total++;

        ch_ptr = strtok(NULL, " , \t");
      }
      if (j_total != ni_total) {
        string msg = "number of columns in the kinship file does not match the number of individuals for row = " + to_string( i_total );
        fail_msg(msg);
      }
      i_total++;
      i_test++;
    }
    if (i_total != ni_total) {
      fail_msg("number of rows in the kinship file does not match the number of individuals.");
    }
  } else {
    map<size_t, size_t> mapID2ID;
    size_t c = 0;
    for (size_t i = 0; i < indicator_idv.size(); i++) {
      if (indicator_idv[i] == 1) {
        mapID2ID[i] = c;
        c++;
      }
    }

    string id1, id2;
    double Cov_d;
    size_t n_id1, n_id2;

    while (getline(infile, line)) {
      ch_ptr = strtok_safe((char *)line.c_str(), " , \t");
      id1 = ch_ptr;
      ch_ptr = strtok_safe(NULL, " , \t");
      id2 = ch_ptr;
      ch_ptr = strtok_safe(NULL, " , \t");
      d = atof(ch_ptr);
      if (mapID2num.count(id1) == 0 || mapID2num.count(id2) == 0) {
        continue;
      }
      if (indicator_idv[mapID2num[id1]] == 0 ||
          indicator_idv[mapID2num[id2]] == 0) {
        continue;
      }

      n_id1 = mapID2ID[mapID2num[id1]];
      n_id2 = mapID2ID[mapID2num[id2]];

      Cov_d = gsl_matrix_get(G, n_id1, n_id2);
      if (Cov_d != 0 && Cov_d != d) {
        cerr << "error! redundant and unequal terms in the "
             << "kinship file, for id1 = " << id1 << " and id2 = " << id2
             << endl;
        fail_msg("");
      } else {
        gsl_matrix_set(G, n_id1, n_id2, d);
        gsl_matrix_set(G, n_id2, n_id1, d);
      }
    }
  }

  infile.close();
  infile.clear();

  return;
}

void ReadFile_S(const string &file_S, vector<int> &indicator_idv, vector<int> &indicator_snp, 
                  bool &error,
                  gsl_matrix *S) 
{
  debug_msg("entered");
  igzstream infile(file_S.c_str(), igzstream::in);
  if (!infile) {
    cout << "error! fail to open S file: " << file_S << endl;
    error = true;
    return;
  }

  size_t ni_total = indicator_idv.size();
  printf("indicator_idv. = %zu \n", indicator_idv.size());
  printf("indicator_snp_idv. = %zu \n", indicator_snp.size());

  gsl_matrix_set_zero(S);

  string line;
  char *ch_ptr;
  double d;

    size_t i_test = 0, i_total = 0, j_test = 0, j_total = 0;

    while (getline(infile, line)) {
      if (i_total == ni_total) {
        fail_msg("number of rows in the S file is larger than the number of phentypes");
      }

      if (indicator_idv[i_total] == 0) {
        i_total++;
        continue;
      }

      j_total = 0;
      j_test = 0;
      ch_ptr = strtok((char *)line.c_str(), " , \t");
      while (ch_ptr != NULL) {
      
       

        d = atof(ch_ptr);
          gsl_matrix_set(S, i_test, j_test, d);
          j_test++;
      
        j_total++;

        ch_ptr = strtok(NULL, " , \t");
      }

       
      i_total++;
      i_test++;
    }

     printf("i_total = %zu \n", i_total);


    if (i_total != ni_total) {
      fail_msg("number of rows in the S file does not match the number of individuals.");
    }
    
  infile.close();
  infile.clear();
  printf("indicator_idv. = %zu \n", indicator_idv.size());
  return;
}


void ReadFile_eigenU(const string &file_ku, bool &error, gsl_matrix *U) {
  debug_msg("entered");
  igzstream infile(file_ku.c_str(), igzstream::in);
  if (!infile) {
    cout << "error! fail to open the U file: " << file_ku << endl;
    error = true;
    return;
  }

  size_t n_row = U->size1, n_col = U->size2, i_row = 0, i_col = 0;

  gsl_matrix_set_zero(U);

  string line;
  char *ch_ptr;
  double d;

  while (getline(infile, line)) {
    if (i_row == n_row) {
      cout << "error! number of rows in the U file is larger "
           << "than expected." << endl;
      error = true;
    }

    i_col = 0;
    ch_ptr = strtok((char *)line.c_str(), " , \t");
    while (ch_ptr != NULL) {
      if (i_col == n_col) {
        cout << "error! number of columns in the U file "
             << "is larger than expected, for row = " << i_row << endl;
        error = true;
      }

      d = atof(ch_ptr);
      gsl_matrix_set(U, i_row, i_col, d);
      i_col++;

      ch_ptr = strtok(NULL, " , \t");
    }

    i_row++;
  }

  infile.close();
  infile.clear();

  return;
}

void ReadFile_eigenD(const string &file_kd, bool &error, gsl_vector *eval) {
  debug_msg("entered");
  igzstream infile(file_kd.c_str(), igzstream::in);
  if (!infile) {
    cout << "error! fail to open the D file: " << file_kd << endl;
    error = true;
    return;
  }

  size_t n_row = eval->size, i_row = 0;

  gsl_vector_set_zero(eval);

  string line;
  char *ch_ptr;
  double d;

  while (getline(infile, line)) {
    if (i_row == n_row) {
      cout << "error! number of rows in the D file is larger "
           << "than expected." << endl;
      error = true;
    }

    ch_ptr = strtok_safe((char *)line.c_str(), " , \t");
    d = atof(ch_ptr);

    ch_ptr = strtok(NULL, " , \t");
    if (ch_ptr != NULL) {
      cout << "error! number of columns in the D file is larger "
           << "than expected, for row = " << i_row << endl;
      error = true;
    }

    gsl_vector_set(eval, i_row, d);

    i_row++;
  }

  infile.close();
  infile.clear();

  return;
}





bool PlinkKinS(const string &file_bed, vector<int> &indicator_snp,
              const int k_mode, const int display_pace,
              gsl_matrix *matrix_kin, gsl_matrix *matrix_S) 
{
  debug_msg("entered");
  ifstream infile(file_bed.c_str(), ios::binary);
  if (!infile) {
    cout << "error reading bed file:" << file_bed << endl;
    return false;
  }

  char ch[1];
  bitset<8> b;

  size_t n_miss, ci_total;
  double d, geno_mean, geno_var, scale, scale0;

  size_t ni_total = matrix_kin->size1;
  gsl_vector *geno = gsl_vector_safe_alloc(ni_total);

  size_t ns_test = 0;
  int n_bit;

  
  printf(" snp.size = %zu \n", indicator_snp.size());
  
  
  vector<gsl_vector*> genotype_vectors;
  
  
  if (ni_total % 4 == 0) {
    n_bit = ni_total / 4;
  } else {
    n_bit = ni_total / 4 + 1;
  }

  
  for (int i = 0; i < 3; ++i) {
    infile.read(ch, 1);
    b = ch[0];
  }
  
  scale = 0.0;
  
  for (size_t t = 0; t < indicator_snp.size(); ++t) {
    if (t % display_pace == 0 || t == (indicator_snp.size() - 1)) {
      ProgressBar("Reading SNPs", t, indicator_snp.size() - 1);
    }
    if (indicator_snp[t] == 0) {
      continue;
    }

    infile.seekg(t * n_bit + 3);

    geno_mean = 0.0;
    scale0 = 0.0;
    n_miss = 0;
    ci_total = 0;
    geno_var = 0.0;
    for (int i = 0; i < n_bit; ++i) {
      infile.read(ch, 1);
      b = ch[0];

      for (size_t j = 0; j < 4; ++j) {
        if ((i == (n_bit - 1)) && ci_total == ni_total) {
          break;
        }

        if (b[2 * j] == 0) {
          if (b[2 * j + 1] == 0) {
            gsl_vector_set(geno, ci_total, 2.0);
            geno_mean += 2.0;
            geno_var += 4.0;
          } else {
            gsl_vector_set(geno, ci_total, 1.0);
            geno_mean += 1.0;
            geno_var += 1.0;
          }
        } else {
          if (b[2 * j + 1] == 1) {
            gsl_vector_set(geno, ci_total, 0.0);
          } else {
            gsl_vector_set(geno, ci_total, -9.0);
            n_miss++;
          }
        }

        ci_total++;
      }
    }

    geno_mean /= (double)(ni_total - n_miss);
    geno_var += geno_mean * geno_mean * (double)n_miss;
    geno_var /= (double)ni_total;
    geno_var -= geno_mean * geno_mean;

    for (size_t i = 0; i < ni_total; ++i) {
      d = gsl_vector_get(geno, i);
      if (d == -9.0) {
        gsl_vector_set(geno, i, geno_mean);
      }
    }

    gsl_vector_add_constant(geno, -1.0 * geno_mean);

    if (k_mode == 2 && geno_var != 0) {
      gsl_vector_scale(geno, 1.0 / sqrt(geno_var));
    }

    gsl_vector *geno_copy = gsl_vector_safe_alloc(ni_total);
    gsl_vector_memcpy(geno_copy, geno);
    genotype_vectors.push_back(geno_copy);
    
    scale0 = geno_mean / 2.0 * (1.0 - geno_mean / 2.0);
    scale += scale0;
    
    ns_test++;
  }

  
  const size_t msize = ns_test;  
  gsl_matrix *Xlarge = gsl_matrix_safe_alloc(ni_total, msize);
  
 

  
  for (size_t i = 0; i < ns_test; ++i) {
    gsl_vector_view col = gsl_matrix_column(Xlarge, i);
    gsl_vector_memcpy(&col.vector, genotype_vectors[i]);
    
   
    gsl_vector_free(genotype_vectors[i]);
  }
  
  
  genotype_vectors.clear();

 
  gsl_matrix_set_zero(matrix_kin);
  fast_eigen_dgemm("N", "T", 1.0, Xlarge, Xlarge, 1.0, matrix_kin);

  cout << endl;
  
  printf("n_snp = %zu \n", indicator_snp.size());
  printf("ns_test = %zu \n", ns_test);
  printf("scale = %.8f \n", scale);
  gsl_matrix_scale(matrix_kin,  1.0 / (2.0 * scale) );

  
  for (size_t i = 0; i < ni_total; ++i) {
    for (size_t j = 0; j < i; ++j) {
      d = gsl_matrix_get(matrix_kin, j, i);
      gsl_matrix_set(matrix_kin, i, j, d);
    }
  }

  
  gsl_matrix_memcpy(matrix_S, Xlarge);
  
   
  gsl_vector_free(geno);
  gsl_matrix_free(Xlarge);

  infile.close();
  infile.clear();

  return true;
}



bool ReadFile_bed(const string &file_bed, vector<int> &indicator_idv,
                  vector<int> &indicator_snp, gsl_matrix *UtX, gsl_matrix *K,
                  const bool calc_K) 
{
  debug_msg("entered");
  ifstream infile(file_bed.c_str(), ios::binary);
  if (!infile) {
    cout << "error reading bed file:" << file_bed << endl;
    return false;
  }

  char ch[1];
  bitset<8> b;

  size_t ni_total = indicator_idv.size();
  size_t ns_total = indicator_snp.size();
  size_t ni_test = UtX->size1;
  size_t ns_test = UtX->size2;
  int n_bit;

  if (ni_total % 4 == 0) {
    n_bit = ni_total / 4;
  } else {
    n_bit = ni_total / 4 + 1;
  }

  
  for (int i = 0; i < 3; ++i) {
    infile.read(ch, 1);
    b = ch[0];
  }

  if (calc_K == true) {
    gsl_matrix_set_zero(K);
  }

  gsl_vector *genotype = gsl_vector_safe_alloc(UtX->size1);

  double geno, geno_mean;
  size_t n_miss;
  size_t c_idv = 0, c_snp = 0, c = 0;

  for (size_t t = 0; t < ns_total; ++t) {
    if (indicator_snp[t] == 0) {
      continue;
    }

    infile.seekg(t * n_bit + 3);

    c_idv = 0;
    geno_mean = 0.0;
    n_miss = 0;
    c = 0;
    for (int i = 0; i < n_bit; ++i) {
      infile.read(ch, 1);
      b = ch[0];

      for (size_t j = 0; j < 4; ++j) {
        if ((i == (n_bit - 1)) && c == ni_total) {
          break;
        }
        if (indicator_idv[c] == 0) {
          c++;
          continue;
        }
        c++;

        if (b[2 * j] == 0) {
          if (b[2 * j + 1] == 0) {
            gsl_vector_set(genotype, c_idv, 2.0);
            geno_mean += 2.0;
          } else {
            gsl_vector_set(genotype, c_idv, 1.0);
            geno_mean += 1.0;
          }
        } else {
          if (b[2 * j + 1] == 1) {
            gsl_vector_set(genotype, c_idv, 0.0);
            geno_mean += 0.0;
          } else {
            gsl_vector_set(genotype, c_idv, -9.0);
            n_miss++;
          }
        }
        c_idv++;
      }
    }

    geno_mean /= (double)(ni_test - n_miss);

    for (size_t i = 0; i < genotype->size; ++i) {
      geno = gsl_vector_get(genotype, i);
      if (geno == -9) {
        geno = 0;
      } else {
        geno -= geno_mean;
      }

      gsl_vector_set(genotype, i, geno);
      gsl_matrix_set(UtX, i, c_snp, geno);
    }

    if (calc_K == true) {
      gsl_blas_dsyr(CblasUpper, 1.0, genotype, K);
    }

    c_snp++;
  }

  if (calc_K == true) {
    gsl_matrix_scale(K, 1.0 / (double)ns_test);

    for (size_t i = 0; i < genotype->size; ++i) {
      for (size_t j = 0; j < i; ++j) {
        geno = gsl_matrix_get(K, j, i);
        gsl_matrix_set(K, i, j, geno);
      }
    }
  }

  gsl_vector_free(genotype);
  infile.clear();
  infile.close();

  return true;
}

bool ReadFile_bed(const string &file_bed, vector<int> &indicator_idv,
                  vector<int> &indicator_snp, vector<vector<unsigned char>> &Xt,
                  gsl_matrix *K, const bool calc_K, const size_t ni_test,
                  const size_t ns_test) 
{
  debug_msg("entered");
  ifstream infile(file_bed.c_str(), ios::binary);
  if (!infile) {
    cout << "error reading bed file:" << file_bed << endl;
    return false;
  }

  Xt.clear();
  vector<unsigned char> Xt_row;
  for (size_t i = 0; i < ni_test; i++) {
    Xt_row.push_back(0);
  }

  char ch[1];
  bitset<8> b;

  size_t ni_total = indicator_idv.size();
  size_t ns_total = indicator_snp.size();
  int n_bit;

  if (ni_total % 4 == 0) {
    n_bit = ni_total / 4;
  } else {
    n_bit = ni_total / 4 + 1;
  }

  for (int i = 0; i < 3; ++i) {
    infile.read(ch, 1);
    b = ch[0];
  }

  if (calc_K == true) {
    gsl_matrix_set_zero(K);
  }

  gsl_vector *genotype = gsl_vector_safe_alloc(ni_test);

  double geno, geno_mean;
  size_t n_miss;
  size_t c_idv = 0, c_snp = 0, c = 0;

  for (size_t t = 0; t < ns_total; ++t) {
    if (indicator_snp[t] == 0) {
      continue;
    }

    infile.seekg(t * n_bit + 3);

    c_idv = 0;
    geno_mean = 0.0;
    n_miss = 0;
    c = 0;
    for (int i = 0; i < n_bit; ++i) {
      infile.read(ch, 1);
      b = ch[0];

      for (size_t j = 0; j < 4; ++j) {
        if ((i == (n_bit - 1)) && c == ni_total) {
          break;
        }
        if (indicator_idv[c] == 0) {
          c++;
          continue;
        }
        c++;

        if (b[2 * j] == 0) {
          if (b[2 * j + 1] == 0) {
            gsl_vector_set(genotype, c_idv, 2.0);
            geno_mean += 2.0;
          } else {
            gsl_vector_set(genotype, c_idv, 1.0);
            geno_mean += 1.0;
          }
        } else {
          if (b[2 * j + 1] == 1) {
            gsl_vector_set(genotype, c_idv, 0.0);
            geno_mean += 0.0;
          } else {
            gsl_vector_set(genotype, c_idv, -9.0);
            n_miss++;
          }
        }
        c_idv++;
      }
    }

    geno_mean /= (double)(ni_test - n_miss);

    for (size_t i = 0; i < genotype->size; ++i) {
      geno = gsl_vector_get(genotype, i);
      if (geno == -9) {
        geno = geno_mean;
      }

      Xt_row[i] = Double02ToUchar(geno);

      geno -= geno_mean;

      gsl_vector_set(genotype, i, geno);
    }
    Xt.push_back(Xt_row);

    if (calc_K == true) {
      gsl_blas_dsyr(CblasUpper, 1.0, genotype, K);
    }

    c_snp++;
  }

  if (calc_K == true) {
    gsl_matrix_scale(K, 1.0 / (double)ns_test);

    for (size_t i = 0; i < genotype->size; ++i) {
      for (size_t j = 0; j < i; ++j) {
        geno = gsl_matrix_get(K, j, i);
        gsl_matrix_set(K, i, j, geno);
      }
    }
  }

  gsl_vector_free(genotype);
  infile.clear();
  infile.close();

  return true;
}

bool CountFileLines(const string &file_input, size_t &n_lines) {
  debug_msg("entered");
  igzstream infile(file_input.c_str(), igzstream::in);
  if (!infile) {
    cout << "error! fail to open file: " << file_input << endl;
    return false;
  }

  n_lines = count(istreambuf_iterator<char>(infile),
                  istreambuf_iterator<char>(), '\n');
  infile.seekg(0, ios::beg);

  return true;
}






bool ReadHeader_io(const string &line, HEADER &header) {
  debug_msg("entered");
  string rs_ptr[] = {"rs",    "RS",    "snp",  "SNP",  "snps",      "SNPS",
                     "snpid", "SNPID", "rsid", "RSID", "MarkerName"};
  set<string> rs_set(rs_ptr, rs_ptr + 11); 
  
  string chr_ptr[] = {"chr", "CHR"};
  set<string> chr_set(chr_ptr, chr_ptr + 2);
  
  string pos_ptr[] = {
      "ps", "PS", "pos", "POS", "base_position", "BASE_POSITION", "bp", "BP"};
  set<string> pos_set(pos_ptr, pos_ptr + 8);
  
  string cm_ptr[] = {"cm", "CM"};
  set<string> cm_set(cm_ptr, cm_ptr + 2);

  string a1_ptr[] = {"a1", "A1", "allele1", "ALLELE1", "Allele1", "INC_ALLELE"};
  set<string> a1_set(a1_ptr, a1_ptr + 5);

  string a0_ptr[] = {"a0", "A0",      "allele0", "ALLELE0", "Allele0",   "a2",
                     "A2", "allele2", "ALLELE2", "Allele2", "DEC_ALLELE"};
  set<string> a0_set(a0_ptr, a0_ptr + 10);

  string z_ptr[] = {"z", "Z", "z_score", "Z_SCORE", "zscore", "ZSCORE"};
  set<string> z_set(z_ptr, z_ptr + 6);

  string beta_ptr[] = {"beta", "BETA", "b", "B"};
  set<string> beta_set(beta_ptr, beta_ptr + 4);
  
  string chisq_ptr[] = {"chisq", "CHISQ", "chisquare", "CHISQUARE"};
  set<string> chisq_set(chisq_ptr, chisq_ptr + 4);
  
  string p_ptr[] = {"p", "P", "pvalue", "PVALUE", "p-value", "P-VALUE"};
  set<string> p_set(p_ptr, p_ptr + 6);

  string n_ptr[] = {"n", "N", "ntotal", "NTOTAL", "n_total", "N_TOTAL"};
  set<string> n_set(n_ptr, n_ptr + 6);

  string nmis_ptr[] = {"nmis", "NMIS", "n_mis", "N_MIS", "n_miss", "N_MISS"};
  set<string> nmis_set(nmis_ptr, nmis_ptr + 6);
  
  string nobs_ptr[] = {"nobs", "NOBS", "n_obs", "N_OBS"};
  set<string> nobs_set(nobs_ptr, nobs_ptr + 4);
  

  string af_ptr[] = {"af",
                     "AF",
                     "maf",
                     "MAF",
                     "f",
                     "F",
                     "allele_freq",
                     "ALLELE_FREQ",
                     "allele_frequency",
                     "ALLELE_FREQUENCY",
                     "Freq.Allele1.HapMapCEU",
                     "FreqAllele1HapMapCEU",
                     "Freq1.Hapmap"};
  set<string> af_set(af_ptr, af_ptr + 13);


  header.rs_col = 0;
  header.chr_col = 0;
  header.pos_col = 0;
  header.cm_col = 0;
  header.a1_col = 0;
  header.a0_col = 0;
  header.z_col = 0;
  header.beta_col = 0;
  header.chisq_col = 0;
  header.p_col = 0;
  header.n_col = 0;
  header.nmis_col = 0;
  header.nobs_col = 0;
  header.af_col = 0;

  char *ch_ptr;
  string type;
  size_t n_error = 0;

  ch_ptr = strtok((char *)line.c_str(), " , \t");
  while (ch_ptr != NULL) {
    type = ch_ptr;
    if (rs_set.count(type) != 0) {
      if (header.rs_col == 0) {
        header.rs_col = header.coln + 1;
      } else {
        cout << "error! more than two rs columns in the file." << endl;
        n_error++;
      }
    } else if (chr_set.count(type) != 0) {
      if (header.chr_col == 0) {
        header.chr_col = header.coln + 1;
      } else {
        cout << "error! more than two chr columns in the file." << endl;
        n_error++;
      }
    } else if (pos_set.count(type) != 0) {
      if (header.pos_col == 0) {
        header.pos_col = header.coln + 1;
      } else {
        cout << "error! more than two pos columns in the file." << endl;
        n_error++;
      }
    } else if (cm_set.count(type) != 0) {
      if (header.cm_col == 0) {
        header.cm_col = header.coln + 1;
      } else {
        cout << "error! more than two cm columns in the file." << endl;
        n_error++;
      }
    } else if (a1_set.count(type) != 0) {
      if (header.a1_col == 0) {
        header.a1_col = header.coln + 1;
      } else {
        cout << "error! more than two allele1 columns in the file." << endl;
        n_error++;
      }
    } else if (a0_set.count(type) != 0) {
      if (header.a0_col == 0) {
        header.a0_col = header.coln + 1;
      } else {
        cout << "error! more than two allele0 columns in the file." << endl;
        n_error++;
      }
    } else if (z_set.count(type) != 0) {
      if (header.z_col == 0) {
        header.z_col = header.coln + 1;
      } else {
        cout << "error! more than two z columns in the file." << endl;
        n_error++;
      }
    } else if (beta_set.count(type) != 0) {
      if (header.beta_col == 0) {
        header.beta_col = header.coln + 1;
      } else {
        cout << "error! more than two beta columns in the file." << endl;
        n_error++;
      }
    } else if (chisq_set.count(type) != 0) {
      if (header.chisq_col == 0) {
        header.chisq_col = header.coln + 1;
      } else {
        cout << "error! more than two z columns in the file." << endl;
        n_error++;
      }
    } else if (p_set.count(type) != 0) {
      if (header.p_col == 0) {
        header.p_col = header.coln + 1;
      } else {
        cout << "error! more than two p columns in the file." << endl;
        n_error++;
      }
    } else if (n_set.count(type) != 0) {
      if (header.n_col == 0) {
        header.n_col = header.coln + 1;
      } else {
        cout << "error! more than two n_total columns in the file." << endl;
        n_error++;
      }
    } else if (nmis_set.count(type) != 0) {
      if (header.nmis_col == 0) {
        header.nmis_col = header.coln + 1;
      } else {
        cout << "error! more than two n_mis columns in the file." << endl;
        n_error++;
      }
    } else if (nobs_set.count(type) != 0) {
      if (header.nobs_col == 0) {
        header.nobs_col = header.coln + 1;
      } else {
        cout << "error! more than two n_obs columns in the file." << endl;
        n_error++;
      }
    } else if (af_set.count(type) != 0) {
      if (header.af_col == 0) {
        header.af_col = header.coln + 1;
      } else {
        cout << "error! more than two af columns in the file." << endl;
        n_error++;
      }
    } else {
    }

    ch_ptr = strtok(NULL, " , \t");
    header.coln++;
  }

  if (header.rs_col == 0) {
    if (header.chr_col != 0 && header.pos_col != 0) {
      cout << "missing an rs column. rs id will be replaced by chr:pos" << endl;
    } else {
      cout << "error! missing an rs column." << endl;
      n_error++;
    }
  }

  if (n_error == 0) {
    return true;
  } else {
    return false;
  }
}


void ReadFile_vector(const string &file_vec, gsl_vector *vec) {
  debug_msg("entered");
  igzstream infile(file_vec.c_str(), igzstream::in);
  if (!infile) {
    cout << "error! fail to open vector file: " << file_vec << endl;
    return;
  }

  string line;
  char *ch_ptr;

  for (size_t i = 0; i < vec->size; i++) {
    safeGetline(infile, line).eof();
    ch_ptr = strtok_safe((char *)line.c_str(), " , \t");
    gsl_vector_set(vec, i, atof(ch_ptr));
  }

  infile.clear();
  infile.close();

  return;
}

void ReadFile_matrix(const string &file_mat, gsl_matrix *mat) {
  debug_msg("entered");
  igzstream infile(file_mat.c_str(), igzstream::in);
  if (!infile) {
    cout << "error! fail to open matrix file: " << file_mat << endl;
    return;
  }

  string line;
  char *ch_ptr;

  for (size_t i = 0; i < mat->size1; i++) {
    safeGetline(infile, line).eof();
    ch_ptr = strtok_safe((char *)line.c_str(), " , \t");
    for (size_t j = 0; j < mat->size2; j++) {
      enforce(ch_ptr);
      gsl_matrix_set(mat, i, j, atof(ch_ptr));
      ch_ptr = strtok(NULL, " , \t");
    }
  }

  infile.clear();
  infile.close();

  return;
}



