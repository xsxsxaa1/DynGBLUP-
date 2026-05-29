/*
    Dynamic Genomic Best linear Unbiased Prediction  (DynGBLUP)
    Copyright © 2026, Meixia Ye
    
*/
#define DYNGBLUP_VERSION "1.0.0"
#define DYNGBLUP_DATE    "2026-05-20"
#define DYNGBLUP_YEAR    "2026"

#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <gsl/gsl_statistics.h>
#ifdef OPENBLAS
#pragma message "Compiling with OPENBLAS"
extern "C" {
 
  int openblas_get_num_threads(void);
  int openblas_get_parallel(void);
  char* openblas_get_config(void);
  char* openblas_get_corename(void);
}
#endif

#include "gsl/gsl_blas.h"
#include "gsl/gsl_cdf.h"
#include "gsl/gsl_eigen.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_version.h"


#include "dyngblup.h"
#include "io.h"
#include "lapack.h"
#include "mathfunc.h"
#include "iterate.h" 
#include "debug.h"
#include "version.h"

using namespace std;

DYNGBLUP::DYNGBLUP(void) : version(DYNGBLUP_VERSION), date(DYNGBLUP_DATE), year(DYNGBLUP_YEAR) {}

void DYNGBLUP_gsl_error_handler (const char * reason,
                              const char * file,
                              int line, int gsl_errno) 
{
  cerr << "GSL ERROR: " << reason << " in " << file
       << " at line " << line << " errno " << gsl_errno <<endl;
  exit(22);
}

#if defined(OPENBLAS) && !defined(OPENBLAS_LEGACY)
#include <openblas_config.h>
#endif

void DYNGBLUP::PrintHeader(void) {

  cout <<
    "DYNGBLUP " << version << " (" << date << ") by Meixia Ye and team (C) 2024-" << year << endl;
  return;
}

void DYNGBLUP::PrintLicense(void) {
  cout << endl;
  cout << "The Software Is Distributed Under GNU General Public "
       << "License, But May Also Require The Following Notifications." << endl;
  cout << endl;

  cout << "Copyright (c) 2026 Beijing Forestry University. "
       << "All rights reserved." << endl;
  cout << endl;

  cout << "$COPYRIGHT$" << endl;
  cout << "Additional copyrights may follow" << endl;
  cout << "$HEADER$" << endl;
  cout << "Redistribution and use in source and binary forms, with or "
       << "without modification, are permitted provided that the following "
       << " conditions are met:" << endl;
  cout << "- Redistributions of source code must retain the above "
       << "copyright notice, this list of conditions and the following "
       << "disclaimer." << endl;
  cout << "- Redistributions in binary form must reproduce the above "
       << "copyright notice, this list of conditions and the following "
       << "disclaimer listed in this license in the documentation and/or "
       << "other materials provided with the distribution." << endl;
  cout << "- Neither the name of the copyright holders nor the names "
       << "of its contributors may be used to endorse or promote products "
       << "derived from this software without specific prior written "
       << "permission." << endl;
  cout << "The copyright holders provide no reassurances that the "
       << "source code provided does not infringe any patent, copyright, "
       << "or any other "
       << "intellectual property rights of third parties. "
       << "The copyright holders disclaim any liability to any recipient "
       << "for claims brought against "
       << "recipient by any third party for infringement of that parties "
       << "intellectual property rights. " << endl;
  cout << "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND "
       << "CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, "
       << "INCLUDING, BUT NOT "
       << "LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND "
       << "FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT "
       << "SHALL THE COPYRIGHT "
       << "OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, "
       << "INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES "
       << "(INCLUDING, BUT NOT "
       << "LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; "
       << "LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) "
       << "HOWEVER CAUSED AND ON ANY "
       << "THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, "
       << "OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY "
       << "OUT OF THE USE "
       << "OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF "
       << "SUCH DAMAGE." << endl;
  cout << endl;

  return;
}

void DYNGBLUP::PrintHelp(size_t option) {

  if (option == 0) {
    cout << endl;
    cout << " type ./dyngblup -h [num] for detailed help" << endl;
    cout << " options: " << endl;
    cout << "  1: quick guide" << endl;
    cout << "  2: file I/O related" << endl;
    cout << "  3: SNP QC" << endl;
    cout << "  4: calculate relatedness matrix" << endl;
    cout << "  5: perform eigen decomposition" << endl;
    
    cout << "  6: fit a multivariate linear mixed model" << endl;
    
    cout << "  7: note" << endl;
    cout << "  8: debug options" << endl;
    cout << endl;
  }

  if (option == 1) {
    cout << " QUICK GUIDE" << endl;

    cout << " to generate a relatedness matrix: " << endl;
    cout << "         ./dyngblup -bfile [prefix] -gk [num] -o [prefix]" << endl;
    cout << "         ./dyngblup -g [filename] -p [filename] -gk [num] -o [prefix]"
         << endl;
    
    cout << " to perform eigen decomposition of the relatedness matrix: "
         << endl;
    cout << "         ./dyngblup -bfile [prefix] -k [filename] -eigen -o [prefix]"
         << endl;
    cout << "         ./dyngblup -g [filename] -p [filename] -k [filename] -eigen "
            "-o [prefix]"
         << endl;
    
    cout << " to fit a multivariate linear mixed model using the common mode: " << endl;
    cout << "         ./dyngblup -bfile [prefix] -c [filename] -k [filename] -lmm [num] -n "
            "[pheno cols...] -cu [num] -t [num] -o [prefix]"
         << endl;
    cout << "         ./dyngblup -g [filename] -p [filename] -a [filename] -c [filename] -k "
            "[filename] -lmm [num] -n [pheno cols...] -cu [num] -o [prefix]"
         << endl;

    cout << "         ./dyngblup -bfile [prefix] -c [filename] -k [filename] -lmm [num] -n "
            "[pheno cols...] -cu [num] -o [prefix]"
         << endl;
    cout << "         ./dyngblup -g [filename] -p [filename] -a [filename] -c [filename] -k "
            "[filename] -lmm [num] -n [pheno cols...] -cu [num] "
            "[num] -o [prefix]"
         << endl;

    cout << endl;
  }


 if (option == 2) {
    cout << " FILE I/O RELATED OPTIONS" << endl;
    cout << " -bfile    [prefix]       "
         << " specify input PLINK binary ped file prefix." << endl;
    cout << "          requires: *.fam, *.bim and *.bed files" << endl;
    cout << "          missing value: -9" << endl;


    cout << " -k        [filename]     "
         << " specify input kinship/relatedness matrix file name" << endl;
    cout << " -u        [filename]     "
         << " specify input file containing the eigen vectors of the "
            "kinship/relatedness matrix"
         << endl;
    cout << " -d        [filename]     "
         << " specify input file containing the eigen values of the "
            "kinship/relatedness matrix"
         << endl;


    cout << " -c        [filename]     "
         << " specify input covariates file name (optional)" << endl;
    
    cout << " -km       [num]          "
         << " specify input kinship/relatedness file type (default 1)." << endl;
    cout << "          options: 1: \"n by n matrix\" format" << endl;
    cout << "                   2: \"id  id  value\" format" << endl;
    
    cout << " -n        [num]          "
         << " specify phenotype column in the phenotype/*.fam file (optional; "
            "default 1)"
         << endl;
    cout << " -pace     [num]          "
         << " specify terminal display update pace (default 1,000 SNPs or "
            "1,000 iterations)."
         << endl;
    cout << " -cu       [num]          "
         << " specify curve type (default 1 for logistic curve, "
            "2 for , 3 for, 4 for )."
         << endl;
		 
   

    cout << "          format: rs#1" << endl;
    cout << "                  rs#2" << endl;
    cout << "                  ..." << endl;
    cout << "          missing value: NA" << endl;

    cout << " -outdir   [path]         "
         << " specify output directory path (default \"./output/\")" << endl;
    cout << " -o        [prefix]       "
         << " specify output file prefix (default \"result\")" << endl;
    cout << "          output: prefix.cXX.txt or prefix.sXX.txt from "
            "kinship/relatedness matrix estimation"
         << endl;
    cout << "          output: prefix.assoc.txt and prefix.log.txt form "
            "association tests"
         << endl;
    cout << endl;
  }

  if (option == 3) {
    cout << " SNP QC OPTIONS" << endl;
    cout << " -miss     [num]          "
         << " specify missingness threshold (default 0.05)" << endl;
    cout << " -maf      [num]          "
         << " specify minor allele frequency threshold (default 0.01)" << endl;
    cout << " -hwe      [num]          "
         << " specify HWE test p value threshold (default 0; no test)" << endl;
    cout << " -r2       [num]          "
         << " specify r-squared threshold (default 0.9999)" << endl;
    cout << " -notsnp                  "
         << " minor allele frequency cutoff is not used" << endl;
    cout << endl;
  }

  if (option == 4) {
    cout << " RELATEDNESS MATRIX (K) CALCULATION OPTIONS" << endl;
    cout << " -a        [filename]     "
         << " specify input BIMBAM SNP annotation file name (LOCO only)"
         << endl;
    cout << " -gk       [num]          "
         << " specify which type of kinship/relatedness matrix to generate "
            "(default 1)"
         << endl;
    cout << "          options: 1: centered XX^T/p" << endl;
    cout << "                   2: standardized XX^T/p" << endl;
    cout << "          note: non-polymorphic SNPs are excluded " << endl;
    cout << endl;
  }

  if (option == 5) {
    cout << " EIGEN-DECOMPOSITION OPTIONS" << endl;
    cout << " -eigen                   "
         << " specify to perform eigen decomposition of the loaded relatedness "
            "matrix"
         << endl;
    cout << endl;
  }

  if (option == 6) {
    cout << " MULTIVARIATE LINEAR MIXED MODEL OPTIONS" << endl;
    cout << " -n [pheno cols...] - range of phenotypes" << endl;
    cout << " -c        [filename]     "
         << " specify input covariates file name (optional)" << endl;
    cout << " -cu       [num]          "
         << " specify curve type "
         << endl;
    cout << " -t       [num]          "
         << " specify threads number." << endl;

    cout << " -snps     [filename]     "
         << " specify input snps file name to only analyze a certain set of snps"
         << endl;     

    cout << " -maxi				   "
         << " specify the maximum number of iterations for the AI method in "
            "the null (default 100)"
         << endl;
    cout << " -cp            "
         << " specify the precision for the AI method in the null (default "
            "0.0001)"
         << endl;
    cout << " -cg            "
         << " specify the precision for the AI method in the null (default "
            "0.0001)"
         << endl;
    cout << " -nrf           "
         << " specify the nf_size for the fixed effect, used when LOP curve was"
         << " selected (default 3)"
         << endl;
    cout << " -nr1				   "
         << " specify the nr1_size for the genetic random effect"
         << "(default 3)"
         << endl;
    cout << " -nr2				   "
         << " specify the nr2_size for the general random effect"
         << "(default 3)"
         << endl;
    
    cout << endl;
  }

  if (option == 7) {
    cout << " NOTE" << endl;
    cout << " 1. Only individuals with non-missing phenotoypes and covariates "
            "will be analyzed."
         << endl;
    cout << " 2. Missing genotoypes will be repalced with the mean genotype of "
            "that SNP."
         << endl;
    cout << " 3. For lmm analysis, memory should be large enough to hold the "
            "relatedness matrix and to perform eigen decomposition."
         << endl;
    cout << " 4. For multivariate lmm analysis, use a large -pnr for each snp "
            "will increase computation time dramatically."
         << endl;
    
    cout << endl;
  }

  if (option == 8) {
    cout << " DEBUG OPTIONS" << endl;
    cout << " -no-check                disable checks" << endl;
    cout << " -strict                  strict mode will stop when there is a problem" << endl;
    cout << " -silence                 silent terminal display" << endl;
    cout << " -debug                   debug output" << endl;
    cout << " -nind       [num]        read up to num individuals" << endl;
    cout << " -issue      [num]        enable tests relevant to issue tracker" << endl;
    cout << " -legacy                  run dyngblup in legacy mode" << endl;
    cout << endl;
  }

  return;
}



void DYNGBLUP::Assign(int argc, char **argv, PARAM &cPar) {
  string str;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-bfile") == 0 || strcmp(argv[i], "--bfile") == 0 ||
        strcmp(argv[i], "-b") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.file_bfile = str;
    } else if (strcmp(argv[i], "-g") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.file_geno = str;
    } else if (strcmp(argv[i], "-p") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.file_pheno = str;
    } else if (strcmp(argv[i], "-a") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.file_anno = str;
    } else if (strcmp(argv[i], "-k") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.file_kin = str;
    } else if (strcmp(argv[i], "-s") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.file_S = str;
    } else if (strcmp(argv[i], "-u") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.file_ku = str;
    } else if (strcmp(argv[i], "-d") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.file_kd = str;
    } else if (strcmp(argv[i], "-c") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.file_cvt = str;
    } else if (strcmp(argv[i], "-km") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.k_mode = atoi(str.c_str());
    } else if (strcmp(argv[i], "-n") == 0) { 
      (cPar.p_column).clear();
      while (argv[i + 1] != NULL && argv[i + 1][0] != '-') {
        ++i;
        str.clear();
        str.assign(argv[i]);
        (cPar.p_column).push_back(atoi(str.c_str()));
      }
    } else if (strcmp(argv[i], "-cu") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.curve_type = atof(str.c_str());
    } else if (strcmp(argv[i], "-pace") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.d_pace = atoi(str.c_str());
    } else if (strcmp(argv[i], "-outdir") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.path_out = str;
    } else if (strcmp(argv[i], "-o") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.file_out = str;
    } else if (strcmp(argv[i], "-miss") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.miss_level = atof(str.c_str());
    } else if (strcmp(argv[i], "-maf") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      if (cPar.maf_level != -1) {
        cPar.maf_level = atof(str.c_str());
      }
    } else if (strcmp(argv[i], "-hwe") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.hwe_level = atof(str.c_str());
    } else if (strcmp(argv[i], "-r2") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.r2_level = atof(str.c_str());
    } else if (strcmp(argv[i], "-nrf") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.nrf = atof(str.c_str());
    } else if (strcmp(argv[i], "-nr1") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.nr1_size = atof(str.c_str());
    } else if (strcmp(argv[i], "-nr2") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.nr2_size = atof(str.c_str());
    } else if (strcmp(argv[i], "-notsnp") == 0) {
      cPar.maf_level = -1;
    } else if (strcmp(argv[i], "-gk") == 0) {
      if (cPar.a_mode != 0) {
        cPar.error = true;
        cout << "error! only one of -gk -eigen -lmm options is allowed."
             << endl;
        break;
      }
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        cPar.a_mode = 21;
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.a_mode = 20 + atoi(str.c_str());
    } else if (strcmp(argv[i], "-eigen") == 0) {
      if (cPar.a_mode != 0) {
        cPar.error = true;
        cout << "error! only one of -gk -eigen -lmm options is allowed."
             << endl;
        break;
      }
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        cPar.a_mode = 31;
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.a_mode = 30 + atoi(str.c_str());
    } else if (strcmp(argv[i], "-lmm") == 0) {
      if (cPar.a_mode != 0) {
        cPar.error = true;
        cout << "error! only one of -gk -eigen -lmm options is allowed."
             << endl;
        break;
      }
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        cPar.a_mode = 1;
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.a_mode = atoi(str.c_str());
    } else if (strcmp(argv[i], "-maxi") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.max_iter = atoi(str.c_str());
    } else if (strcmp(argv[i], "-cp") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.cc_par = atof(str.c_str());
    } else if (strcmp(argv[i], "-cg") == 0) {
      if (argv[i + 1] == NULL || argv[i + 1][0] == '-') {
        continue;
      }
      ++i;
      str.clear();
      str.assign(argv[i]);
      cPar.cc_gra = atof(str.c_str());
    } else if (strcmp(argv[i], "-debug") == 0) {
      debug_set_debug_mode(true);
    } else if (strcmp(argv[i], "-no-check") == 0) {
      debug_set_no_check_mode(true);
    } else if (strcmp(argv[i], "-strict") == 0) {
      debug_set_strict_mode(true);
    } else if (strcmp(argv[i], "-legacy") == 0) {
      debug_set_legacy_mode(true);
      warning_msg("you are running in legacy mode - support may drop in future versions of DYNGBLUP");
    } else {
      cout << "error! unrecognized option: " << argv[i] << endl;
      cPar.error = true;
      continue;
    }
  }

  return;
}

void DYNGBLUP::BatchRun(PARAM &cPar) {
  clock_t time_begin, time_start;
  time_begin = clock();

  cout << "Reading Files ... " << endl;
  cPar.ReadFiles();
  if (cPar.error == true) {
    cout << "error! fail to read files. " << endl;
    return;
  }
  cPar.CheckData();
  if (cPar.error == true) {
    cout << "error! fail to check data. " << endl;
    return;
  }

  if (cPar.a_mode == 21 || cPar.a_mode == 22) {
    cout << "Calculating Relatedness Matrix ... " << endl;

    gsl_matrix *G = gsl_matrix_safe_alloc(cPar.ni_total, cPar.ni_total);
    enforce_msg(G, "allocate G"); 
    gsl_matrix *S = gsl_matrix_safe_alloc(cPar.ni_total, cPar.ns_test);

    gsl_matrix_set_zero( G );
    gsl_matrix_set_zero( S );

    time_start = clock();
    cPar.CalcKinS(G,S);

    cPar.time_G = (clock() - time_start) / (double(CLOCKS_PER_SEC) * 60.0);
    if (cPar.error == true) {
      cout << "error! fail to calculate relatedness matrix. " << endl;
      return;
    }
    
    validate_K(G);

    if (cPar.a_mode == 21) {
      cPar.WriteMatrix(G, "cXX");
      cPar.WriteMatrix(S, "cSS");
    } else {
      cPar.WriteMatrix(G, "sXX");
      cPar.WriteMatrix(S, "sSS");
    }

    gsl_matrix_safe_free(G);
  }


 
  if (cPar.a_mode == 1 || cPar.a_mode == 2 || cPar.a_mode == 3 ||
      cPar.a_mode == 4 || cPar.a_mode == 5 ||
      cPar.a_mode == 31) 
  { 
    gsl_matrix *Y = gsl_matrix_safe_alloc(cPar.ni_test, cPar.n_ph);
    enforce_msg(Y, "allocate Y"); 
    gsl_matrix *W = gsl_matrix_safe_alloc(Y->size1, cPar.n_cvt);
    gsl_matrix *G = gsl_matrix_safe_alloc(Y->size1, Y->size1);
    gsl_matrix *S = gsl_matrix_safe_alloc(Y->size1,  cPar.ns_test);
    gsl_matrix *U = gsl_matrix_safe_alloc(Y->size1, Y->size1);
    gsl_vector *eval = gsl_vector_calloc(Y->size1);



    gsl_matrix_set_zero( G );
    gsl_matrix_set_zero( S );

    cPar.CopyCvtPhen(W, Y, 0);
   

   
    if (!(cPar.file_kin).empty()) {
      ReadFile_kin(cPar.file_kin, cPar.indicator_idv, cPar.mapID2num,
                   cPar.k_mode, cPar.error, G);

      if (cPar.error == true) {
        cout << "error! fail to read kinship/relatedness file. " << endl;
        return;
      }

      validate_K(G);


      cout << "Start Eigen-Decomposition..." << endl;
      time_start = clock();
     
     
     gsl_matrix *G2 = gsl_matrix_safe_alloc(Y->size1, Y->size1);
     gsl_matrix_set_zero( G2);
     gsl_matrix_memcpy(G2,G);
    

      if (cPar.a_mode == 31) {        
        cPar.trace_G = EigenDecomp_Zeroed(G2, U, eval, 1);
      } else {
        cPar.trace_G = EigenDecomp_Zeroed(G2, U, eval, 0);
      }
      gsl_matrix_free(G2);

      double eval_min = 0.0;
      for (size_t i = 0; i < eval->size; i++) {
        if ((gsl_vector_get(eval, i)) > 0){
          eval_min = gsl_vector_get(eval, i);
          i = eval->size;
        }
      }
      printf("eval_min = %.5f \n", eval_min);
      for (size_t i = 0; i < eval->size; i++) {
        if ((gsl_vector_get(eval, i)) == 0)
        gsl_vector_set(eval, i, eval_min/2.0);
      }

      cPar.time_eigen =
          (clock() - time_start) / (double(CLOCKS_PER_SEC) * 60.0);
    } else {
      ReadFile_eigenU(cPar.file_ku, cPar.error, U);
      if (cPar.error == true) {
        cout << "error! fail to read the U file. " << endl;
        return;
      }

      ReadFile_eigenD(cPar.file_kd, cPar.error, eval);
      if (cPar.error == true) {
        cout << "error! fail to read the D file. " << endl;
        return;
      }

      cPar.trace_G = 0.0;
      for (size_t i = 0; i < eval->size; i++) {
        if (gsl_vector_get(eval, i) < 1e-10) {
          gsl_vector_set(eval, i, 0);
        }
        cPar.trace_G += gsl_vector_get(eval, i);
      }
      cPar.trace_G /= (double)eval->size;
    }
 
  

    ReadFile_S(cPar.file_S, cPar.indicator_idv, cPar.indicator_snp,
                   cPar.error, S);


gsl_matrix *Dk = gsl_matrix_alloc(Y->size1, Y->size1);
gsl_matrix *temp = gsl_matrix_alloc(Y->size1, Y->size1);
gsl_matrix *Ki = gsl_matrix_alloc(Y->size1, Y->size1);

  for (size_t i = 0; i < Y->size1; ++i) {
    double delta = 1.0 / gsl_vector_get(eval, i);
    gsl_matrix_set(Dk, i,i, delta);
  }
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, U, Dk, 0.0, temp);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, temp, U, 0.0, Ki);
  
  gsl_matrix_free(Dk);
  gsl_matrix_free(temp);


  gsl_matrix *SKi  = gsl_matrix_alloc( cPar.ns_test, Y->size1);
  gsl_matrix *SKiS = gsl_matrix_alloc(cPar.ns_test,cPar.ns_test);
    
  gsl_matrix_set_zero( SKi );
  gsl_matrix_set_zero( SKiS );


  gsl_blas_dgemm(CblasTrans,   CblasNoTrans, 1.0,  S, Ki, 0.0, SKi);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, SKi,  S, 0.0, SKiS);




                   
    if (cPar.a_mode == 31) {
      cPar.WriteMatrix(U, "eigenU");
      cPar.WriteVector(eval, "eigenD");
    } else {
    
      size_t curve_num = cPar.curve_type;
      gsl_vector *Times = gsl_vector_alloc(Y->size2);
      if(curve_num == 1 ){
        for (size_t j = 0; j < Y->size2; ++j) {
          gsl_vector_set(Times, j, j+1);
        }
      }else{
       
        for (size_t j = 0; j < Y->size2; ++j) {
          double tmp = 0;
          for (size_t i = 0; i < Y->size1; ++i) {            
            tmp = tmp + gsl_matrix_get(Y,i,j);
          }
          tmp = tmp / (Y->size1);
          gsl_vector_set(Times, j, tmp);
         
        }
      
      }

    MVLMM cMvlmm;
    if (cPar.a_mode == 1 || cPar.a_mode == 2 || cPar.a_mode == 3 ||
    cPar.a_mode == 4) {
  
    cMvlmm.CopyFromParam(cPar);

    if (!cPar.file_bfile.empty()) {
    cMvlmm.AnalyzePlink_0S(U, eval, S, Y, Times, W, curve_num);
}   else {
    cerr << "Error: Only PLINK format (-bfile) is supported. Please provide -bfile input." << endl;
    exit(1);
}
}
          cMvlmm.WriteFiles_yxx(); 
          cMvlmm.CopyToParam(cPar);
      
      gsl_vector_free(Times);
    }

   
    gsl_matrix_safe_free(Y);
    gsl_matrix_safe_free(W);
    gsl_matrix_safe_free(G);
    gsl_matrix_safe_free(U);
    gsl_vector_safe_free(eval);
  
}
  cPar.time_total = (clock() - time_begin) / (double(CLOCKS_PER_SEC) * 60.0);

  return;
}

#include "Eigen/Dense"

void DYNGBLUP::WriteLog(int argc, char **argv, PARAM &cPar) {
  string file_str;
  file_str = cPar.path_out + "/" + cPar.file_out;
  file_str += ".log.txt";

  ofstream outfile(file_str.c_str(), ofstream::out);
  if (!outfile) {
    cout << "error writing log file: " << file_str.c_str() << endl;
    return;
  }

  outfile << "##" << endl;
  outfile << "## DYNGBLUP Version    = " << version << " (" << date << ")" << endl;
  outfile << "## GSL Version      = " << GSL_VERSION << endl;
  outfile << "## Eigen Version    = " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << endl;
#ifdef OPENBLAS

  #ifndef OPENBLAS_LEGACY
  outfile << "## OpenBlas         =" << OPENBLAS_VERSION << " - " << openblas_get_config() << endl;
  outfile << "##   arch           = " << openblas_get_corename() << endl;
  outfile << "##   threads        = " << openblas_get_num_threads() << endl;
  #else
  outfile << "## OpenBlas         = " << openblas_get_config() << endl;
  #endif
  string* pStr = new string[4] { "sequential", "threaded", "openmp" };
  outfile << "##   parallel type  = " << pStr[openblas_get_parallel()] << endl;
#endif

  outfile << "##" << endl;
  outfile << "## Command Line Input = ";
  for (int i = 0; i < argc; i++) {
    outfile << argv[i] << " ";
  }
  outfile << endl;

  outfile << "##" << endl;
  time_t rawtime;
  time(&rawtime);
  tm *ptm = localtime(&rawtime);

  outfile << "## Date = " << asctime(ptm);

  outfile << "##" << endl;
  outfile << "## Summary Statistics:" << endl;
  outfile << "## number of total individuals = " << cPar.ni_total << endl;
  outfile << "## number of covariates = " << cPar.n_cvt << endl;
  outfile << "## number of phenotypes = " << cPar.n_ph << endl;
  outfile << endl;

  if (cPar.a_mode == 1 || cPar.a_mode == 2 || cPar.a_mode == 3 ||
      cPar.a_mode == 4 || cPar.a_mode == 5) 
  {
    outfile << "## REMLE log-likelihood in the null model = "
            << cPar.logl_remle_H0 << endl;
    outfile << "## MLE log-likelihood in the null model = " 
            << cPar.logl_mle_H0 << endl;
    outfile << endl;


    outfile << "## REMLE estimate for Vg in the null model: " << endl;
    for (size_t i = 0; i <= cPar.nr1_size; i++) {
      outfile << cPar.Vg_remle_null[i] << "\t";
    }
    outfile << endl;
    outfile << "## REMLE estimate for Vp in the null model: " << endl;
    for (size_t i = 0; i <= cPar.nr2_size; i++) {
      outfile << cPar.Vp_remle_null[i] << "\t";
    }
    outfile << endl;
    outfile << "## REMLE estimate for Ve in the null model: " << endl;
    for (size_t i = 0; i < (cPar.Ve_remle_null).size(); i++) {
      outfile << cPar.Ve_remle_null[i] << "\t";
    }
    outfile << endl;

    outfile << "## estimate for beta (c_size by ne_size) in the null model: "
            << endl;
    outfile << "## REMLE estimate for beta in the null model: " << endl;
    for (size_t i = 0; i < (cPar.beta_remle_null).size(); i++) {
      outfile << cPar.beta_remle_null[i] << "\t";
    }
    outfile << endl;
  }


  outfile << "##" << endl;
  outfile << "## Computation Time:" << endl;
  outfile << "## total computation time = " << cPar.time_total << " min "
          << endl;
  outfile << "## computation time break down: " << endl;
  if (cPar.a_mode == 21 || cPar.a_mode == 22) {
    outfile << "##      time on calculating relatedness matrix = "
            << cPar.time_G << " min " << endl;
  }
  if (cPar.a_mode == 31) {
    outfile << "##      time on eigen-decomposition = " << cPar.time_eigen
            << " min " << endl;
  }
  if (cPar.a_mode == 1 || cPar.a_mode == 2 || cPar.a_mode == 3 ||
      cPar.a_mode == 4 || cPar.a_mode == 5 || cPar.a_mode == 11 ||
      cPar.a_mode == 12 || cPar.a_mode == 13 || cPar.a_mode == 14 ||
      cPar.a_mode == 16) {
    outfile << "##      time on eigen-decomposition = " << cPar.time_eigen
            << " min " << endl;
  }
  if ((cPar.a_mode >= 1 && cPar.a_mode <= 4) ||
      (cPar.a_mode >= 51 && cPar.a_mode <= 54)) {
    outfile << "##      time on optimization = " << cPar.time_opt << " min "
            << endl;
  }
  if (cPar.a_mode == 41 || cPar.a_mode == 42) {
    outfile << "##      time on eigen-decomposition = " << cPar.time_eigen
            << " min " << endl;
  }
  outfile << "##" << endl;

  outfile.close();
  outfile.clear();
  return;
}
