/*
    Dynamic Genomic Best linear Unbiased Prediction  (DynGBLUP)
    Copyright © 2026, Meixia Ye
    
*/

#ifndef __DYNGBLUP_H__
#define __DYNGBLUP_H__

#include "param.h"

using namespace std;

class DYNGBLUP {

public:

  string version;
  string date;
  string year;


  DYNGBLUP(void);

  void PrintHeader(void);
  void PrintHelp(size_t option);
  void PrintLicense(void);
  void Assign(int argc, char **argv, PARAM &cPar);
  void BatchRun(PARAM &cPar);
  void WriteLog(int argc, char **argv, PARAM &cPar);
};

void DYNGBLUP_gsl_error_handler(const char* reason, const char* file, int line, int gsl_errno);

#endif
