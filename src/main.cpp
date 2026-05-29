/*
    Dynamic Genomic Best linear Unbiased Prediction  (DynGBLUP)
    Copyright © 2026, Meixia Ye
    
*/

#include "dyngblup.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

int main(int argc, char *argv[]) {
  DYNGBLUP cDyngblup;
  PARAM cPar;
  
  if(cPar.error == true){ cPar.error = false; }
  gsl_set_error_handler (&DYNGBLUP_gsl_error_handler);
 

  if (argc <= 1) {
    cDyngblup.PrintHeader();
    cDyngblup.PrintHelp(0);
    return EXIT_SUCCESS;
  }
  if (argc == 2 && argv[1][0] == '-' && argv[1][1] == 'h') {
    cDyngblup.PrintHeader();
    cDyngblup.PrintHelp(0);
    return EXIT_SUCCESS;
  }
  if (argc == 3 && argv[1][0] == '-' && argv[1][1] == 'h') {
    string str;
    str.assign(argv[2]);
    cDyngblup.PrintHeader();
    cDyngblup.PrintHelp(atoi(str.c_str()));
    return EXIT_SUCCESS;
  }
  if (argc == 2 && argv[1][0] == '-' && argv[1][1] == 'l') {
    cDyngblup.PrintHeader();
    cDyngblup.PrintLicense();
    return EXIT_SUCCESS;
  }
  

  cDyngblup.Assign(argc, argv, cPar);

  ifstream check_dir((cPar.path_out).c_str());
  if (!check_dir) {
    mkdir((cPar.path_out).c_str(), S_IRWXU | S_IRGRP | S_IROTH);
  }

  if (!is_quiet_mode())
    cDyngblup.PrintHeader();

  if (cPar.error == true) {
    return EXIT_FAILURE;
  }

  if (is_quiet_mode()) {
    stringstream ss;
    cout.rdbuf(ss.rdbuf());
  }

  cPar.CheckParam();

  if (cPar.error == true) {
    return EXIT_FAILURE;
  }

  cDyngblup.BatchRun(cPar);

  if (cPar.error == true) {
    return EXIT_FAILURE;
  }

  cDyngblup.WriteLog(argc, argv, cPar);

  return EXIT_SUCCESS;
}
