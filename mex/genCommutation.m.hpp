// Automatically translated using Matlab2cpp 0.5 on 2017-01-25 15:42:02
            
#ifndef GENCOMMUTATION_M_HPP
#define GENCOMMUTATION_M_HPP

// #include "mconvert.h"
#include "armadillo.hpp"
using namespace arma ;

mat genCommutation(int m) ;
mat eyePart(int m, int i, int j) ;

mat genCommutation(int m) {
  mat K ;
  int iComm, jComm ;
  // K = zeros(m^2)
  K = zeros(pow(m, 2), pow(m, 2)) ;
  for (iComm=1; iComm<=m; iComm++)
  {
    for (jComm=1; jComm<=m; jComm++)
    {
      // K = K + kron(eyePart(m, iComm, jComm), eyePart(m, iComm, jComm)')
      K = K + kron(eyePart(m, iComm, jComm), trans(eyePart(m, iComm, jComm)));
    }
  }
  return K ;
}

mat eyePart(int m, int i, int j) {
  // out = [zeros(i-1, m); zeros(1, j-1), 1, zeros(1, m-j); zeros(m-i, m)]
  
  rowvec temp1;
  temp1 = join_rows(join_rows(zeros(1, j-1), eye(1,1)), zeros(1, m-j));
  mat out;
  out = join_cols(join_cols(zeros(i-1, m), temp1), zeros(m-i, m));
  
  return out ;
}
#endif