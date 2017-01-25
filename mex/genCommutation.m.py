# Automatically translated using Matlab2cpp 0.5 on 2017-01-25 15:42:02
#
# encoding: utf-8
#
# Supplement file
#
# Valid inputs:
#
# uword   int     float   double cx_double
# uvec    ivec    fvec    vec    cx_vec
# urowvec irowvec frowvec rowvec cx_rowvec
# umat    imat    fmat    mat    cx_mat
# ucube   icube   fcube   cube   cx_cube
#
# char    string  struct  structs func_lambda

functions = {
  "eyePart" : {
      "i" : "int",
      "j" : "int",
      "m" : "int",
    "out" : "mat",
  },
  "genCommutation" : {
        "K" : "imat",
    "iComm" : "int",
    "jComm" : "int",
        "m" : "int",
  },
}
includes = [
  '#include "mconvert.h"',
  '#include <armadillo>',
  'using namespace arma ;',
]