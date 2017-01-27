/*
 * kfiltersmoother_multi.cpp
 *
 * Computes the multivariate filter and smoother of the Kalman filter/smoother 
 *
 * To compile in MATLAB, run the make_kfiltersmoother_multi.m script.
 * 
 * Note that any comments styled " // % " indicate Matlab code 
 *   that is being translated.
 *
 * This function uses the matrix library armadillo: http://arma.sourceforge.net/
 *
 * Copyright: David Kelley 2017
 *
 */

#define ARMA_DONT_PRINT_ERRORS
#include "armaMex.hpp"
#include "genCommutation.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate inputs
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("genCommutation:nrhs", "One inputs required.");
    }
    if(nlhs > 1) {
        mexErrMsgIdAndTxt("genCommutation:nlhs", "One output required.");
    }

    int m = (int) mxGetScalar(prhs[0]);

    // Compute
    mat commutation;
    commutation = genCommutation(m);
    
    // Set outputs
    plhs[0] = armaCreateMxMatrix(commutation.n_rows, commutation.n_cols);
    armaSetPr(plhs[0], commutation);  
}
