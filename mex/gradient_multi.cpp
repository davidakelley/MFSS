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
#include "gradient_multi_mex.hpp"

 cube getMaybeCube(const mxArray *cubePtr)
 {
    cube armaCube;
    if (mxGetNumberOfDimensions(cubePtr)==2) {
        armaCube.set_size(mxGetM(cubePtr), mxGetN(cubePtr), 1);
        mat temp1 = armaGetPr(cubePtr);
        armaCube.slice(0) = temp1;
    }
    else {
        armaCube = armaGetCubePr(cubePtr);
    }
    
    return armaCube;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate inputs
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("gradient_multi:nrhs", "Four inputs required.");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("gradient_multi:nlhs", "Only one output allowed.");
    }

    // y data
    if (!(mxIsDouble(prhs[0]))){
        mexErrMsgIdAndTxt( "gradient_multi:inputNotDouble",
            "y must be of type double.");
    }
    mat y = armaGetPr(prhs[0]);
    
    // StateSpace structure
    if(!mxIsStruct(prhs[1]))
        mexErrMsgIdAndTxt( "gradient_multi:inputNotStruct",
            "ss must be a structure.");
    _Ss ss;
    ss.p = (int) mxGetScalar(mxGetField(prhs[1], 0, "p"));
    ss.m = (int) mxGetScalar(mxGetField(prhs[1], 0, "m"));
    ss.g = (int) mxGetScalar(mxGetField(prhs[1], 0, "g"));
    ss.n = (int) mxGetScalar(mxGetField(prhs[1], 0, "n"));

    ss.Z = getMaybeCube(mxGetField(prhs[1], 0, "Z"));
    ss.d = armaGetPr(mxGetField(prhs[1], 0, "d"));
    ss.H = getMaybeCube(mxGetField(prhs[1], 0, "H"));
    ss.T = getMaybeCube(mxGetField(prhs[1], 0, "T"));
    ss.c = armaGetPr(mxGetField(prhs[1], 0, "c"));
    ss.R = getMaybeCube(mxGetField(prhs[1], 0, "R"));
    ss.Q = getMaybeCube(mxGetField(prhs[1], 0, "Q"));
    ss.a0 = (vec) armaGetPr(mxGetField(prhs[1], 0, "a0"));
    ss.P0 = armaGetPr(mxGetField(prhs[1], 0, "P0"));

    mxArray *tauPtr = mxGetField(prhs[1], 0, "tau");

    if (tauPtr == NULL) 
        mexErrMsgIdAndTxt( "gradient_multi:no_field",
            "ss.tau does not exist.");

    if(!mxIsStruct(tauPtr))
        mexErrMsgIdAndTxt( "gradient_multi:inputNotStruct",
            "ss.tau must be a structure.");
    _Tau tau;     
    tau.Z = armaGetPr(mxGetField(tauPtr, 0, "Z"));
    tau.d = armaGetPr(mxGetField(tauPtr, 0, "d"));
    tau.H = armaGetPr(mxGetField(tauPtr, 0, "H"));
    tau.T = armaGetPr(mxGetField(tauPtr, 0, "T"));
    tau.c = armaGetPr(mxGetField(tauPtr, 0, "c"));
    tau.R = armaGetPr(mxGetField(tauPtr, 0, "R"));
    tau.Q = armaGetPr(mxGetField(tauPtr, 0, "Q"));
    ss.tau = tau;

    // Parameter gradient structure
    if(!mxIsStruct(prhs[2]))
        mexErrMsgIdAndTxt( "gradient_multi:inputNotStruct",
            "G must be a structure.");
    _G G;
    G.Z = getMaybeCube(mxGetField(prhs[2], 0, "Z"));
    G.d = getMaybeCube(mxGetField(prhs[2], 0, "d"));
    G.H = getMaybeCube(mxGetField(prhs[2], 0, "H"));
    G.T = getMaybeCube(mxGetField(prhs[2], 0, "T"));
    G.c = getMaybeCube(mxGetField(prhs[2], 0, "c"));
    G.R = getMaybeCube(mxGetField(prhs[2], 0, "R"));
    G.Q = getMaybeCube(mxGetField(prhs[2], 0, "Q"));
    G.a0 = armaGetPr(mxGetField(prhs[2], 0, "a0"));
    G.P0 = armaGetPr(mxGetField(prhs[2], 0, "P0"));

    // fOut structure
    if(!mxIsStruct(prhs[3]))
        mexErrMsgIdAndTxt( "gradient_multi:inputNotStruct",
            "fOut must be a structure.");

    _Fout fOut;

    fOut.a = armaGetPr(mxGetField(prhs[3], 0, "a"));
    fOut.P = getMaybeCube(mxGetField(prhs[3], 0, "P"));
    fOut.v = armaGetPr(mxGetField(prhs[3], 0, "v"));
    fOut.F = getMaybeCube(mxGetField(prhs[3], 0, "F"));
    fOut.M = getMaybeCube(mxGetField(prhs[3], 0, "M"));
    fOut.K = getMaybeCube(mxGetField(prhs[3], 0, "K"));
    fOut.L = getMaybeCube(mxGetField(prhs[3], 0, "L"));
    fOut.w = armaGetPr(mxGetField(prhs[3], 0, "w"));
    fOut.Finv = getMaybeCube(mxGetField(prhs[3], 0, "Finv"));

    // // Compute
    vec gradient;
    gradient = gradient_multi_mex(y, ss, G, fOut);

    // mat gradient;
    // gradient = tau.Z;

    // Set outputs
    plhs[0] = armaCreateMxMatrix(gradient.n_rows, gradient.n_cols);
    armaSetPr(plhs[0], gradient);  
}
