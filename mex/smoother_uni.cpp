/* smoother_uni.cpp

Runs the univariate Kalman smoother with the exact initialization. 
Requires that the filter has been previously run.

To compile in MATLAB, see the make.m script.

This function uses the matrix library armadillo: http://arma.sourceforge.net/

The utility Matlab2cpp 0.5 was used in the initial stages of converting this 
from Matlab. Note that many changes have followed. 

Copyright: David Kelley, 2017. 
*/

#define ARMA_DONT_PRINT_ERRORS
#include "armaMex.hpp"
#include "smoother_uni_mex.hpp"

cube getMaybeCube(const mxArray *cubePtr) {
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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Validate inputs
  if (nrhs != 3) {
    mexErrMsgIdAndTxt("smoother_uni:nrhs", "3 inputs required: y, ssStruct, and fOut.");
  }
  if (nlhs > 6) {
    mexErrMsgIdAndTxt("smoother_uni:nlhs", "Maximum of 6 outputs allowed.");
  }

  // y data
  if (!(mxIsDouble(prhs[0]))){
    mexErrMsgIdAndTxt( "smoother_uni:inputNotDouble", "y must be of type double.");
  }
  mat y = armaGetPr(prhs[0]);

  // StateSpace structure
  if(!mxIsStruct(prhs[1]))
    mexErrMsgIdAndTxt( "smoother_uni:inputNotStruct", "ss must be a structure.");
  
  _Ss ss;

  ss.Z = getMaybeCube(mxGetField(prhs[1], 0, "Z"));
  ss.d = armaGetPr(mxGetField(prhs[1], 0, "d"));
  ss.H = getMaybeCube(mxGetField(prhs[1], 0, "H"));
  ss.T = getMaybeCube(mxGetField(prhs[1], 0, "T"));
  ss.c = armaGetPr(mxGetField(prhs[1], 0, "c"));
  ss.R = getMaybeCube(mxGetField(prhs[1], 0, "R"));
  ss.Q = getMaybeCube(mxGetField(prhs[1], 0, "Q"));
  ss.a0 = (vec) armaGetPr(mxGetField(prhs[1], 0, "a0"));
  ss.A0 = armaGetPr(mxGetField(prhs[1], 0, "A0"));
  ss.R0 = armaGetPr(mxGetField(prhs[1], 0, "R0"));
  ss.Q0 = armaGetPr(mxGetField(prhs[1], 0, "Q0"));

  mxArray *tauPtr = mxGetField(prhs[1], 0, "tau");

  if (tauPtr == NULL) 
    mexErrMsgIdAndTxt( "smoother_uni:no_field",
      "ss.tau does not exist.");

  if(!mxIsStruct(tauPtr))
    mexErrMsgIdAndTxt( "smoother_uni:inputNotStruct",
      "tau must be a structure.");
  _Tau tau;     
  tau.Z = armaGetPr(mxGetField(tauPtr, 0, "Z"));
  tau.d = armaGetPr(mxGetField(tauPtr, 0, "d"));
  tau.H = armaGetPr(mxGetField(tauPtr, 0, "H"));
  tau.T = armaGetPr(mxGetField(tauPtr, 0, "T"));
  tau.c = armaGetPr(mxGetField(tauPtr, 0, "c"));
  tau.R = armaGetPr(mxGetField(tauPtr, 0, "R"));
  tau.Q = armaGetPr(mxGetField(tauPtr, 0, "Q"));

  // fOut structure
  if(!mxIsStruct(prhs[2]))
    mexErrMsgIdAndTxt( "gradient_multi:inputNotStruct",
      "fOut must be a structure.");

  _Fout fOut;

  fOut.a = armaGetPr(mxGetField(prhs[2], 0, "a"));
  fOut.P = getMaybeCube(mxGetField(prhs[2], 0, "P"));
  fOut.Pd = getMaybeCube(mxGetField(prhs[2], 0, "Pd"));
  fOut.v = armaGetPr(mxGetField(prhs[2], 0, "v"));
  fOut.F = armaGetPr(mxGetField(prhs[2], 0, "F"));
  fOut.Fd = armaGetPr(mxGetField(prhs[2], 0, "Fd"));
  fOut.K = getMaybeCube(mxGetField(prhs[2], 0, "K"));
  fOut.Kd = getMaybeCube(mxGetField(prhs[2], 0, "Kd"));
  fOut.dt = mxGetScalar(mxGetField(prhs[2], 0, "dt"));
  
  // Compute
  _smoother output;
  output = smoother_uni_mex(y, ss.Z, ss.d, ss.H, ss.T, ss.c, ss.R, ss.Q, 
    ss.a0, ss.A0, ss.R0, ss.Q0, tau, fOut);

  // Set outputs
  plhs[0] = armaCreateMxMatrix(output.alpha.n_rows, output.alpha.n_cols);
  armaSetPr(plhs[0], output.alpha);  

  plhs[1] = armaCreateMxMatrix(output.eta.n_rows, output.eta.n_cols);
  armaSetPr(plhs[1], output.eta);  

  plhs[2] = armaCreateMxMatrix(output.r.n_rows, output.r.n_cols);
  armaSetPr(plhs[2], output.r);

  plhs[3] = armaCreateMxMatrix(output.N.n_rows, output.N.n_cols, output.N.n_slices);
  armaSetCubeData(plhs[3], output.N);

  plhs[4] = armaCreateMxMatrix(output.V.n_rows, output.V.n_cols, output.V.n_slices);
  armaSetCubeData(plhs[4], output.V);

  plhs[5] = armaCreateMxMatrix(output.a0tilde.n_rows, output.a0tilde.n_cols);
  armaSetPr(plhs[5], output.a0tilde);
}
