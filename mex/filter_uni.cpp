/* filter_uni.cpp

Runs the univariate Kalman filter with the exact initialization. 

To compile in MATLAB, see the make.m script.

This function uses the matrix library armadillo: http://arma.sourceforge.net/

The utility Matlab2cpp 0.5 was used in the initial stages of converting this 
from Matlab. Note that many changes have followed. 

Copyright: David Kelley, 2017. 
*/

#define ARMA_DONT_PRINT_ERRORS
#include "armaMex.hpp"
#include "filter_uni_mex.hpp"

cube getMaybeCube(const mxArray *cubePtr) {
  cube armaCube;
  if (mxGetNumberOfDimensions(cubePtr)==2) {
    armaCube.set_size(mxGetM(cubePtr), mxGetN(cubePtr), 1);
    mat temp1; 
    if (mxGetData(cubePtr) != NULL) {
      temp1 = armaGetPr(cubePtr);
    } else {
      temp1.zeros(mxGetM(cubePtr), mxGetN(cubePtr));
    }  
    armaCube.slice(0) = temp1;
  } else {
    armaCube = armaGetCubePr(cubePtr);
  }
  return armaCube;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Validate inputs
  if (nrhs != 3) {
    mexErrMsgIdAndTxt("filter_uni:nrhs", "Three inputs required: y, x and ssStruct.");
  }
  if (nlhs > 10) {
    mexErrMsgIdAndTxt("filter_uni:nlhs", "Maximum of 10 outputs allowed.");
  }

  // y data
  if (!(mxIsDouble(prhs[0]))){
    mexErrMsgIdAndTxt( "filter_uni:inputNotDouble", "y must be of type double.");
  }
  mat y = armaGetPr(prhs[0]);

  // x data
  if (!(mxIsDouble(prhs[1]))){
    mexErrMsgIdAndTxt( "filter_uni:inputNotDouble", "x must be of type double.");
  }
  mat x;
  // Test if x is empty. If it is, we can't use the standard amadillo method 
  // to pull it and have to just create the empty matrix.
  if (mxGetData(prhs[1]) != NULL) {
    x = armaGetPr(prhs[1]);  
  } else {
    x.zeros(0, y.n_cols);
  }  

  // StateSpace structure
  if(!mxIsStruct(prhs[2]))
    mexErrMsgIdAndTxt( "filter_uni:inputNotStruct", "ss must be a structure.");
  
  _Ss ss;

  ss.Z = getMaybeCube(mxGetField(prhs[2], 0, "Z"));
  ss.d = armaGetPr(mxGetField(prhs[2], 0, "d"));
  ss.beta = getMaybeCube(mxGetField(prhs[2], 0, "beta"));
  ss.H = getMaybeCube(mxGetField(prhs[2], 0, "H"));
  ss.T = getMaybeCube(mxGetField(prhs[2], 0, "T"));
  ss.c = armaGetPr(mxGetField(prhs[2], 0, "c"));
  ss.R = getMaybeCube(mxGetField(prhs[2], 0, "R"));
  ss.Q = getMaybeCube(mxGetField(prhs[2], 0, "Q"));
  ss.a0 = (vec) armaGetPr(mxGetField(prhs[2], 0, "a0"));
  ss.A0 = armaGetPr(mxGetField(prhs[2], 0, "A0"));
  ss.R0 = armaGetPr(mxGetField(prhs[2], 0, "R0"));
  ss.Q0 = armaGetPr(mxGetField(prhs[2], 0, "Q0"));

  mxArray *tauPtr = mxGetField(prhs[2], 0, "tau");

  if (tauPtr == NULL) 
    mexErrMsgIdAndTxt( "filter_uni:no_field",
      "ss.tau does not exist.");

  if(!mxIsStruct(tauPtr))
    mexErrMsgIdAndTxt( "filter_uni:inputNotStruct",
      "tau must be a structure.");
  _Tau tau;     
  tau.Z = armaGetPr(mxGetField(tauPtr, 0, "Z"));
  tau.d = armaGetPr(mxGetField(tauPtr, 0, "d"));
  tau.beta = armaGetPr(mxGetField(tauPtr, 0, "beta"));
  tau.H = armaGetPr(mxGetField(tauPtr, 0, "H"));
  tau.T = armaGetPr(mxGetField(tauPtr, 0, "T"));
  tau.c = armaGetPr(mxGetField(tauPtr, 0, "c"));
  tau.R = armaGetPr(mxGetField(tauPtr, 0, "R"));
  tau.Q = armaGetPr(mxGetField(tauPtr, 0, "Q"));

  // Compute
  _filter output;
  output = filter_uni_mex(y, x, ss.Z, ss.d, ss.beta, ss.H, ss.T, ss.c, ss.R, ss.Q, 
    ss.a0, ss.A0, ss.R0, ss.Q0, tau);

  // Set outputs
  plhs[0] = armaCreateMxMatrix(output.a.n_rows, output.a.n_cols);
  armaSetPr(plhs[0], output.a);  

  plhs[1] = armaCreateMxMatrix(1, 1);
  *mxGetPr(plhs[1]) = output.logli;

  plhs[2] = armaCreateMxMatrix(output.P.n_rows, output.P.n_cols, output.P.n_slices);
  armaSetCubeData(plhs[2], output.P);

  plhs[3] = armaCreateMxMatrix(output.Pd.n_rows, output.Pd.n_cols, output.Pd.n_slices);
  armaSetCubeData(plhs[3], output.Pd);

  plhs[4] = armaCreateMxMatrix(output.v.n_rows, output.v.n_cols);
  armaSetPr(plhs[4], output.v);

  plhs[5] = armaCreateMxMatrix(output.F.n_rows, output.F.n_cols);
  armaSetPr(plhs[5], output.F);

  plhs[6] = armaCreateMxMatrix(output.Fd.n_rows, output.Fd.n_cols);
  armaSetPr(plhs[6], output.Fd);

  plhs[7] = armaCreateMxMatrix(output.K.n_rows, output.K.n_cols, output.K.n_slices);
  armaSetCubeData(plhs[7], output.K);

  plhs[8] = armaCreateMxMatrix(output.Kd.n_rows, output.Kd.n_cols, output.Kd.n_slices);
  armaSetCubeData(plhs[8], output.Kd);

  plhs[9] = armaCreateMxMatrix(1, 1);
  *mxGetPr(plhs[9]) = output.dt;
}
