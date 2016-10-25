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
 * Copyright: David Kelley 2015
 *
 */

#define ARMA_DONT_PRINT_ERRORS
#include "armaMex.hpp"
#define uint (unsigned int)
 
void ksmooth_multi(
		mat y, cube Zt, mat tauZ, cube Ht, mat tauH, 
		cube Tt, mat tauT, cube Rt, mat tauR, cube Qt, mat tauQ,
		mat a, cube P, mat v, cube M, cube K, cube L, cube Finv, 
		mat a0, mat P0, 
		mat &alpha, mat &eta, mat &epsilon, mat &r, cube &N, cube &V, cube &J, mat &a0tilde)
{
   
	unsigned int p = y.n_rows;
	unsigned int n = y.n_cols;
	unsigned int m = Tt.n_rows;
	unsigned int ii;                // Loop index

    mat u = mat(p, n, fill::zeros);

	uvec ind;
	mat W;
	mat eyeP = eye(p, p);
	 
	double logDetF;
	uvec iiVec(1);

	for(ii = n; ii-- > 0; ) { // Iterate from n-1 to 0
		// % % Create W matrix for possible missing values
		// % W = eye(p);
		// % ind = ~isnan(y(:,ii));
		// % W = W((ind==1),:);
		ind = find_finite(y.col(ii));
		W = eyeP.rows(ind);
		iiVec.fill(ii); // Allow easier submats
		
		// % r(:,ii) = (W*Zt(:,:,tauZ(ii)))'*Finv(ind,ind,ii)*v(ind,ii) + ...
		// %     L(:,:,ii)'*r(:,ii+1);
		r.col(ii) = (W * Zt.slice(uint tauZ(ii)-1)).t() \
				* Finv.slice(ii).submat(ind,ind) * v(ind,iiVec) \
				+ L.slice(ii).t() * r.col(ii+1);
		
		// % N(:,:,ii) = (W*Zt(:,:,tauZ(ii)))'*Finv(ind,ind,ii)*(W*Zt(:,:,tauZ(ii))) ...
		// %     + L(:,:,ii)'*N(:,:,ii+1)*L(:,:,ii);
		N.slice(ii) = (W * Zt.slice(uint tauZ(ii)-1)).t() \
				* Finv.slice(ii).submat(ind,ind) \
				* (W * Zt.slice(uint tauZ(ii)-1)) \
				+ L.slice(ii).t() * N.slice(ii+1) * L.slice(ii);
		
		// % u(ind,ii) = Finv(ind,ind,ii)*v(ind,ii)-K(:,ind,ii)'*r(:,ii);
		u(ind,iiVec) = Finv.slice(ii).submat(ind,ind) * v(ind,iiVec) \
		        - K.slice(ii).cols(ind).t() * r.col(ii);

		// % alpha(:,ii) = a(:,ii) + P(:,:,ii)*r(:,ii);
		alpha.col(ii) = a.col(ii) + P.slice(ii) * r.col(ii);
		
		// % eta(:,ii) = Qt(:,:,tauQ(ii))*Rt(:,:,tauR(ii))'*r(:,ii);
		eta.col(ii) = Qt.slice(uint tauQ(ii)-1) * Rt.slice(uint tauR(ii)-1).t() * r.col(ii);
		
		// % epsilon(ind,ii) = (W*Ht(:,:,tauH(ii))*W')*u(ind,ii);
		epsilon.submat(ind,iiVec) = (W * Ht.slice(uint tauH(ii)-1) * W.t()) \
		        * u(ind,iiVec);

		// % V(:,:,ii) = P(:,:,ii) - P(:,:,ii)*N(:,:,ii)*P(:,:,ii);
		V.slice(ii) = P.slice(ii) - P.slice(ii) * N.slice(ii) * P.slice(ii);
		
		// % J(:,:,ii) = P(:,:,ii)*L(:,:,ii)'*eye(m)*(eye(m)-N(:,:,ii+1)*P(:,:,ii+1));
		J.slice(ii) = P.slice(ii) * L.slice(ii).t() * eye(m,m) \
				* (eye(m,m) - N.slice(ii+1) * P.slice(ii+1));
	}
	// % a0tilde = a0 + P0*Tt(:,:,tauT(1))'*r(:,1);
	a0tilde = a0 + P0*Tt.slice(uint tauT(0)-1).t() * r.col(0);

	return;
}

#define checkDouble(matName, n_prhs)\
if(!mxIsDouble(prhs[n_prhs])) {\
	mexErrMsgIdAndTxt("kfilter:notDouble", matName " must be a double matrix.");\
}

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

vec getColumn(const mxArray *vecPtr)
{
	vec armaVec;
	if (mxGetM(vecPtr)!=1 && mxGetN(vecPtr)==1) {
		armaVec = armaGetPr(vecPtr);
	}
	else if (mxGetM(vecPtr)==1 && mxGetN(vecPtr)!=1) {
		rowvec armaRowvec = armaGetPr(vecPtr);
		armaVec = armaRowvec.t();
	}
	else {
		
	}
	return armaVec;
}

void printDims(mat matToPrint, char *matName)
{
	mexPrintf("%s: %d x %d\n", matName, matToPrint.n_cols, matToPrint.n_rows);
}
 
void printDims(cube cubeToPrint, char *matName)
{
	mexPrintf("%s: %d x %d x %d\n", matName, cubeToPrint.n_cols, cubeToPrint.n_rows, cubeToPrint.n_slices);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// Validate inputs
	if(nrhs!=20) {
		mexErrMsgIdAndTxt("ksmoother_uni:nrhs", "Nineteen input required.");
	}
	if(nlhs!=8) {
		mexErrMsgIdAndTxt("ksmoother_uni:nlhs", "Eight output required.");
	}
	
	// Input variables:
	// y, tauZ, Zt, taud, dt, tauT, Tt, tauc, ct, tauQ, Qt, tauR, Rt, tauH, Ht, a0, m, kappa

	// Check input sizes
	checkDouble("y", 0);
	checkDouble("Zt", 1);
	checkDouble("tauZ", 2);
	checkDouble("Ht", 3);
	checkDouble("tauH", 4);
	checkDouble("Tt", 5);
	checkDouble("tauT", 6);
	checkDouble("Rt", 7);
	checkDouble("tauR", 8);
	checkDouble("Qt", 9);
	checkDouble("tauQ", 10);
	checkDouble("a", 11);
	checkDouble("P", 12);
	checkDouble("v", 13);
	checkDouble("M", 14);
	checkDouble("K", 15);
	checkDouble("L", 16);
	checkDouble("Finv", 17);
	checkDouble("a0", 18);
	checkDouble("P0", 19);

	// Get inputs & outputs
	mat y    = armaGetPr(prhs[0]);
	cube Zt  = getMaybeCube(prhs[1]);
	mat tauZ = armaGetPr(prhs[2]);
	cube Ht  = getMaybeCube(prhs[3]);
	mat tauH = armaGetPr(prhs[4]);
	cube Tt  = getMaybeCube(prhs[5]);
	mat tauT = armaGetPr(prhs[6]);
	cube  Rt = getMaybeCube(prhs[7]);
	mat tauR = armaGetPr(prhs[8]);
	cube  Qt = getMaybeCube(prhs[9]);
	mat tauQ = armaGetPr(prhs[10]);
	mat a    = armaGetPr(prhs[11]);
	cube P   = getMaybeCube(prhs[12]);
	mat v    = armaGetPr(prhs[13]);
	cube M   = getMaybeCube(prhs[14]);
	cube K   = getMaybeCube(prhs[15]);
	cube L   = getMaybeCube(prhs[16]);
	cube Finv= getMaybeCube(prhs[17]);
	mat a0   = armaGetPr(prhs[18]);
	mat P0   = armaGetPr(prhs[19]);

	int n    = y.n_cols;
	int p    = y.n_rows;
	int m    = Tt.n_rows;
	int g    = Rt.n_cols;
	
	// Initialize output matricies
	mat alpha(m, n, fill::zeros);
	plhs[0] = armaCreateMxMatrix(m, n);

	mat eta(g, n, fill::zeros);
	plhs[1] = armaCreateMxMatrix(g, n);

	mat epsilon(p, n, fill::zeros);
	plhs[2] = armaCreateMxMatrix(p, n);

	mat r(m, n+1, fill::zeros);
	plhs[3] = armaCreateMxMatrix(m, n+1);
	
	cube N(m, m, n+1, fill::zeros);
	plhs[4] = armaCreateMxMatrix(m, m, n+1);
	
	cube V(m, m, n, fill::zeros);
	plhs[5] = armaCreateMxMatrix(m, m, n);

	cube J(m, m, n, fill::zeros);
	plhs[6] = armaCreateMxMatrix(m, m, n);
	
	mat a0tilde(m, 1, fill::zeros);
	plhs[7] = armaCreateMxMatrix(m, 1);

	// Compute
	try
	{
		ksmooth_multi(y, Zt, tauZ, Ht, tauH, Tt, tauT, Rt, tauR, Qt, tauQ, 
				a, P, v, M, K, L, Finv, a0, P0, alpha, eta, epsilon, r, N, V, J, a0tilde);
	}
	catch (int e)
	{
		mexErrMsgIdAndTxt("ksmoother_multi:unknown", "An unknown error occured.");
	}
	
	// Set outputs
	armaSetPr(plhs[0], alpha);
	armaSetPr(plhs[1], eta);
	armaSetPr(plhs[2], epsilon);
	armaSetPr(plhs[3], r);
	armaSetCubeData(plhs[4], N);
	armaSetCubeData(plhs[5], V);
	armaSetCubeData(plhs[6], J);
	armaSetPr(plhs[7], a0tilde);

	return;
}
