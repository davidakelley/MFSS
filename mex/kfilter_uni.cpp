/*
 * kfilter.cpp
 *
 * Computes the filter part of the Kalman filter/smoother from
 *   GeneralizedKFilterSmootherUni.m
 *
 * To compile in MATLAB, run the make_kfilter.m script.
 * 
 * Note that any comments styled " // % " indicate Matlab code 
 *   that is being translated.
 *
 * This function uses the matrix library armadillo: http://arma.sourceforge.net/
 *
 * Copyright: David Kelley 2015
 *
 */

#include "armaMex.hpp"
#define uint (unsigned int)

void kfilt(
        mat y, cube Zt, mat tauZ, mat dt, mat taud, cube Ht, mat tauH, 
        cube Tt, mat tauT, mat ct, mat tauc, cube Rt, mat tauR, cube Qt, mat tauQ, 
        mat a0, mat P0, mat &LogL, mat &a, cube &P, 
        mat &v, mat &F, cube &M, cube &K, cube &L)
{
    int m = Tt.n_rows;

    // Set up for Kalman filter
    // % a(:,1) = Tt(:,:,tauT(1))*a0+ ct(:,tauc(1));
    // % P(:,:,1) = Tt(:,:,tauT(1))*P0*Tt(:,:,tauT(1))'...
    // %     +Rt(:,:,tauR(1))*Qt(:,:,tauQ(1))*Rt(:,:,tauR(1))';
    a.col(0) = Tt.slice(uint tauT(0)-1) * a0 + ct.col(uint tauc(0)-1) ;
    P.slice(0) = Tt.slice(uint tauT(0)-1) \
        * P0 \
        * Tt.slice(uint tauT(0)-1).t() \
        + Rt.slice(uint tauR(0)-1) \
        * Qt.slice(uint tauQ(0)-1) \
        * Rt.slice(uint tauR(0)-1).t();
    
    unsigned int ii, jj;				// Loop indexes
    unsigned int n = y.n_cols;
    unsigned int indJJ;
        
    uvec ind;
    vec ati;
    mat Pti;    

    for(ii=0; ii < n; ii++) { // Loop over n 
        // Store temp matricies
        ind = find_finite(y.col(ii));
        ati = a.col(ii);
        Pti = P.slice(ii);
        
        for(jj=0; jj < ind.n_elem; jj++) { // Loop over ind.n_elem 
            indJJ = ind(jj);

            // % v(ind(jj),ii) = y(ind(jj),ii) - Zt(ind(jj),:,tauZ(ii))*ati ...
            // % - dt(ind(jj),taud(ii));
            v(indJJ,ii) = y(indJJ,ii) \
                - as_scalar(Zt.slice(uint tauZ(ii)-1).row(indJJ) * ati) \
                - dt(indJJ, uint taud(ii)-1);
            
            
            // % F(ind(jj),ii) = Zt(ind(jj),:,tauZ(ii))*Pti*Zt(ind(jj),:,tauZ(ii))'...
            // % + Ht(ind(jj),ind(jj),tauH(ii));
            F(indJJ,ii) = as_scalar(Zt.slice(uint tauZ(ii)-1).row(indJJ) \
                * Pti \
                * Zt.slice(uint tauZ(ii)-1).row(indJJ).t() \
                + Ht.slice(uint tauH(ii)-1)(indJJ, indJJ));

            // % LogL(ind(jj),ii) = -1/2*(log(F(ind(jj),ii))+...
            // % (v(ind(jj),ii)^2)/F(ind(jj),ii));
            LogL(indJJ,ii) = as_scalar(log(F(indJJ,ii)) + v(indJJ,ii) * v(indJJ,ii) / F(indJJ,ii));

            // % M(:,ind(jj),ii) = Pti*Zt(ind(jj),:,tauZ(ii))';
            M.slice(ii).col(indJJ) = Pti * Zt.slice(uint tauZ(ii)-1).row(indJJ).t();
            
            // % ati = ati + M(:,ind(jj),ii)/F(ind(jj),ii)*v(ind(jj),ii);
            ati = ati + M.slice(ii).col(indJJ) / F(indJJ,ii) * v(indJJ,ii);

            // % Pti = Pti - M(:,ind(jj),ii)/F(ind(jj),ii)*M(:,ind(jj),ii)';
            Pti = Pti - M.slice(ii).col(indJJ) / F(indJJ,ii) * M.slice(ii).col(indJJ).t();
            
            
        }
        // % a(:,ii+1) = Tt(:,:,tauT(ii+1))*ati + ct(:,tauc(ii+1));
        a.col(ii+1) = Tt.slice(uint tauT(ii+1)-1) * ati + ct.col(uint tauc(ii+1)-1);

        // % P(:,:,ii+1) = Tt(:,:,tauT(ii+1))*Pti*Tt(:,:,tauT(ii+1))'+...
        // %   Rt(:,:,tauR(ii+1))*Qt(:,:,tauQ(ii+1))*Rt(:,:,tauR(ii+1))';
        P.slice(ii+1) = Tt.slice(uint tauT(ii+1)-1) \
            * Pti \
            * Tt.slice(uint tauT(ii+1)-1).t() \
            + Rt.slice(uint tauR(ii+1)-1) \
            * Qt.slice(uint tauQ(ii+1)-1) \
            * Rt.slice(uint tauR(ii+1)-1).t();

        K.slice(ii).cols(ind) = Tt.slice(uint tauT(ii+1)-1) * M.slice(ii).cols(ind);
        L.slice(ii) = Tt.slice(uint tauT(ii+1)-1) \
            - K.slice(ii).cols(ind) * Zt.slice(uint tauZ(ii)-1).rows(ind);
    }
    
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


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate inputs
    if(nrhs!=17) {
        mexErrMsgIdAndTxt("kfilter:nrhs", "Seventeen input required.");
    }
    if(nlhs!=8) {
        mexErrMsgIdAndTxt("kfilter:nlhs", "Eight output required.");
    }
        
    // Input variables:
    // kfilter_uni(y, Z, tau.Z, d, tau.d, H, tau.H, ...
    //              T, tau.T, c, tau.c, R, tau.R, Q, tau.Q, a0, P0);
            
    // Check input sizes
    checkDouble("y", 0);
    checkDouble("Zt", 1);
    checkDouble("tauZ", 2);
    checkDouble("dt", 3);
    checkDouble("taud", 4);
    checkDouble("Ht", 5);
    checkDouble("tauH", 6);
    checkDouble("Tt", 7);
    checkDouble("tauT", 8);
    checkDouble("ct", 9);
    checkDouble("tauc", 10);
    checkDouble("Rt", 11);
    checkDouble("tauR", 12);
    checkDouble("Qt", 13);
    checkDouble("tauQ", 14);
    checkDouble("a0", 15);
    checkDouble("P0", 16);

    // Get inputs & outputs
    mat y    = armaGetPr(prhs[0]);
    cube Zt   = getMaybeCube(prhs[1]);
    mat tauZ = armaGetPr(prhs[2]);
    mat dt   = armaGetPr(prhs[3]);
    mat taud = armaGetPr(prhs[4]);
    cube Ht    = getMaybeCube(prhs[5]);
    mat tauH = armaGetPr(prhs[6]);
    cube Tt   = getMaybeCube(prhs[7]);
    mat tauT = armaGetPr(prhs[8]);
    mat ct   = armaGetPr(prhs[9]);
    mat tauc = armaGetPr(prhs[10]);
    cube  Rt   = getMaybeCube(prhs[11]);
    mat tauR = armaGetPr(prhs[12]);
    cube  Qt   = getMaybeCube(prhs[13]);
    mat tauQ = armaGetPr(prhs[14]);
    mat a0   = armaGetPr(prhs[15]);
    mat P0   = armaGetPr(prhs[16]);
        
    int p    = y.n_rows;
    int n    = y.n_cols;
    int m    = Tt.n_rows;
    
    // Initialize new matricies
    mat LogL = mat(p, n, fill::zeros);
    plhs[0] = armaCreateMxMatrix(p, n);
    
    mat a  = mat(m, n+1, fill::zeros);
    plhs[1] = armaCreateMxMatrix(m, n+1);

    cube P = cube(m, m, n+1, fill::zeros);
    plhs[2] = armaCreateMxMatrix(m, m, n+1);

    mat v  = mat(p, n, fill::zeros);
    plhs[3] = armaCreateMxMatrix(p, n);

    mat F  = mat(p, n, fill::zeros);
    plhs[4] = armaCreateMxMatrix(p, n);

    cube M = cube(m, p, n, fill::zeros);
    plhs[5] = armaCreateMxMatrix(m, p, n);

    cube K = cube(m, p, n, fill::zeros);
    plhs[6] = armaCreateMxMatrix(m, p, n);

    cube L = cube(m, m, n, fill::zeros);
    plhs[7] = armaCreateMxMatrix(m, m, n);

    // Compute
    try
    {
        kfilt(y, Zt, tauZ, dt, taud, Ht, tauH, 
            Tt, tauT, ct, tauc, Rt, tauR, Qt, tauQ,
            a0, P0, LogL, a, P, v, F, M, K, L);
    }
    catch (int e)
    {
        mexErrMsgIdAndTxt("kfilter:unknown", "An unknown error occured.");
    }
    
    // Set outputs
    armaSetPr(plhs[0], LogL);
    armaSetPr(plhs[1], a);
    armaSetCubeData(plhs[2], P);
    armaSetPr(plhs[3], v);
    armaSetPr(plhs[4], F);
    armaSetCubeData(plhs[5], M);
    armaSetCubeData(plhs[6], K);
    armaSetCubeData(plhs[7], L);
  
    
    return;
}
