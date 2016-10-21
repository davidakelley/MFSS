/*
 * ksmoother.cpp
 *
 * Computes the smoother part of the Kalman filter/smoother from
 *   GeneralizedKFilterSmootherUni.m
 *
 * To compile in MATLAB, run the make_ksmoother.m script.
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

void ksmooth(
        mat y, cube Zt, mat tauZ, cube Ht, mat tauH, 
        cube Tt, mat tauT, cube Rt, mat tauR, cube Qt, mat tauQ,
        mat a, cube P, mat v, mat F, cube M, cube L, 
        mat a0, mat P0, 
        mat &alpha, mat &eta, mat &r, cube &N, cube &V, cube &J)
{
    // loop indexes
    unsigned int ii, jj;
    uvec ind;
    unsigned int indJJ;

    // Dimensions
    unsigned int p = y.n_rows;
    unsigned int n = y.n_cols;
    unsigned int m = Tt.n_rows;
    // unsigned int g = Rt.n_cols;

    // Matricies
    mat rti(m, 1, fill::zeros);
    mat Nti(m, m, fill::zeros);
    mat Lti;

    // 
    mat eyeM = eye(m, m);
    mat iZt;
    
    // Compute
    for(ii=n-1; ii != (unsigned int) -1; --ii) {
        ind = find_finite(y.col(ii));
        for(jj=ind.n_elem-1; jj != (unsigned int) -1; --jj) {
            indJJ = ind(jj);
            
            iZt = Zt.slice(uint tauZ(ii)-1).row(indJJ);

            // Lti = eye(m) - M(:,ind(jj),ii)*Zt(ind(jj),:,tauZ(ii))/F(ind(jj),ii);
            Lti = eyeM - M.slice(ii).col(indJJ) * iZt / F(indJJ,ii);                    
            
            // rti = Zt(ind(jj),:,tauZ(ii))'/F(ind(jj),ii)*v(ind(jj),ii) + Lti'*rti;
            rti = iZt.t() / F(indJJ,ii) * v(indJJ,ii) + Lti.t() * rti;
    
            // Nti = Zt(ind(jj),:,tauZ(ii))'/F(ind(jj),ii)*Zt(ind(jj),:,tauZ(ii))...
            //  + Lti'*Nti*Lti;
            Nti = iZt.t() / F(indJJ,ii) * iZt + Lti.t() * Nti * Lti;
        }        
        
        // r(:,ii) = rti;
        r.col(ii) = rti;
        
        // N(:,:,ii) = Nti;
        N.slice(ii) = Nti;
        
        // alpha(:,ii) = a(:,ii) + P(:,:,ii)*r(:,ii);
        alpha.col(ii) = a.col(ii) + P.slice(ii) * r.col(ii);
        
       // eta(:,ii)    = Qt(:,:,tauQ(ii))*Rt(:,:,tauR(ii))'*r(:,ii);
        eta.col(ii) = Qt.slice(uint tauQ(ii)-1) * Rt.slice(uint tauR(ii)-1).t() * r.col(ii);
                 
        // V(:,:,ii) = P(:,:,ii) - P(:,:,ii)*N(:,:,ii)*P(:,:,ii);
        // FIXME? I feel like the last P here should be transposed.
        V.slice(ii) = P.slice(ii) - P.slice(ii) * N.slice(ii) * P.slice(ii);
        
        // J(:,:,ii) = P(:,:,ii)*L(:,:,ii)'*(eye(m)-N(:,:,ii+1)*P(:,:,ii+1));
        J.slice(ii) = P.slice(ii) * L.slice(ii).t() \
                * (eyeM - N.slice(ii+1) * P.slice(ii+1));
                
        // rti = Tt(:,:,tauT(ii))'*rti;
        rti = Tt.slice(uint tauT(ii)-1).t() * rti;
        
        // Nti = Tt(:,:,tauT(ii))'*Nti*Tt(:,:,tauT(ii));    
        Nti = Tt.slice(uint tauT(ii)-1).t() * Nti * Tt.slice(uint tauT(ii)-1);
        
    } 
    
    // a0tilde = a0 + P0*rti;
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
    if(nrhs!=19) {
        mexErrMsgIdAndTxt("ksmoother_uni:nrhs", "Nineteen input required.");
    }
    if(nlhs!=6) {
        mexErrMsgIdAndTxt("ksmoother_uni:nlhs", "Six output required.");
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
    checkDouble("F", 14);
    checkDouble("M", 15);
    checkDouble("L", 16);
    checkDouble("a0", 17);
    checkDouble("P0", 18);
    
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
    mat F    = armaGetPr(prhs[14]);
    cube M   = getMaybeCube(prhs[15]);
    cube L   = getMaybeCube(prhs[16]);
    mat a0   = armaGetPr(prhs[17]);
    mat P0   = armaGetPr(prhs[18]);
    
    int n    = y.n_cols;
    int m    = Tt.n_rows;
    int g    = Rt.n_cols;
    
    // Initialize output matricies
    mat alpha(m, n, fill::zeros);
    plhs[0] = armaCreateMxMatrix(m, n);

    mat eta(g, n, fill::zeros);
    plhs[1] = armaCreateMxMatrix(g, n);

    mat r(m, n, fill::zeros);
    plhs[2] = armaCreateMxMatrix(m, n);
    
    cube N(m, m, n+1, fill::zeros);
    plhs[3] = armaCreateMxMatrix(m, m, n+1);
    
    cube V(m, m, n, fill::zeros);
    plhs[4] = armaCreateMxMatrix(m, m, n);

    cube J(m, m, n, fill::zeros);
    plhs[5] = armaCreateMxMatrix(m, m, n);
    
    
    // Compute
    try
    {
        ksmooth(y, Zt, tauZ, Ht, tauH, Tt, tauT, Rt, tauR, Qt, tauQ, 
                a, P, v, F, M, L, a0, P0, alpha, eta, r, N, V, J);
    }
    catch (int e)
    {
        mexErrMsgIdAndTxt("ksmoother_uni:unknown", "An unknown error occured.");
    }
    
    // Set outputs
    armaSetPr(plhs[0], alpha);
    armaSetPr(plhs[1], eta);
    armaSetPr(plhs[2], r);
    armaSetCubeData(plhs[3], N);
    armaSetCubeData(plhs[4], V);
    armaSetCubeData(plhs[5], J);
    
    return;
}
