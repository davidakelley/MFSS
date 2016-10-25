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

void kfilt_multi(
        mat y, cube Zt, mat tauZ, mat dt, mat taud, cube Ht, mat tauH, 
        cube Tt, mat tauT, mat ct, mat tauc, cube Rt, mat tauR, cube Qt, mat tauQ, 
        mat a0, mat P0, mat &LogL, mat &a, cube &P, 
        mat &v, cube &F, cube &M, cube &K, cube &L, mat &w, cube &Finv)
{         
// void kfilt_multi(
//         mat y, cube Zt, vec tauZ, mat dt, vec taud, cube Tt, vec tauT, mat ct, vec tauc, 
//         cube Qt, vec tauQ, cube Rt, vec tauR, cube Ht, vec tauH, vec a0, mat P0,
//         bool UseSmoother, double &LogLikelihood, mat &alpha, mat &a0tilde, mat &a,
//         mat &eta, cube &V, cube &J, vec &LogL, mat &epsilon) 
    // Rest of function call : bool compOutStru, vec &sLastRaw, mat &pLast)
    unsigned int p    = y.n_rows;
    unsigned int n    = y.n_cols;
    unsigned int m    = a0.n_rows;
    unsigned int ii;				// Loop index
    
    // Set up for Kalman filter
    // % P0 = kappa*eye(m); 
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
        
    // Run filter
    uvec ind;
    
    mat W;
    mat eyeP = eye(p, p);
    
    mat PSVD;
    mat DSVD;
    mat PSVDinv;
    vec DSVDvec;
    mat tempF;

    double tolZero = 1.0e-12;
    uvec firstZero;
    uword firstZeroInd;
    
    double logDetF;
    uvec iiVec(1);

    // Multivariate Kalman filter
    for (ii=0; ii < n; ii++) {
        // Create W matrix for possible missing values
        // % W = eye(p);
        // % ind = ~isnan(y(:,ii));
        // % W = W((ind==1),:);
        ind = find_finite(y.col(ii));
        W = eyeP.rows(ind);
    
        // % v(ind,ii) = y(ind,ii) - (W*Zt(:,:,tauZ(ii)))*a(:,ii)-W*dt(:,taud(ii));
        iiVec.fill(ii);
        v.submat(ind,iiVec) = y.submat(ind, iiVec) \
                - (W*Zt.slice(uint tauZ(ii)-1)) * a.col(ii) \
                - W * dt.col(uint taud(ii)-1);
        
        // % F(ind,ind,ii) = (W*Zt(:,:,tauZ(ii)))*P(:,:,ii)*(W*Zt(:,:,tauZ(ii)))' ...
        // %     + W*Ht(:,:,tauH(ii))*W';
        F.slice(ii).submat(ind,ind) = (W*Zt.slice(uint tauZ(ii)-1)) * P.slice(ii) \
                * (W * Zt.slice(uint tauZ(ii)-1)).t() \
                + W * Ht.slice(uint tauH(ii)-1) * W.t();

        // % % Finv trick
        // % %     if isempty(Ht)==false
        // %     tempF = F(ind,ind,ii);
        // % %     else
        // % %         tempF= Zt*P0*Zt';
        // % %     end
        
        tempF = F.slice(ii).submat(ind,ind);

        // % tempF = 0.5*(tempF+tempF');
        tempF = 0.5 * (tempF+tempF.t());
        
        // % [PSVD,DSDV,PSVDinv] = svd(tempF);
        svd(PSVD, DSVDvec, PSVDinv, tempF);

        // % % 1.b Truncate small singular values
        // % tolZero=1e-12;
        // % firstZero=min(find(diag(DSDV) < tolZero));
        // % if isempty(firstZero)==false
        // %     PSVD=PSVD(:,1:firstZero-1);
        // %     PSVDinv=PSVDinv(:,1:firstZero-1);
        // %     DSDV=DSDV(1:firstZero-1,1:firstZero-1);
        // % end
        // % Finv(ind,ind,ii) = PSVD*(DSDV\eye(length(DSDV)))*PSVDinv';

        firstZero = find(DSVDvec < tolZero, 1);
        if (firstZero.n_elem > 0) {
            // What if firstZero is 0?
            firstZeroInd = firstZero(0) - 1;
            PSVD = PSVD.cols(0,firstZeroInd); 
            PSVDinv = PSVDinv.cols(0,firstZeroInd);
            DSVDvec = DSVDvec.subvec(0,firstZeroInd);
        }
        DSVD = diagmat(DSVDvec);

//         invDSVD = inv(DSVD);
        Finv.slice(ii).submat(ind,ind) = PSVD * inv(DSVD) * PSVDinv.t();

        // % logDetF=sum(log(diag(DSDV)));
        logDetF = sum(log(DSVDvec));
        
        // Resume multivariate filter
        // %  LogL(ii) = -1/2*(log(det(F(ind,ind,ii)))+...
        // %    v(ind,ii)'*Finv(ind,ind,ii)*v(ind,ii));
        LogL(ii) = logDetF \
            + as_scalar(v(ind,iiVec).t() * Finv.slice(ii).submat(ind,ind) * v(ind,iiVec));
        
        M.slice(ii).cols(ind) = P.slice(ii) \
            * (W * Zt.slice(uint tauZ(ii)-1)).t() * Finv.slice(ii).submat(ind,ind);

        // % K(:,ind,ii) = Tt(:,:,tauT(ii+1))*P(:,:,ii)*...
        // %     (W*Zt(:,:,tauZ(ii)))'*Finv(ind,ind,ii);
        K.slice(ii).cols(ind) = Tt.slice(uint tauT(ii+1)-1) * M.slice(ii).cols(ind);

        // % L(:,:,ii) = Tt(:,:,tauT(ii+1)) - K(:,ind,ii)*(W*Zt(:,:,tauZ(ii)));
        L.slice(ii) = Tt.slice(uint tauT(ii+1)-1) \
            - K.slice(ii).cols(ind) * W * Zt.slice(uint tauZ(ii)-1); // Removed () around W*Zt

        w.submat(ind, iiVec) = Finv.slice(ii).submat(ind,ind) * v.submat(ind, iiVec);

        // % a(:,ii+1) = Tt(:,:,tauT(ii+1))*a(:,ii)+...
        // %     ct(:,tauc(ii+1))+K(:,ind,ii)*v(ind,ii);
        a.col(ii+1) = Tt.slice(uint tauT(ii+1)-1) * a.col(ii) \
            + ct.col(uint tauc(ii+1)-1) + K.slice(ii).cols(ind) * v(ind,iiVec);

        // % P(:,:,ii+1) = Tt(:,:,tauT(ii+1))*P(:,:,ii)*L(:,:,ii)'+ ...
        // %     Rt(:,:,tauR(ii+1))*Qt(:,:,tauQ(ii+1))*Rt(:,:,tauR(ii+1))';
        P.slice(ii+1) = Tt.slice(uint tauT(ii+1)-1) \
            * P.slice(ii) * L.slice(ii).t() \
            + Rt.slice(uint tauR(ii+1)-1) * Qt.slice(uint tauQ(ii+1)-1) \
            * Rt.slice(uint tauR(ii+1)-1).t();
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
    if(nrhs!=17) {
        mexErrMsgIdAndTxt("kfilter:nrhs", "Seventeen input required.");
    }
    if(nlhs!=10) {
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

    cube F  = cube(p, p, n, fill::zeros);
    plhs[4] = armaCreateMxMatrix(p, p, n);

    cube M = cube(m, p, n, fill::zeros);
    plhs[5] = armaCreateMxMatrix(m, p, n);

    cube K = cube(m, p, n, fill::zeros);
    plhs[6] = armaCreateMxMatrix(m, p, n);

    cube L = cube(m, m, n, fill::zeros);
    plhs[7] = armaCreateMxMatrix(m, m, n);

    mat w = mat(p, n, fill::zeros);
    plhs[8] = armaCreateMxMatrix(p, n);

    cube Finv = cube(p, p, n, fill::zeros);
    plhs[9] = armaCreateMxMatrix(p, p, n);

    // Compute
    try
    {
        kfilt_multi(y, Zt, tauZ, dt, taud, Ht, tauH, 
            Tt, tauT, ct, tauc, Rt, tauR, Qt, tauQ,
            a0, P0, LogL, a, P, v, F, M, K, L, w, Finv);
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
    armaSetCubeData(plhs[4], F);
    armaSetCubeData(plhs[5], M);
    armaSetCubeData(plhs[6], K);
    armaSetCubeData(plhs[7], L);
    armaSetPr(plhs[8], w);
    armaSetCubeData(plhs[9], Finv);
  
}
