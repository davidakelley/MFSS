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
 
void kfiltsmooth(
        mat y, cube Zt, vec tauZ, mat dt, vec taud, cube Tt, vec tauT, mat ct, vec tauc, 
        cube Qt, vec tauQ, cube Rt, vec tauR, cube Ht, vec tauH, vec a0, mat P0,
        bool UseSmoother, double &LogLikelihood, mat &alpha, mat &a0tilde, mat &a,
        mat &eta, cube &V, cube &J, vec &LogL, mat &epsilon) 
{
    // Rest of function call : bool compOutStru, vec &sLastRaw, mat &pLast)
    unsigned int p    = y.n_rows;
    unsigned int n    = y.n_cols;
    unsigned int m    = a0.n_rows;
    unsigned int ii, jj;				// Loop indexes
    unsigned int indJJ;
    
    //Preallocate
    // % v = zeros(p,n);
    // % F = zeros(p,n);
    // % M = zeros(m,p,n);
    // % P = zeros(m,m,n+1);
    // % a = zeros(m,n+1);
    // % LogL = zeros(p,n);
    mat v = mat(p, n, fill::zeros);
    cube F = cube(p, p, n, fill::zeros);
    cube M = cube(m, p, n, fill::zeros);
    cube P = cube(m, m, n+1, fill::zeros);
    cube K = cube(m, p, n, fill::zeros); //Potentially gXpXn
    cube L = cube(m, m, n, fill::zeros); //Not sure why
    mat u = mat(p, n, fill::zeros);

    mat r = mat(m, n+1, fill::zeros);
    cube N = cube(m,m,n+1, fill::zeros);

    // Set up for Kalman filter
    // % P0 = kappa*eye(m); 
    // % a(:,1) = Tt(:,:,tauT(1))*a0+ ct(:,tauc(1));
    // % P(:,:,1) = Tt(:,:,tauT(1))*P0*Tt(:,:,tauT(1))'...
    // %     +Rt(:,:,tauR(1))*Qt(:,:,tauQ(1))*Rt(:,:,tauR(1))';
//     mat P0 = kappa * eye(m,m);
       
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
    cube Finv = cube(p, p, n, fill::zeros);

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
        LogL(ii) = -0.5 * (log(det(F.slice(ii).submat(ind,ind))) \
            + as_scalar(v(ind,iiVec).t() * Finv.slice(ii).submat(ind,ind) * v(ind,iiVec)));
        
        // % K(:,ind,ii) = Tt(:,:,tauT(ii+1))*P(:,:,ii)*...
        // %     (W*Zt(:,:,tauZ(ii)))'*Finv(ind,ind,ii);
        K.slice(ii).cols(ind) = Tt.slice(uint tauT(ii+1)-1) * P.slice(ii) \
            * (W * Zt.slice(uint tauZ(ii)-1)).t() * Finv.slice(ii).submat(ind,ind);

        // % L(:,:,ii) = Tt(:,:,tauT(ii+1)) - K(:,ind,ii)*(W*Zt(:,:,tauZ(ii)));
        L.slice(ii) = Tt.slice(uint tauT(ii+1)-1) \
            - K.slice(ii).cols(ind) * W * Zt.slice(uint tauZ(ii)-1); // Removed () around W*Zt

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

    // Compute Likelihood
    // % LogLikelihood = -(sum(sum(isfinite(y)))/2)*log(2*pi) + sum(sum(LogL));
    LogLikelihood = -(((uvec) find_finite(y)).n_elem/2.0) * log(2*datum::pi) + accu(LogL);

    // Run smoother
    if (UseSmoother) {
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
        
    } else {
        // % alpha = [];
        // % a0tilde = [];
        alpha.reset();
        a0tilde.reset();
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
    if(nrhs!=18 && nrhs!=19) {
        mexErrMsgIdAndTxt("kfilter:nrhs", "18 or 19 input required.");
    }

    // Input variables:
    // y, Zt, tauZ, dt, taud, ct, tauc, Qt, tauQ, Rt, tauR, Ht, tauH, Tt, tauT, a0, m, kappa, UseSmoother
            
    // Check input sizes
    checkDouble("y", 0);
    checkDouble("Zt", 1);
    checkDouble("tauZ", 2);
    checkDouble("dt", 3);
    checkDouble("taud", 4);
    checkDouble("ct", 5);
    checkDouble("tauc", 6);
    checkDouble("Qt", 7);
    checkDouble("tauQ", 8);
    checkDouble("Rt", 9);
    checkDouble("tauR", 10);
    checkDouble("Ht", 11);
    checkDouble("tauH", 12);
    checkDouble("Tt", 13);
    checkDouble("tauT", 14);
    checkDouble("a0", 15);
    checkDouble("P0", 16);
    if(!mxIsLogicalScalar(prhs[17])) {
        mexErrMsgIdAndTxt("kfilter:notLogical", "UseSmoother must be a logical scalar.");
    }
    if(nrhs==19) {
        if(!mxIsLogicalScalar(prhs[18])) {
            mexErrMsgIdAndTxt("kfilter:notLogical", "printDebug must be a logical scalar.");
        }
    }
    
    // Get inputs & outputs
    mat y    = armaGetPr(prhs[0]);
    cube Zt  = getMaybeCube(prhs[1]);
    vec tauZ = getColumn(prhs[2]);
    mat dt   = armaGetPr(prhs[3]);
    vec taud = getColumn(prhs[4]);
    cube Tt  = getMaybeCube(prhs[5]);
    vec tauT = getColumn(prhs[6]);
    mat ct   = armaGetPr(prhs[7]);
    vec tauc = getColumn(prhs[8]);
    cube  Qt = getMaybeCube(prhs[9]);
    vec tauQ = getColumn(prhs[10]);
    cube  Rt = getMaybeCube(prhs[11]);
    vec tauR = getColumn(prhs[12]);
    cube Ht  = getMaybeCube(prhs[13]);
    vec tauH = getColumn(prhs[14]);
    vec a0   = getColumn(prhs[15]);
	mat P0   = armaGetPr(prhs[16]);
    bool UseSmoother = (bool) mxGetLogicals(prhs[17]);
    bool printDebug;
    if(nrhs==19) {
        printDebug = (bool) mxGetLogicals(prhs[18]);
    }
    else {
        printDebug = false;
    }
    
    int n    = y.n_cols;
    int m    = a0.n_rows;
    int g    = Rt.n_cols;
    int p    = y.n_rows;
    
    // Initialize output matricies
    double LogLikelihood = 0.0;
    mat alpha = mat(m, n, fill::zeros);
    mat a0tilde = mat(m, 1, fill::zeros);
    
    bool compOutStru = (nlhs>3);
    mat a = mat(m, n+1, fill::zeros);
    mat eta = mat(g,n, fill::zeros);
    mat epsilon = mat(p,n, fill::zeros);
    cube V = cube(m,m,n, fill::zeros);
    cube J = cube(m,m,n, fill::zeros);
    
    vec LogL = vec(n, fill::zeros);
        
    // Compute
    try
    {
        kfiltsmooth(y, Zt, tauZ, dt, taud, Tt, tauT, ct, tauc, Qt, tauQ, Rt, tauR,
                Ht, tauH, a0, P0, UseSmoother, LogLikelihood, alpha, a0tilde, a, eta,
                V, J, LogL, epsilon);
    }
    catch (int e)
    {
        mexErrMsgIdAndTxt("kfilter:unknown", "An unknown error occured.");
    }
        
    // Set outputs
    plhs[0] = mxCreateDoubleScalar(LogLikelihood);
    if(nlhs>1) {
        plhs[1] = armaCreateMxMatrix(m, n);
        armaSetPr(plhs[1], alpha);
    }
    if(nlhs>2) {
        plhs[2] = armaCreateMxMatrix(m, 1);
        armaSetPr(plhs[2], a0tilde);
    }
    if(nlhs>3) {
        plhs[3] = armaCreateMxMatrix(m, n+1);
        armaSetPr(plhs[3], a);
    }
    if(nlhs>4) {
        plhs[4] = armaCreateMxMatrix(m, m, n);
        armaSetCubeData(plhs[4], V);
    }
    if(nlhs>5) {
        plhs[5] = armaCreateMxMatrix(m, m, n);
        armaSetCubeData(plhs[5], J);
    }
    if(nlhs>6) {
        plhs[6] = armaCreateMxMatrix(g, n);
        armaSetPr(plhs[6], eta);
    }
    if(nlhs>7) {
        plhs[7] = armaCreateMxMatrix(n, 1);
        armaSetPr(plhs[7], LogL);
    }
    if(nlhs>8) {
        plhs[8] = armaCreateMxMatrix(p, n);
        armaSetPr(plhs[8], epsilon);
    }
    return;
}
