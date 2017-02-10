/* Filter using exact initial conditions

See "Fast Filtering and Smoothing for Multivariate State Space Models",
Koopman & Durbin (2000).

Copyright: David Kelley, 2017. 
*/

#ifndef FILTER_UNI_MEX_M_HPP
#define FILTER_UNI_MEX_M_HPP

#include "mex.h"
#include "armadillo.hpp"
using namespace arma ;


struct _Tau {
  vec Z, d, H, T, c, R, Q;
};

struct _Ss {
  cube Z, H, T, R, Q;
  mat d, c, a0, A0, R0, Q0;
  _Tau tau;
};

struct _filter {
  int dt;
  double logli;
  mat a, v, F, Fd;
  cube P, Pd, K, Kd;
};

const double PI = 3.1415926535897931;

_filter filter_uni_mex(mat y, cube Z, mat d, cube H, cube T, mat c, cube R, cube Q, 
  mat a0, mat A0, mat R0, mat Q0, _Tau tau) {

  cube K, Kd, Kstar, P, Pd, Pstar;
  double logli;
  int dt, ii;
  size_t iInd;
  mat F, Fd, Fstar, LogL, Pti, Pstarti, Pdti, Tii, a, ati, v;
  rowvec Zjj;
  uvec ind;
  uword jj;

  vec tauZ, taud, tauH, tauT, tauc, tauR, tauQ;
  tauZ = tau.Z;   taud = tau.d;   tauH = tau.H;
  tauT = tau.T;   tauc = tau.c;   tauR = tau.R;   tauQ = tau.Q;
  
  int p = Z.n_rows;
  int m = Z.n_cols;
  int n = y.n_cols;
  
  // assert(isdiag(H), 'Univarite filter requires diagonal H.');
  

  // Preallocate
  // Note Pd is the "diffuse" P matrix (P_\infty).
  a = zeros(m, n+1) ;
  v = zeros(p, n) ;
  
  Pd = zeros(m, m, n+1) ;
  Pstar = zeros(m, m, n+1) ;
  Fd = zeros(p, n) ;
  Fstar = zeros(p, n) ;
  Kd = zeros(m, p, n) ;
  Kstar = zeros(m, p, n) ;

  LogL = zeros(p, n) ;

  // Initialize
  ii = 0;
  // Tii = T(:,:,tauT(ii+1));
  Tii = T.slice((uword) tauT(ii)-1);
  // a(:,ii+1) = Tii * a0 + c(:,tauc(ii+1));
  a.col(ii) = Tii * a0 + c.col((uword) tauc(ii)-1);

  // Pd0 = A0 * A0';
  // Pstar0 = R0 * Q0 * R0';
  // Pd(:,:,ii+1)  = Tii * Pd0 * Tii';
  // Pstar(:,:,ii+1) = Tii * Pstar0 * Tii' + ...
  //   R(:,:,tauR(ii+1)) * Q(:,:,tauQ(ii+1)) * R(:,:,tauR(ii+1))';
  Pd.slice(0) = Tii * (A0 * trans(A0)) * trans(Tii);
  Pstar.slice(0) = Tii * (R0 * Q0 * trans(R0)) * trans(Tii) + 
    R.slice((uword) tauR(ii)-1) * Q.slice((uword) tauQ(ii)-1) * trans(R.slice((uword) tauR(ii)-1));

  // Initial recursion
  mat zeroMat = zeros(m, m);
  while (! approx_equal(Pd.slice(ii), zeroMat, "absdiff", 0.00000001)) {
    if (ii>=n) {
      // error('Degenerate model. Exact initial filter unable to transition to standard filter.');
      mexErrMsgIdAndTxt("filter_uni:degenerate", 
        "Degenerate model. Exact initial filter unable to transition to standard filter.");
    }

    ii = ii+1;
    // ind = find( ~isnan(y(:,ii)) )
    ind = find_finite(y.col(ii-1)) + 1;

    // ati = a(:,ii)
    ati = a.col(ii-1);

    // Pstarti = Pstar(:,:,ii)
    Pstarti = Pstar.slice(ii-1);
    // Pdti = Pd(:,:,ii)
    Pdti = Pd.slice(ii-1);
    
    for (iInd=1; iInd <= ind.n_elem; iInd++) {
      // jj = ind(iInd)
      jj = ind(iInd-1) - 1;     // Corrected to 0-based index

      // Zjj = Z(jj,:,tauZ(ii))
      Zjj = Z.slice((uword) tauZ(ii-1)-1).row(jj);

      // v(jj,ii) = y(jj, ii) - Zjj * ati - d(jj,taud(ii))
      v(jj, ii-1) = y(jj, ii-1) - as_scalar(Zjj * ati) - as_scalar(d(jj, (uword) taud(ii-1)-1));

      // Fd(jj,ii) = Zjj * Pdti * Zjj'
      Fd(jj, ii-1) = as_scalar(Zjj * Pdti * trans(Zjj));
      // Fstar(jj,ii) = Zjj * Pstarti * Zjj' + H(jj,jj,tauH(ii))
      Fstar(jj, ii-1) = as_scalar(Zjj * Pstarti * trans(Zjj) + H(jj, jj, (uword) tauH(ii-1)-1));

      // Kd(:,jj,ii) = Pdti * Zjj'
      Kd.slice(ii-1).col(jj) = Pdti * trans(Zjj);
      // Kstar(:,jj,ii) = Pstarti * Zjj'
      Kstar.slice(ii-1).col(jj) = Pstarti * trans(Zjj);
      
      if (Fd(jj, ii-1) != 0) {
        // F diffuse nonsingular
        // ati = ati + Kd(:,jj,ii) ./ Fd(jj,ii) * v(jj,ii)
        ati = ati + Kd.slice(ii-1).col(jj) / Fd(jj,ii-1) * v(jj,ii-1);

        // Pstarti = Pstarti + Kd(:,jj,ii) * Kd(:,jj,ii)' * Fstar(jj,ii) * (Fd(jj,ii).^-2) - 
        //          (Kstar(:,jj,ii) * Kd(:,jj,ii)' + Kd(:,jj,ii) * Kstar(:,jj,ii)') ./ Fd(jj,ii)
        Pstarti = Pstarti + Kd.slice(ii-1).col(jj) * trans(Kd.slice(ii-1).col(jj)) * 
        Fstar(jj, ii-1) * pow(Fd(jj, ii-1), -2) - 
        (Kstar.slice(ii-1).col(jj) * trans(Kd.slice(ii-1).col(jj)) + 
          Kd.slice(ii-1).col(jj) * trans(Kstar.slice(ii-1).col(jj)) ) / 
        Fd(jj, ii-1);

        // Pdti = Pdti - Kd(:,jj,ii) .* Kd(:,jj,ii)' ./ Fd(jj,ii)
        Pdti = Pdti - Kd.slice(ii-1).col(jj) * trans(Kd.slice(ii-1).col(jj)) / Fd(jj, ii-1);
        
        // LogL(jj,ii) = log(Fd(jj,ii))
        LogL(jj, ii-1) = log(Fd(jj, ii-1));
      }
      else {
        // F diffuse = 0
        // ati = ati + Kstar(:,jj,ii) ./ Fstar(jj,ii) * v(jj,ii)
        ati = ati + Kstar.slice(ii-1).col(jj) / Fstar(jj, ii-1) * v(jj, ii-1);

        // Pstarti = Pstarti - Kstar(:,jj,ii) ./ Fstar(jj,ii) * Kstar(:,jj,ii)'
        Pstarti = Pstarti - Kstar.slice(ii-1).col(jj) / Fstar(jj, ii-1) * 
        trans(Kstar.slice(ii-1).col(jj));

        // LogL(jj,ii) = (log(Fstar(jj,ii)) + (v(jj,ii)^2) ./ Fstar(jj,ii))
        LogL(jj, ii-1) = (log(Fstar(jj, ii-1)) + (pow(v(jj, ii-1), 2)) / Fstar(jj, ii-1));
      }
    }

    // Tii = T(:,:,tauT(ii+1))
    Tii = T.slice((uword) tauT(ii)-1);

    // a(:,ii+1) = Tii * ati + c(:,tauc(ii+1))
    a.col(ii) = Tii * ati + c.col((uword) tauc(ii)-1);

    // Pd(:,:,ii+1)  = Tii * Pdti * Tii'
    Pd.slice(ii) = Tii * Pdti * trans(Tii);

    // Pstar(:,:,ii+1) = Tii * Pstarti * Tii' + ...
    //     R(:,:,tauR(ii+)) * Q(:,:,tauQ(ii+1)) * R(:,:,tauR(ii+1))'
    Pstar.slice(ii) = Tii * Pstarti * trans(Tii) + 
      R.slice((uword) tauR(ii)-1) * Q.slice((uword) tauQ(ii)-1) * trans(R.slice((uword) tauR(ii)-1));
  }
 
  dt = ii;
  F = Fstar;
  K = Kstar;
  P = Pstar;

  // Standard Kalman filter recursion
  for (ii=dt+1; ii<=n; ii++) {
    // ind = find( ~isnan(y(:,ii)) )
    ind = find_finite(y.col(ii-1)) + 1;

    ati = a.col(ii-1);
    Pti = P.slice(ii-1);

    for (iInd=1; iInd<= ind.n_elem; iInd++) {
        // jj = ind(iInd)
      jj = ind(iInd-1) - 1;

        // Zjj = Z(jj,:,tauZ(ii))
      Zjj = Z.slice((uword) tauZ(ii-1)-1).row(jj);

        // v(jj,ii) = y(jj,ii) - Zjj * ati - d(jj,taud(ii))
      v(jj, ii-1) = y(jj, ii-1) - as_scalar(Zjj * ati) - as_scalar(d(jj, (uword) taud(ii-1)-1));

        // F(jj,ii) = Zjj * Pti * Zjj' + H(jj,jj,tauH(ii))
      F(jj, ii-1) = as_scalar(Zjj * Pti * trans(Zjj) + H(jj, jj, (uword) tauH(ii-1)-1));

        // M(:,jj,ii) = Pti * Zjj'
      K.slice(ii-1).col(jj) = Pti * trans(Zjj);

        // LogL(jj,ii) = (log(F(jj,ii)) + (v(jj,ii)^2) / F(jj,ii))
      LogL(jj, ii-1) = (log(F(jj, ii-1)) + (pow(v(jj, ii-1), 2)) / F(jj, ii-1));

        // ati = ati + M(:,jj,ii) / F(jj,ii) * v(jj,ii)
      ati = ati + K.slice(ii-1).col(jj) / F(jj, ii-1) * v(jj, ii-1);

        // Pti = Pti - M(:,jj,ii) / F(jj,ii) * M(:,jj,ii)'
      Pti = Pti - K.slice(ii-1).col(jj) / F(jj, ii-1) * trans(K.slice(ii-1).col(jj));
    }

      // Tii = T(:,:,tauT(ii))
    Tii = T.slice((uword) tauT(ii)-1);

      // a(:,ii+1) = Tii * ati + c(:,tauc(ii))
    a.col(ii) = Tii * ati + c.col((uword) tauc(ii)-1);
      // P(:,:,ii+1) = Tii * Pti * Tii' + ...
      //     R(:,:,tauR(ii)) * Q(:,:,tauQ(ii)) * R(:,:,tauR(ii))'
    P.slice(ii) = Tii * Pti * trans(Tii) + 
    R.slice((uword) tauR(ii)-1) * Q.slice((uword) tauQ(ii)-1) * trans(R.slice((uword) tauR(ii)-1));
  }

  // logli = -(0.5 * sum(sum(isfinite(y)))) * log(2 * pi) - 0.5 * sum(sum(LogL))
  uvec finite_y = find_finite(y);
  logli = -(0.5 * finite_y.n_elem) * log(2 * PI) - 0.5 * sum(sum(LogL)) ;

    // Compile output
  _filter output;
  output.dt = dt;
  output.logli = logli;
  output.a = a;
  output.P = P;
  output.Pd = Pd;
  output.v = v;
  output.F = F;
  output.Fd = Fd;
  output.K = K;
  output.Kd = Kd;
  return output;
}

#endif
