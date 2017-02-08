/* Smoother using exact initial conditions

See "Fast Filtering and Smoothing for Multivariate State Space Models",
Koopman & Durbin (2000).

Copyright: David Kelley, 2017. 
*/

#ifndef SMOOTHER_UNI_MEX_M_HPP
#define SMOOTHER_UNI_MEX_M_HPP

#include "mex.h"
#include "armadillo.hpp"
using namespace arma;


struct _Tau {
  vec Z, d, H, T, c, R, Q;
};

struct _Ss {
  cube Z, H, T, R, Q;
  mat d, c, a0, A0, R0, Q0;
  _Tau tau;
};

struct _Fout {
  cube P, Pd, K, Kd;
  mat a, v, F, Fd;
  int dt;
};

struct _smoother {
  mat alpha, eta, r;
  cube N;
  vec a0tilde;
};

_smoother smoother_uni_mex(mat y, cube Z, mat d, cube H, cube T, mat c, cube R, cube Q, 
  mat a0, mat A0, mat R0, mat Q0, _Tau tau, _Fout fOut) {

  cube N, P, Pd, K, Kd;
  mat alpha, eta, r, rti, Nti, Lti, r0, r1, r0ti, r1ti, Ldti, Lstarti, L0ti, a, v, F, Fd, Pstar0, Pd0;
  vec a0tilde;

  int dt;

  int ii;
  size_t iInd;
  rowvec Zjj;
  mat Tii;
  uvec ind;
  uword jj;

  vec tauZ, taud, tauH, tauT, tauc, tauR, tauQ;
  tauZ = tau.Z;   taud = tau.d;   tauH = tau.H;
  tauT = tau.T;   tauc = tau.c;   tauR = tau.R;   tauQ = tau.Q;
  
  a = fOut.a;
  P = fOut.P;
  Pd = fOut.Pd;
  v = fOut.v;
  F = fOut.F;
  Fd = fOut.Fd;
  K = fOut.K;
  Kd = fOut.Kd;
  dt = fOut.dt;

  int p = Z.n_rows;
  int m = Z.n_cols;
  int n = y.n_cols;
  int g = Q.n_cols;

  // assert(isdiag(H), 'Univarite filter requires diagonal H.');

  mat eyeM = eye(m, m);

  // Preallocate
  alpha = zeros(m, n);
  eta = zeros(g, n);
  r = zeros(m, n);
  N = zeros(m, m, n+1);

  rti = zeros(m, 1);
  Nti = zeros(m, m);

  // Standard univariate Kalman smoother
  for (ii = n; ii > dt; --ii) {

    ind = find_finite(y.col(ii-1)) + 1;

    for (iInd = ind.n_elem; iInd > 0; --iInd) {
      jj = ind(iInd-1) - 1;       // Corrected to be 0-based
      Zjj = Z.slice((uword) tauZ(ii-1)-1).row(jj);

      Lti = eyeM - K.slice(ii-1).col(jj) * Zjj / F(jj,ii-1);
      
      rti = trans(Zjj) / F(jj,ii-1) * v(jj,ii-1) + trans(Lti) * rti;

      Nti = trans(Zjj) / F(jj,ii-1) * Zjj + trans(Lti) * Nti * Lti;
    }
    r.col(ii-1) = rti;
    N.slice(ii-1) = Nti;

    alpha.col(ii-1) = a.col(ii-1) + P.slice(ii-1) * r.col(ii-1);
    eta.col(ii-1) = Q.slice((uword) tauQ(ii-1) - 1) * trans(R.slice((uword) tauR(ii-1) - 1)) * r.col(ii-1);

    Tii = T.slice((uword) tauT(ii-1) - 1);
    rti = trans(Tii) * rti;
    Nti = trans(Tii) * Nti * Tii;
  }

  r0 = r;
  r1 = zeros(m, dt+1);

  // Initial values smoother
  for (ii=dt; ii > 0; --ii) {
    r0ti = r0.col(ii);      // Yes, ii. Was ii+1 in Matlab.
    r1ti = r1.col(ii);      // Yes, ii. Was ii+1 in Matlab.

    ind = find_finite(y.col(ii-1));

    for (iInd = ind.n_elem; iInd > 0; --iInd) {
      jj = ind(iInd-1);       // Note 0-based
      Zjj = Z.slice((uword) tauZ(ii-1)-1).row(jj);

      if (Fd(jj,ii-1) != 0) {
        // Diffuse case
        Ldti = eyeM - Kd.slice(ii-1).col(jj) * Zjj / Fd(jj,ii-1);
        L0ti = (Kd.slice(ii-1).col(jj) * F(jj,ii-1) / Fd(jj, ii-1) + K.slice(ii-1).col(jj)) * Zjj / Fd(jj,ii-1);

        r1ti = trans(Zjj) / Fd(jj,ii-1) * v(jj,ii-1) - trans(L0ti) * r0ti + trans(Ldti) * r1ti;

        r0ti = trans(Ldti) * r0ti;
      }
      else {
        // Known forecast variance
        Lstarti = eyeM - K.slice(ii-1).col(jj) * Zjj / F(jj,ii-1);
        r0ti = trans(Zjj) / F(jj,ii-1) * v(jj,ii-1) + trans(Lstarti) * r0ti;
      }
    }

    // Move to next period
    r0.col(ii-1) = r0ti;
    r1.col(ii-1) = r1ti;

    alpha.col(ii-1) = a.col(ii-1) + P.slice(ii-1) * r0.col(ii-1) + Pd.slice(ii-1) * r1.col(ii-1);

    eta.col(ii-1) = Q.slice((uword) tauQ(ii-1) - 1) * trans(R.slice((uword) tauR(ii-1) - 1)) * r0.col(ii-1);

    r0ti = trans(T.slice((uword) tauT(ii-1) - 1))  * r0ti;
    r1ti = trans(T.slice((uword) tauT(ii-1) - 1))  * r1ti;
  }

  Pstar0 = R0 * Q0 * trans(R0);
  if (dt > 0) {
    Pd0 = A0 * trans(A0);
    a0tilde = a0 + Pstar0 * r0ti + Pd0 * r1ti;
  }
  else {
    a0tilde = a0 + Pstar0 * rti;
  }

  // Compile output
  _smoother output;
  output.alpha = alpha;
  output.eta = eta;
  output.r = r;
  output.N = N;
  output.a0tilde = a0tilde;
  return output;
}

#endif
