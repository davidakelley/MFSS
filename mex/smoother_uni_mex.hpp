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
  cube V, N;
  vec a0tilde;
};

_smoother smoother_uni_mex(mat y, cube Z, mat d, cube H, cube T, mat c, cube R, cube Q, 
  mat a0, mat A0, mat R0, mat Q0, _Tau tau, _Fout fOut) {

  cube N, P, Pd, K, Kd, V;
  mat alpha, eta, r, rti, Nti, Lti, r1, r0ti, r1ti, N0ti, N1ti, N2ti, Ldti, Lstarti, L0ti, a, v, F, Fd, Pstar0, Pd0, Ntemp;
  vec a0tilde;

  int dt;

  int ii;
  size_t iInd;
  rowvec Zti;
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
  N = zeros(m, m, n);
  V = zeros(m, m, n);

  rti = zeros(m, 1);
  Nti = zeros(m, m);

  // Standard univariate Kalman smoother
  for (ii = n; ii > dt; --ii) {

    ind = find_finite(y.col(ii-1)) + 1;

    for (iInd = ind.n_elem; iInd > 0; --iInd) {
      jj = ind(iInd-1) - 1;       // Corrected to be 0-based
      Zti = Z.slice((uword) tauZ(ii-1)-1).row(jj);

      Lti = eyeM - K.slice(ii-1).col(jj) * Zti;
      
      rti = trans(Zti) / F(jj,ii-1) * v(jj,ii-1) + trans(Lti) * rti;

      Nti = trans(Zti) / F(jj,ii-1) * Zti + trans(Lti) * Nti * Lti;
    }
    r.col(ii-1) = rti;
    N.slice(ii-1) = Nti;

    alpha.col(ii-1) = a.col(ii-1) + P.slice(ii-1) * r.col(ii-1);
    V.slice(ii-1) = P.slice(ii-1) - P.slice(ii-1) * N.slice(ii-1) * P.slice(ii-1);
    eta.col(ii-1) = Q.slice((uword) tauQ(ii-1) - 1) * trans(R.slice((uword) tauR(ii-1) - 1)) * r.col(ii-1);

    Tii = T.slice((uword) tauT(ii-1) - 1);
    rti = trans(Tii) * rti;
    Ntemp = trans(Tii) * Nti * Tii;
    Nti = 0.5 * (Ntemp + Ntemp.t());
  }

  r1 = zeros(m, dt+1);
  
  r0ti = rti;      // Yes, ii. Was ii+1 in Matlab.
  r1ti = r1.col(ii);      // Yes, ii. Was ii+1 in Matlab.
  N0ti = Nti;
  N1ti = zeros(m, m);
  N2ti = zeros(m, m);

  // Initial values smoother
  for (ii=dt; ii > 0; --ii) {
    ind = find_finite(y.col(ii-1));

    for (iInd = ind.n_elem; iInd > 0; --iInd) {
      jj = ind(iInd-1);       // Note 0-based
      Zti = Z.slice((uword) tauZ(ii-1)-1).row(jj);

      if (Fd(jj,ii-1) != 0) {
        // Diffuse case
        Ldti = eyeM - Kd.slice(ii-1).col(jj) * Zti;
        L0ti = (Kd.slice(ii-1).col(jj) - K.slice(ii-1).col(jj)) * Zti * F(jj,ii-1) / Fd(jj,ii-1);

        r1ti = trans(Zti) / Fd(jj,ii-1) * v(jj,ii-1) + trans(L0ti) * r0ti + trans(Ldti) * r1ti;
        r0ti = trans(Ldti) * r0ti;

        N0ti = trans(Ldti) * N0ti * Ldti;
        N1ti = trans(Zti) / Fd(jj,ii-1) * Zti + trans(Ldti) * N0ti * L0ti + trans(Ldti) * N1ti * Ldti;
        N2ti = trans(Zti) * pow(Fd(jj,ii-1), -2) * Zti * F(jj,ii-1) + 
          trans(L0ti) * N1ti * L0ti + trans(Ldti) * N1ti * L0ti + 
          trans(L0ti) * N1ti * Ldti + trans(Ldti) * N2ti * Ldti;
      }
      else {
        // Known forecast variance
        Lstarti = eyeM - K.slice(ii-1).col(jj) * Zti;
        r0ti = trans(Zti) / F(jj,ii-1) * v(jj,ii-1) + trans(Lstarti) * r0ti;

        N0ti = trans(Zti) / F(jj,ii-1) * Zti + trans(Lstarti) * N0ti * Lstarti;
      }
    }

    // Move to next period
    r.col(ii-1) = r0ti;
    r1.col(ii-1) = r1ti;
    N.slice(ii-1) = N0ti;

    alpha.col(ii-1) = a.col(ii-1) + P.slice(ii-1) * r.col(ii-1) + 
      Pd.slice(ii-1) * r1.col(ii-1);
    V.slice(ii-1) = P.slice(ii-1) - P.slice(ii-1) * N0ti * P.slice(ii-1) - 
      trans(Pd.slice(ii-1) * N1ti * P.slice(ii-1)) - P.slice(ii-1) * N1ti * Pd.slice(ii-1) - 
      Pd.slice(ii-1) * N2ti * Pd.slice(ii-1);

    eta.col(ii-1) = Q.slice((uword) tauQ(ii-1) - 1) * trans(R.slice((uword) tauR(ii-1) - 1)) * r.col(ii-1);

    r0ti = trans(T.slice((uword) tauT(ii-1) - 1))  * r0ti;
    r1ti = trans(T.slice((uword) tauT(ii-1) - 1))  * r1ti;

    N0ti = trans(T.slice((uword) tauT(ii-1) - 1))  * N0ti * T.slice((uword) tauT(ii-1) - 1) ;
    N1ti = trans(T.slice((uword) tauT(ii-1) - 1))  * N1ti * T.slice((uword) tauT(ii-1) - 1) ;
    N2ti = trans(T.slice((uword) tauT(ii-1) - 1))  * N2ti * T.slice((uword) tauT(ii-1) - 1) ;
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
  output.V = V;
  output.eta = eta;
  output.r = r;
  output.N = N;
  output.a0tilde = a0tilde;
  return output;
}

#endif
