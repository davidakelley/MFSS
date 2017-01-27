// Automatically translated using Matlab2cpp 0.5 on 2017-01-25 16:42:54

#ifndef GRADIENT_MULTI_MEX_M_HPP
#define GRADIENT_MULTI_MEX_M_HPP

#include "genCommutation.hpp"
#include "armadillo.hpp"
#include "mex.h"
using namespace arma ;

#define uint (unsigned int)

vec gradient_multi_mex(mat y, struct ss, struct G, struct fOut) ;

struct _Tau {
  vec Z, d, H, T, c, R, Q ;
} ;

struct _Ss {
  int p, m, g, n ;
  cube Z, H, T, R, Q ;
  mat d, c, a0, P0 ;
  _Tau tau ;
} ;

struct _G {
  cube Z, d, H, T, c, R, Q ;
  mat a0, P0 ;
} ;

struct _Fout {
  cube P, F, M, K, L, Finv ;
  mat a, v, w ;
} ;

struct _unique {
  mat rows;
  vec index;
};

// Precision isn't a worry here since this will be used for comparing ints.
template <typename T>
inline bool rows_equal(const T& lhs, const T& rhs, double tol = 0.00000001) {
    return approx_equal(lhs, rhs, "absdiff", tol);
}

_unique unique_rows(const mat& x) {
  // Initially from 
  // stackoverflow.com/questions/37143283/finding-unique-rows-in-armamat
  // Substantially altered since. 

  unsigned int count, matchRow, i, j, nr, nc;
  count = 1; // Matlab-style (1-based) count of rows in result
  nr = x.n_rows;
  nc = x.n_cols;

  mat result(nr, nc);
  vec index(nr);

  // Initialize with first row (must be unique)
  result.row(0) = x.row(0);
  index(0) = count;

  for (i = 1; i < nr; i++) {
    // Looping over rows of the input array
    bool matched = false;

    for (j = 0; j < count; j++) {
      // Looping over rows of the unique array
      if (rows_equal(x.row(i), result.row(j))) {
        matched = true;
        matchRow = j + 1; // 1-based
        break;
      }
    }

    if (!matched) {
      // Didn't find current row, add to result
      count++;
      result.row(count-1) = x.row(i);
      matchRow = count;
    }

    // 1-based index of which row of result we matched on
    index(i) = matchRow;
  }

  _unique output;
  output.rows = result.rows(0, count-1);
  output.index = index;
  return output;
}

vec gradient_multi_mex(mat y, _Ss ss, _G G, _Fout fOut) {
  _Tau tau ;
  cube F, Finv, GH, GQ, GR, GT, GZ, Gc, Gd, H, K, L, M, P, Q, R, T, Z, kronQRI, kronRR ;
  size_t g, m, n, p ;
  unsigned int iR, iQR, ii ;
  uvec ind ;
  mat GP, GP0, Ga, Ga0, Mv, Nm, P0, PL, W, W_base, Zii, a, c, commutation, d, grad, kronAMvI, kronKK,
    kronLL, kronPLI, kronPLK, kronPLw, kronWW, kronZwL, kronaMvK, kronwK, tauQRrows, v, w, ww, wii;
  uword nTheta ;
  vec a0, discard, gradient, tauH, tauQ, tauQR, tauR, tauT, tauZ, tauc, taud ;
  
  // State space parameters
  Z = ss.Z ;   d = ss.d ;   H = ss.H ;
  T = ss.T ;   c = ss.c ;   R = ss.R ;   Q = ss.Q ;
  a0 = ss.a0 ;   P0 = ss.P0 ;
  
  // State space dimensions
  p = ss.p ;   m = ss.m ;
  g = ss.g ;   n = ss.n ;
  
  // Parameter gradients 
  GZ = G.Z ;   Gd = G.d ;   GH = G.H ;
  GT = G.T ;   Gc = G.c ;   GR = G.R ;   GQ = G.Q ;
  Ga0 = G.a0 ;   GP0 = G.P0 ;
  
  // Mixed-frequency parameter indexes
  tau = ss.tau ;
  tauZ = tau.Z ;   taud = tau.d ;   tauH = tau.H ;
  tauT = tau.T ;   tauc = tau.c ;   tauR = tau.R ;   tauQ = tau.Q ;
  
  // Results from the multivariate filter
  a = fOut.a ;
  P = fOut.P ;
  v = fOut.v ;
  F = fOut.F ;
  M = fOut.M ;
  K = fOut.K ;
  L = fOut.L ;
  w = fOut.w ;
  Finv = fOut.Finv ;
  
  // nTheta = size(GT, 1)
  nTheta = GT.n_rows ;
  // commutation = genCommutation(m)
  commutation = genCommutation(m) ;
  // Nm = (eye(m^2) + commutation)
  Nm = (eye((int) pow(m, 2), (int) pow(m, 2)) + commutation) ;
  // kronRR = zeros(g*g, m*m, max(tauR))
  kronRR = zeros(g*g, m*m, R.n_slices) ;

  mat tempR;
  for (iR=1; iR<=R.n_slices; iR++) {
    // kronRR(:, :, iR) = kron(R(:,:,iR)', R(:,:,iR)')
    tempR = trans(R.slice(iR-1));
    kronRR.slice(iR-1) = kron(tempR, tempR);
  }
  // [tauQRrows, ~, tauQR] = unique([tauR tauQ], 'rows')
  _unique uniquetauQR = unique_rows(join_rows(tauR, tauQ));
  tauQRrows = uniquetauQR.rows;
  tauQR = uniquetauQR.index;

  // kronQRI = zeros(g * m, m * m, max(tauQR))
  kronQRI = zeros(g*m, m*m, tauQRrows.n_rows) ;
  for (iQR=1; iQR<=tauQRrows.n_rows; iQR++) {
    // kronQRI(:, :, iQR) = kron(Q(:,:,tauQRrows(iQR, 2)) * R(:,:,tauQRrows(iQR, 1))', eye(m))
    kronQRI.slice(iQR-1) = kron(Q.slice((uword) tauQRrows(iQR-1, 1)-1) * 
      trans(R.slice((uword) tauQRrows(iQR-1, 0)-1)), eye(m, m)) ;
  }

  // Ga = Ga0 * T(:,:,tauT(1))' + Gc(:, :, tauc(1)) + GT(:,:,tauT(1)) * kron(a0, eye(m))
  Ga = Ga0 * trans(T.slice((uword) tauT(0)-1)) + 
    Gc.slice((uword) tauc(0)-1) + 
    GT.slice((uword) tauT(0)-1) * kron(a0, eye(m, m));

  /* GP = GP0 * kron(T(:,:,tauT(1))', T(:,:,tauT(1))') + 
    GQ(:,:,tauQ(1)) * kron(R(:,:,tauR(1))', R(:,:,tauR(1))') + 
    (GT(:,:,tauT(1)) * kron(P0 * T(:,:,tauT(1))', eye(m)) + 
    GR(:,:,tauR(1)) * kron(Q(:,:,tauQ(1)) * R(:,:,tauR(1))', eye(m))) * 
  (eye(m^2) + commutation) */

  GP = GP0 * kron(trans(T.slice((uword) tauT(0)-1)), trans(T.slice((uword) tauT(0)-1))) + 
    GQ.slice((uword) tauQ(0)-1) * kronRR.slice((uword) tauR(0)-1) + 
    (GT.slice((uword) tauT(0)-1) * 
      kron(P0 * trans(T.slice((uword) tauT(0)-1)), eye(m, m)) + 
      GR.slice((uword) tauR(0)-1) * 
        kron(Q.slice((uword) tauQ(0)-1) * trans(R.slice((uword) tauR(0)-1)), eye(m, m))) * Nm;

  // W_base = logical(sparse(eye(p)))
  W_base = eye(p, p);

  uvec iiVec(1);

  // grad = zeros(nTheta, n)
  grad = zeros(nTheta, n) ;
  for (ii=1; ii<=n; ii++)  {
    ind = find_finite(y.col(ii-1));
    W = W_base.rows(ind);
    
    // ind = ~isnan(y(:,ii))
    // ind = not isnan(y.col(ii-1)) ;
    // W = W_base((ind==1),:)
    // W = W_base((ind==1)-1, m2cpp::span(0, W_base.n_cols-1)-1) ;
    // kronWW = kron(W', W')
    kronWW = kron(trans(W),  trans(W)) ;
    // Zii = W * Z(:, :, tauZ(ii))
    Zii = W*Z.slice((uword) tauZ(ii-1)-1) ;
    // ww = w(ind,ii) * w(ind,ii)'
    iiVec.fill(ii-1);
    wii = w.submat(ind, iiVec);
    ww = wii * trans(wii) ;
    // Mv = M(:,:,ii) * v(:, ii)
    Mv = M.slice(ii-1)*v.col(ii-1) ;

    /* grad(:, ii) = Ga * Zii' * w(ind,ii) + 
      0.5 * GP * vec(Zii' * ww * Zii - Zii' * Finv(ind,ind,ii) * Zii) + 
      Gd(:,:,taud(ii)) * W' * w(ind,ii) + 
      GZ(:,:,tauZ(ii)) * vec(W' * (w(ind,ii) * a(:,ii)' + w(ind,ii) * Mv' - M(:,ind,ii)')) + 
      0.5 * GH(:,:,tauH(ii)) * kronWW * vec(ww - Finv(ind,ind,ii)) */
    grad.col(ii-1) = Ga*trans(Zii)*wii+
      0.5*GP*vectorise(trans(Zii)* ww * Zii- trans(Zii) * Finv.slice(ii-1).submat(ind, ind)*Zii) + 
      Gd.slice((uword) taud(ii-1)-1) * trans(W) * wii + 
      GZ.slice((uword) tauZ(ii-1)-1) * 
        vectorise(trans(W) * (wii*trans(a.col(ii-1)) + wii*trans(Mv) - trans(M.slice(ii-1).cols(ind)) )) + 
      0.5*GH.slice((uword) tauH(ii-1)-1) * kronWW * 
        vectorise(ww-Finv.slice(ii-1).submat(ind, ind)) ;

    // PL = P(:,:,ii) * L(:,:,ii)'
    PL = P.slice(ii-1) * 
    trans(L.slice(ii-1)) ;
    // kronZwL = kron(Zii' * w(ind,ii), L(:,:,ii)')
    kronZwL = kron(trans(Zii)*wii, trans(L.slice(ii-1))) ;
    // kronPLw = kron(PL, w(:,ii))
    kronPLw = kron(PL, w.col(ii-1)) ;
    // kronaMvK = kron(a(:,ii) + Mv, K(:,:,ii)')
    kronaMvK = kron(a.col(ii-1)+Mv, trans(K.slice(ii-1))) ;
    // kronwK = kron(w(:,ii), K(:,:,ii)')
    kronwK = kron(w.col(ii-1), trans(K.slice(ii-1))) ;
    // kronAMvI = kron(a(:,ii) + Mv, eye(m))
    kronAMvI = kron(a.col(ii-1)+Mv, eye(m, m)) ;
    
    /* Ga = Ga * L(:,:,ii)' + 
      GP * kronZwL + 
      Gc(:,:,tauc(ii+1)) - 
      Gd(:,:,taud(ii)) * K(:,:,ii)' + 
      GZ(:,:,tauZ(ii)) * (kronPLw - kronaMvK) - 
      GH(:,:,tauH(ii)) * kronwK + 
      GT(:,:,tauT(ii+1)) * kronAMvI */
    Ga = Ga * trans(L.slice(ii-1)) + 
      GP * kronZwL + 
      Gc.slice((uword) tauc(ii)-1) - 
      Gd.slice((uword) taud(ii-1)-1) * trans(K.slice(ii-1)) + 
      GZ.slice((uword) tauZ(ii-1)-1) * (kronPLw-kronaMvK) - 
      GH.slice((uword) tauH(ii-1)-1) * kronwK + 
      GT.slice((uword) tauT(ii)-1)*kronAMvI ;

    // kronLL = kron(L(:,:,ii)', L(:,:,ii)')
    kronLL = kron(trans(L.slice(ii-1)), trans(L.slice(ii-1))) ;
    // kronKK = kron(K(:,:,ii)', K(:,:,ii)')
    kronKK = kron(trans(K.slice(ii-1)), trans(K.slice(ii-1))) ;
    // kronPLI = kron(PL, eye(m))
    kronPLI = kron(PL, eye(m, m)) ;
    // kronPLK = kron(PL, K(:,:,ii)')
    kronPLK = kron(PL, trans(K.slice(ii-1))) ;
    
    /* GP = GP * kronLL + 
      GH(:,:,tauH(ii)) * kronKK + 
      GQ(:,:,tauQ(ii+1)) * kronRR(:,:, tauR(ii+1)) + 
      (GT(:,:,tauT(ii+1)) * kronPLI - 
        GZ(:,:,tauZ(ii)) * kronPLK + 
        GR(:,:,tauR(ii+1)) * kronQRI(:, :, tauQR(ii+1))) 
      * Nm */
    GP = GP * kronLL + 
      GH.slice((uword) tauH(ii-1)-1) * kronKK + 
      GQ.slice((uword) tauQ(ii)-1) * kronRR.slice((uword) tauR(ii)-1) + 
      (GT.slice((uword) tauT(ii)-1) * kronPLI - 
        GZ.slice((uword) tauZ(ii-1)-1) * kronPLK + 
        GR.slice((uword) tauR(ii)-1) * kronQRI.slice((uword) tauQR(ii)-1)) * 
      Nm ;

  }

  // gradient = sum(grad, 2)'
  gradient =  sum(grad, 1);
  return gradient ;
}

#endif


