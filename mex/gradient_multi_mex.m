function gradient = gradient_multi_mex(y, ss, G, fOut)
% Analytic gradient function for conversion to C++

% David Kelley, 2017


% d, H and c aren't needed
Z = ss.Z;
d = ss.d; %#ok<NASGU>
H = ss.H; %#ok<NASGU>
T = ss.T;
c = ss.c; %#ok<NASGU>
R = ss.R;
Q = ss.Q;
a0 = ss.a0;
P0 = ss.P0;

p = ss.p;
m = ss.m;
g = ss.g;
n = ss.n;

GZ = G.Z;
Gd = G.d;
GH = G.H;
GT = G.T;
Gc = G.c;
GR = G.R;
GQ = G.Q;
Ga0 = G.a0;
GP0 = G.P0;

tau = ss.tau;
tauZ = tau.Z;
taud = tau.d;
tauH = tau.H;
tauT = tau.T;
tauc = tau.c;
tauR = tau.R;
tauQ = tau.Q;

% F not used, needed for matlab2cpp
a = fOut.a;
P = fOut.P;
v = fOut.v;
F = fOut.F;       %#ok<NASGU>
M = fOut.M;
K = fOut.K;
L = fOut.L;
w = fOut.w;
Finv = fOut.Finv;

nTheta = size(GT, 1);

commutation = genCommutation(m);
Nm = (eye(m^2) + commutation);

% Compute partial results that have less time-variation (even with TVP)
kronRR = zeros(g*g, m*m, max(tauR));
for iR = 1:max(tauR)
  kronRR(:, :, iR) = kron(R(:,:,iR)', R(:,:,iR)');
end

[tauQRrows, discard, tauQR] = unique([tauR tauQ], 'rows');
kronQRI = zeros(g * m, m * m, max(tauQR));
for iQR = 1:max(tauQR)
  kronQRI(:, :, iQR) = kron(Q(:,:,tauQRrows(iQR, 2)) * R(:,:,tauQRrows(iQR, 1))', eye(m));
end

% Initial period: G.a and G.P capture effects of a0, T
Ga = Ga0 * T(:,:,tauT(1))' + Gc(:, :, tauc(1)) + GT(:,:,tauT(1)) * kron(a0, eye(m));
% Yes, G.c is 3D.
GP = GP0 * kron(T(:,:,tauT(1))', T(:,:,tauT(1))') + GQ(:,:,tauQ(1)) * kron(R(:,:,tauR(1))', R(:,:,tauR(1))') + (GT(:,:,tauT(1)) * kron(P0 * T(:,:,tauT(1))', eye(m)) + GR(:,:,tauR(1)) * kron(Q(:,:,tauQ(1)) * R(:,:,tauR(1))', eye(m))) * (eye(m^2) + commutation);

% Recursion through time periods
W_base = logical(sparse(eye(p)));

grad = zeros(n, nTheta);
for ii = 1:n
  ind = ~isnan(y(:,ii));
  W = W_base((ind==1),:);
  kronWW = kron(W', W');

  Zii = W * Z(:, :, tauZ(ii));
  
  ww = w(ind,ii) * w(ind,ii)';
  Mv = M(:,:,ii) * v(:, ii);
  
  grad(ii, :) = Ga * Zii' * w(ind,ii) + 0.5 * GP * vec(Zii' * ww * Zii - Zii' * Finv(ind,ind,ii) * Zii) + Gd(:,:,taud(ii)) * W' * w(ind,ii) + GZ(:,:,tauZ(ii)) * vec(W' * (w(ind,ii) * a(:,ii)' + w(ind,ii) * Mv' - M(:,ind,ii)')) + 0.5 * GH(:,:,tauH(ii)) * kronWW * vec(ww - Finv(ind,ind,ii));
  
  % Set t+1 values
  PL = P(:,:,ii) * L(:,:,ii)';
  
  kronZwL = kron(Zii' * w(ind,ii), L(:,:,ii)');
  kronPLw = kron(PL, w(:,ii));
  kronaMvK = kron(a(:,ii) + Mv, K(:,:,ii)');
  kronwK = kron(w(:,ii), K(:,:,ii)');
  kronAMvI = kron(a(:,ii) + Mv, eye(m));
  
  Ga = Ga * L(:,:,ii)' + GP * kronZwL + Gc(:,:,tauc(ii+1)) - Gd(:,:,taud(ii)) * K(:,:,ii)' + GZ(:,:,tauZ(ii)) * (kronPLw - kronaMvK) - GH(:,:,tauH(ii)) * kronwK + GT(:,:,tauT(ii+1)) * kronAMvI;
  
  kronLL = kron(L(:,:,ii)', L(:,:,ii)');
  kronKK = kron(K(:,:,ii)', K(:,:,ii)');
  kronPLI = kron(PL, eye(m));
  kronPLK = kron(PL, K(:,:,ii)');
  
  GP = GP * kronLL + GH(:,:,tauH(ii)) * kronKK + GQ(:,:,tauQ(ii+1)) * kronRR(:,:, tauR(ii+1)) + (GT(:,:,tauT(ii+1)) * kronPLI - GZ(:,:,tauZ(ii)) * kronPLK + GR(:,:,tauR(ii+1)) * kronQRI(:, :, tauQR(ii+1))) * Nm;
end

gradient = sum(grad, 1)';
end

function out = vec(M)
out = reshape(M, [], 1);
end
