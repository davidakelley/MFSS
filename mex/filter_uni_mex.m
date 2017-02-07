function logli = filter_uni_mex(y, Z, d, H, T, c, R, Q, A0, R0, Q0, tauZ, taud, tauH, tauT, tauc, tauR, tauQ, m, p)
% Filter using exact initial conditions
%
% See "Fast Filtering and Smoothing for Multivariate State Space Models",
% Koopman & Durbin (2000).


n = size(y, 2);

% assert(isdiag(H), 'Univarite only!');
% 
% % Preallocate
% % Note Pd is the "diffuse" P matrix (P_\infty).
a = zeros(m, n+1);
v = zeros(p, n);

Pd = zeros(m, m, n+1);
Pstar = zeros(m, m, n+1);
Fd = zeros(p, n);
Fstar = zeros(p, n);

Kd = zeros(m, p, n);
Kstar = zeros(m, p, n);

LogL = zeros(p, n);

% Initialize
a(:,1) = a0;
Pd(:,:,1) = A0 * A0';
Pstar(:,:,1) = R0 * Q0 * R0';
 
ii = 0;
% Initial recursion
while ~all(all(Pd(:,:,ii+1) == 0))
  if ii >= n
%     error('Degenerate model. Exact initial filter unable to transition to standard filter.');
  end
  
  ii = ii + 1;
  ind = find( ~isnan(y(:,ii)) );
  
  ati = a(:,ii);
  Pstarti = Pstar(:,:,ii);
  Pdti = Pd(:,:,ii);
  for jj = ind'    
    Zjj = Z(jj,:,tauZ(ii));
    v(jj,ii) = y(jj, ii) - Zjj * ati - d(jj,taud(ii));
    
    Fd(jj,ii) = Zjj * Pdti * Zjj';
    Fstar(jj,ii) = Zjj * Pstarti * Zjj' + H(jj,jj,tauH(ii));
    
    Kd(:,jj,ii) = Pdti * Zjj';
    Kstar(:,jj,ii) = Pstarti * Zjj';
    
    if Fd(jj,ii) ~= 0
      % F diffuse nonsingular
      ati = ati + Kd(:,jj,ii) ./ Fd(jj,ii) * v(jj,ii);
      
      Pstarti = Pstarti + Kd(:,jj,ii) * Kd(:,jj,ii)' * Fstar(jj,ii) * (Fd(jj,ii).^-2) - (Kstar(:,jj,ii) * Kd(:,jj,ii)' + Kd(:,jj,ii) * Kstar(:,jj,ii)') ./ Fd(jj,ii);
      
      Pdti = Pdti - Kd(:,jj,ii) .* Kd(:,jj,ii)' ./ Fd(jj,ii);
      
      LogL(jj,ii) = log(Fd(:,ii));
    else
      % F diffuse = 0
      ati = ati + Kstar(:,jj,ii) ./ Fstar(jj,ii) * v(jj,ii);
      
      Pstarti = Pstarti - Kstar(:,jj,ii) ./ Fstar(jj,ii) * Kstar(:,jj,ii)';
      
      LogL(jj,ii) = (log(Fstar(jj,ii)) + (v(jj,ii)^2) ./ Fstar(jj,ii));
    end
  end
  
  Tii = T(:,:,tauT(ii));
  a(:,ii+1) = Tii * ati + c(:,tauc(ii));
  
  Pd(:,:,ii+1)  = Tii * Pdti * Tii';
  Pstar(:,:,ii+1) = Tii * Pstarti * Tii' + ...
    R(:,:,tauR(ii)) * Q(:,:,tauQ(ii)) * R(:,:,tauR(ii))';
end

dt = ii;

F = Fstar;
M = Kstar;
P = Pstar;

% Standard Kalman filter recursion
for ii = dt+1:n
  ind = find( ~isnan(y(:,ii)) );
  ati    = a(:,ii);
  Pti    = P(:,:,ii);
  for jj = ind'
    Zjj = Z(jj,:,tauZ(ii));
    
    v(jj,ii) = y(jj,ii) - Zjj * ati - d(jj,taud(ii));
    
    F(jj,ii) = Zjj * Pti * Zjj' + H(jj,jj,tauH(ii));
    M(:,jj,ii) = Pti * Zjj';
    
    LogL(jj,ii) = (log(F(jj,ii)) + (v(jj,ii)^2) / F(jj,ii));
    
    ati = ati + M(:,jj,ii) / F(jj,ii) * v(jj,ii);
    Pti = Pti - M(:,jj,ii) / F(jj,ii) * M(:,jj,ii)';
  end
  
  Tii = T(:,:,tauT(ii));
  
  a(:,ii+1) = Tii * ati + c(:,tauc(ii));
  P(:,:,ii+1) = Tii * Pti * Tii' + ...
    R(:,:,tauR(ii)) * Q(:,:,tauQ(ii)) * R(:,:,tauR(ii))';
end

logli = -(0.5 * sum(sum(isfinite(y)))) * log(2 * pi) - 0.5 * sum(sum(LogL));

end