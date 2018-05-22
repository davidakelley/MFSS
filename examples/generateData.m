function [Y, alpha, x, eta, epsilon] = generateData(ss, timeDim)
% Generate test data from a state-space model

% Make sure StateSpace is set up
if isempty(ss.tau)
  ss.n = timeDim;
  ss = ss.setInvariantTau();
end

if isempty(ss.n) && ~isempty(ss.tau)
  % Preset initial values, invariant tau
  ss.n = timeDim;
  ss = ss.setInvariantTau();
end

if isempty(ss.a0) || isempty(ss.Q0)
  ss = ss.setDefaultInitial();
end

% Generate data
eta = nan(ss.g, timeDim);
rawEta = randn(ss.g, timeDim);
alpha = nan(ss.m, timeDim);

eta(:,1) = ss.Q(:,:,ss.tau.Q(1))^(1/2) * rawEta(:,1);

alpha(:,1) = ss.T(:,:,ss.tau.T(1)) * ss.a0 + ...
  ss.c(:,ss.tau.c(1)) + ...
  ss.R(:,:,ss.tau.R(1)) * eta(:,1);

for iT = 2:timeDim
  eta(:,iT) = ss.Q(:,:,ss.tau.Q(iT))^(1/2) * rawEta(:, iT);
  
  alpha(:,iT) = ss.T(:,:,ss.tau.T(iT)) * alpha(:,iT-1) + ...
    ss.c(:,ss.tau.c(iT)) + ...
    ss.R(:,:,ss.tau.R(iT)) * eta(:,iT);
end

% Generate exogenous data
x = randn(ss.k, timeDim);

% Generate observe data
epsilon = nan(ss.p, timeDim);
rawEpsilon = randn(ss.p, timeDim);
Y = nan(ss.p, timeDim);
for iT = 1:timeDim
  epsilon(:,iT) = ss.H(:,:,ss.tau.H(iT))^(1/2) * rawEpsilon(:,iT);
  
  Y(:,iT) = ss.Z(:,:,ss.tau.Z(iT)) * alpha(:,iT) + ...
    ss.d(:,ss.tau.d(iT)) + ...
    ss.beta(:,:,ss.tau.beta(iT)) * x(:,iT) + ...
    epsilon(:,iT);
end

end
