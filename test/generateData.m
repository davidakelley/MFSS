function [Y, alpha, eta, epsilon] = generateData(ss, timeDim)
% Generate test data from a state-space model

% Generate data
eta = ss.R * ss.Q^(1/2) * randn(ss.g, timeDim);
alpha = nan(ss.m, timeDim);
alpha(:,1) = eta(:,1);
for iT = 2:timeDim
  alpha(:,iT) = ss.T * alpha(:,iT-1) + eta(:,iT);
end

epsilon = ss.H^(1/2) * randn(ss.p, timeDim);
Y = nan(ss.p, timeDim);
for iT = 1:timeDim
  Y(:,iT) = ss.Z * alpha(:,iT) + epsilon(:,iT);
end

end
