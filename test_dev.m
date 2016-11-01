function [ss, ss0, y] = test_dev 
% Development tests
% Call with
%{
[ss, ss0, y] = test_dev;
ssE = ss.estimate(y, ss0);
%}

% Sam's transformations. 
% Stationarity constraint. 

%%
y = generateData();
Z = [1 0; nan 0];
d = zeros(2, 1);
H = [.1 0; 0 .3];
H(H == 1) = nan;
T = [nan, nan; 1 0];
c = [0; 0];
R = [1; -0];
Q = nan;
ss = StateSpace(Z, d, H, T, c, R, Q, []);

%% Set up initial values
ss0 = ss;
ss0.Z(isnan(ss0.Z)) = 1;
ss0.H(isnan(ss0.H)) = 1;
ss0.Q(isnan(ss0.Q)) = 1;
ss0.T(isnan(ss0.T)) = 0.1;

end

function [Y, alpha] = generateData()
m = 2;
p = 2;
g = 1;
timeDim = 1000;

Z = [1 0; 0.9 0];
H = [.1 0; 0 .3];

T = [0.7 0.2; 1 0];
R = [1; 0];
Q = 0.1;

eta = R * Q^(1/2) * randn(g, timeDim);
alpha = nan(m, timeDim);
alpha(:,1) = eta(:,1);
for iT = 2:timeDim
  alpha(:,iT) = T * alpha(:,iT-1) + eta(:,iT);
end

epsilon = H^(1/2) * randn(p, timeDim);
Y = nan(p, timeDim);
for iT = 1:timeDim
  Y(:,iT) = Z * alpha(:,iT) + epsilon(:,iT);
end

end