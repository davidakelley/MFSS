function ssOut = generateARmodel(p, lags, univariate)
% Generate the parameters of an AR factor model with p observables and m-1 lags
g = 1;

m = lags + 1;

% Observation equation
coefs = rand(p, 1);
Z = [coefs ./ coefs(1) zeros(p, lags)];
d = zeros(p, 1);
baseH = (rand(p, p) + (diag(3 + rand(p, 1)))) ./ 10;
H = baseH * baseH' ./ 2;
if univariate 
  H = diag(diag(H));
end

% State equation
weightsAR = 2 * rand(1, m) - 1;
T = [weightsAR ./ sum(abs(weightsAR)) * .5;
  [eye(lags), zeros(lags, 1)]];
c = zeros(m, 1);
R = [1; zeros(lags, 1)];
Q = diag(rand(g));

ssOut = StateSpace(Z, d, H, T, c, R, Q, []);
end