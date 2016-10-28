function ssOut = generateARmodel(p, m, univariate)
% Generate the parameters of an AR factor model with p observables and m-1 lags
g = 1;

% Observation equation
Z = [rand(p, 1) zeros(p, m-1)];
d = zeros(p, 1);
baseH = (rand(p, p) + (diag(3 + rand(p, 1)))) ./ 10;
H = baseH * baseH' ./ 2;
if univariate 
  H = diag(diag(H));
end

% State equation
weightsAR = 2 * rand(1, m-1) - 1;
T = [weightsAR ./ sum(abs(weightsAR)) * .5, 0;
  eye(m-1), zeros(m-1, 1)];
c = zeros(m, 1);
R = [1; zeros(m-1, 1)];
Q = diag(rand(g));

ssOut = StateSpace(Z, d, H, T, c, R, Q, []);
end