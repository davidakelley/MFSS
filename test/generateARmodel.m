function ssOut = generateARmodel(p, m)
% Generate the parameters of an AR factor model with p observables and m-1 lags
g = 1;

% Observation equation
Z = rand(p, m);
d = zeros(p, 1);
baseH = (rand(p, p) + (diag(3 + rand(p, 1)))) ./ 10;
H = baseH * baseH' ./ 2;

% State equation
weightsAR = 2 * rand(1, m-1) - 1;
T = [weightsAR ./ sum(weightsAR) * .9, 0;
  eye(m-1), zeros(m-1, 1)];
c = zeros(m, 1);
R = [1; zeros(m-1, 1)];
Q = diag(rand(g));

ssOut = StateSpace(Z, d, H, T, c, R, Q, []);
end