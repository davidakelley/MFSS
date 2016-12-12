% autoregressive factor model

% Set up test
ssTrue = generateARmodel(4, 2, false);
ssTrue.T(1,:) = [0.8 0.05 0.02];
Y = generateData(ssTrue, 600);

Z = zeros(4, 3);
Z(:, 1) = nan;
d = zeros(4, 1);
H = eye(4, 4);
H(H == 1) = nan;

T = [nan(1, 3); [eye(2) zeros(2, 1)]];
c = zeros(3, 1);
R = [1; zeros(2, 1)];
Q = nan;

estim = StateSpaceEstimation(Z, d, H, T, c, R, Q);

Z0 = [ones(4, 1) zeros(4, 2)];
H0 = eye(4);
T0 = [0.3 0.3 0.3; zeros(2, 3)];
Q0 = 1;

ss0 = StateSpace(Z0, d, H0, T0, c, R, Q0);

ssE = estim.estimate(Y, ss0);