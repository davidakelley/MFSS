% Smooth trend-cycle model

%% Get data
data = cbd.data('LN(GDPH)');
y = data{:,:};

%% Set up model
syms rho lambda sigmaKappa sigmaZeta

Z = [1, 0, 1, 0];
H = nan;

T = blkdiag([1 1; 0 1], rho .* [cos(lambda), sin(lambda); -sin(lambda) cos(lambda)]);
R = [zeros(1, 3); eye(3)];
Q = diag([sigmaZeta; sigmaKappa; sigmaKappa]);

ssE = StateSpaceEstimation(Z, H, T, Q, 'R', R);

% Estimate the parameters
ssML = ssE.estimate(y);
