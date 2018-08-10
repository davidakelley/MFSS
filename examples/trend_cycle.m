% Smooth trend-cycle model

%% Get data
data = cbd.data('LN(GDPH)');
y = data{:,:};

%% Set up model
syms rho lambda sigmaKappa sigmaXi

Z = [1, 0, 1, 0];
H = nan;

T = blkdiag([1 1; 0 1], rho .* [cos(lambda), sin(lambda); -sin(lambda) cos(lambda)]);
R = [zeros(1, 3); eye(3)];
Q = diag([sigmaXi; sigmaKappa; sigmaKappa]);

ssE = StateSpaceEstimation(Z, H, T, Q, 'R', R);
ssE.ThetaMapping = ssE.ThetaMapping.addStructuralRestriction(lambda, 0, 2*pi);
ssE.ThetaMapping = ssE.ThetaMapping.addStructuralRestriction(rho, -1, 1);

% Estimate the parameters
ssML = ssE.estimate(y);
