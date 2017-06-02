% Estimate a local-level model on the Nile River dataset

% Load data
data = load('examples\data\dk.mat');
nileData = data.nile';

% Set up the state space
Z = 1; d = 0; H = nan;
T = 1; c = 0; R = 1; Q = nan;
ss = StateSpaceEstimation(Z, d, H, T, c, R, Q);

% Set the initial values used in the estimation
H0 = 1000;
Q0 = 1000;
ss0 = StateSpace(Z, d, H0, T, c, R, Q0);

% Estimate unknown parameters
ssE = ss.estimate(nileData, ss0);

state = ssE.smooth(nileData);
plot([nileData' state'])