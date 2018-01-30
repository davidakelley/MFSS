% Estimate a local-level model on the Nile River dataset

% Load data
data = load('examples\data\dk.mat');
nileData = data.nile';

% Set up the state space
Z = 1; d = 0; H = nan;
T = 1; c = 0; R = 1; Q = nan;
ssE = StateSpaceEstimation(Z, d, H, T, c, R, Q);

% Estimate unknown parameters
ssOpt = ssE.estimate(nileData);

state = ssOpt.smooth(nileData);
plot([nileData' state'])

[decompData, decompConst] = ssOpt.decompose_filtered(nileData);