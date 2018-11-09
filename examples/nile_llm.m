% Estimate a local-level model on the Nile River dataset

%% Load data
data = load('durbin_koopman.mat');
y = data.nile';

%% Set up the state space
Z = 1; H = nan;
T = 1; Q = nan;
ssE = StateSpaceEstimation(Z, H, T, Q);

%% Estimate unknown parameters
ssOpt = ssE.estimate(y);

%% Decompose the trend
state = ssOpt.smooth(y);

[ssOpt, y, x] = ssOpt.checkSample(y, []);
[~, ~, fOut] = ssOpt.filter(y, x);
ssOpt = ssOpt.checkSample(y, x);
ssMulti = ssOpt;
[ssOpt, ~, ~, C] = ssOpt.prepareFilter(y, x, []);
[~, ~, ~, ~, weights] = ssOpt.decompose_smoothed(y);

w = cell2mat(weights.y);

before = sum(tril(w, -1),2);
after = sum(triu(w, 1),2);
current = diag(w);
timeContrib = 100 .* [before current after] ./ state';

%% Make plot of percent contribution to the state at each point
figure('Color', ones(1,3));
b1 = bar(timeContrib, 'stacked');
xlim([0.4 100.6]);
ylim([0 100]);
box off;
legend('Previous data', 'Current data', 'Subsequent data', ...
  'Location', 'southoutside', 'Orientation', 'horizontal');
title('Percent Contribution to Estimated Trend by Observations');
