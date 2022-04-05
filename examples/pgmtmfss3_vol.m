% Stochastic volatility estimation 
%
% Data are daily trade weighted exchange rate, and the monthly trade volume of world trade. 

% See Also:
%   pgmtmfss_replication.m - Replication code runner script

load('data_trade');
rng(1); 

%% Code Example 3: 
Y = log(data_diffl.^2);

% Set up accumulator
accum = Accumulator.GenerateFromDates(dates, 1, {'Month'}, {'avg'});

% Set up state space parameters
Z = ones(2,1);
d = nan(2,1);
H = diag(nan(2,1));
T = nan;
Q = nan;

% Set up MFVAR object with one lag and estimate coefficients 
mdl = StateSpaceEstimation(Z, H, T, Q, 'd', d);
mdlA = accum.augmentStateSpaceEstimation(mdl);
mdlA.P0 = diag(Inf(2,1));

% Estimate parameters
mdlMLE = mdlA.estimate(Y);

% Get smoothed estimate of volatility parameter
alpha = mdlMLE.smooth(Y);
alphaSim = mdlMLE.smoothSample(Y, [], [], [], 100);
sigmaHat = exp(0.5 .* alpha(:,1));
sigmaHatDraws = squeeze(exp(0.5 .* alphaSim(:,1,:)));
sigmaHatBands = prctile(sigmaHatDraws, [5 95], 2);

%% Plot time series of estimated volatility
startInx = 1;

figure('Color', ones(1,3));
hold on
fill([dates(startInx:end); flipud(dates(startInx:end))], ...
  [sigmaHatBands(startInx:end,1); flipud(sigmaHatBands(startInx:end,2))], ...
  0.6 * ones(1,3), 'EdgeColor', 'none');
plot(dates(startInx:end), sigmaHat(startInx:end), ...
  'LineWidth', 2, 'Color', [0 0.447 0.741]);
box off;
datetick('x', 'keeplimits')
ylabel('Estimated volatility');
xlim([dates(startInx)-15, dates(end)+15]);

print 'pgmtmfss3_vol.png' -dpng
