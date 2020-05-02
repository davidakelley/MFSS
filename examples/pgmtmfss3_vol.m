% 
%

% See Also:
%   pgmtmfss_replication.m - Replication code runner script

load('data_trade');

%% Code Example 3: 
Y = log(data_diffl.^2);

% Set up accumulator
calendar = GenerateIrregular(dates, struct('Type', 'Monthly'));
accum = Accumulator(1, [calendar(:,2); 1], ones(size(calendar,1)+1,1));

% Set up state space parameters
Z = ones(2,1);
d = nan(2,1);
H = diag(nan(2,1));
T = nan;
Q = nan;

% Set up MFVAR object with one lag and estimate coefficients 
mdl = StateSpaceEstimation(Z, H, T, Q, 'd', d);
mdlA = accum.augmentStateSpaceEstimation(mdl);

mdlMLE = mdlA.estimate(Y);

% DK: I'm getting phi = 0.995 and a likelihood of -12030. This seems nice. 

% Get daily interest rate changes
alpha = mdlMLE.smooth(Y);

%% Plot time series of estimated volatility
startInx = 1;
sigmaHat = exp(0.5 .* alpha(:,1));

figure('Color', ones(1,3));
plot(dates(startInx:end), sigmaHat(startInx:end), 'LineWidth', 2);
box off;
datetick('x', 'keeplimits')
xlim([dates(startInx)-15, dates(end)+15]);

print 'pgmtmfss3_vol.png' -dpng
