% Mixed-frequency trend-cycle decomposition of GDP
% 
% Data consist of the quarterly log-level of Real Gross Domestic Product aligned to a 
% monthly frequency. Data retreived from FRED as of August 2, 2018.
% Variables: 
%     y: vector time series of log levels of GDP. Quarterly obs. placed every 3rd period.
%     dates: monthly dates aligned with y
data = load('data_gdp.mat');
y = data.y;
dates = data.dates;

%% Code Example 3: Estimating a Mixed Frequency Stochastic Trend-Cycle Decomposition
syms lambda rho sigmaKappa sigmaZeta
Z = [1, 0, 1, 0];
H = 0;
T = blkdiag([1 1; 0 1], rho .* [cos(lambda), sin(lambda); -sin(lambda) cos(lambda)]);
R = [zeros(1, 3); eye(3)];
Q = diag([sigmaZeta; sigmaKappa; sigmaKappa]);
ssE = StateSpaceEstimation(Z, H, T, Q, 'R', R);

accum = Accumulator.GenerateRegular(y, {'avg'}, 1);
ssEA = accum.augmentStateSpaceEstimation(ssE);

ssEA.ThetaMapping = ssEA.ThetaMapping.addStructuralRestriction(rho, 0, 1);
ssEA.ThetaMapping = ssEA.ThetaMapping.addStructuralRestriction(lambda, pi/72, pi/9);

% Estimate quarterly model - Initial values from estimated model of Jarvey & Jaeger
ssMLq = ssE.estimate(y(3:3:end), [pi/22.2; 0.92; log(625/1e7); log(8/1e7)]);
alpha_q = ssMLq.smooth(y(3:3:end));
% Estimate monthly model - Initial values frequency adjusted from quarterly initial values
ssML = ssEA.estimate(y, [0.0943; 0.9610; log(0.00003379); log(0.0000003789)]);
alpha_m = ssML.smooth(y);
% Estimate HP filter (and construct latent state similar to trend-cycle models)
trend_hp = hpfilter(y(3:3:end), 1600);
alpha_hp = [trend_hp, [diff(trend_hp); nan], y(3:3:end)-trend_hp, nan(size(y(3:3:end)))];

%% Plot trend and cycle estimates
figH = figure('Color', ones(1,3));
lineWidth = 1.5;

subplot(2,2,1);
plot(dates(3:3:end), y(3:3:end), 'k', 'LineWidth', lineWidth);
hold on;
plot(dates, alpha_m(:,1), 'LineWidth', lineWidth);
plot(dates(3:3:end), alpha_q(:,1), 'LineWidth', lineWidth);
plot(dates(3:3:end), alpha_hp(:,1), 'LineWidth', lineWidth);
box off
datetick('x');
xticklabels([]);
xlim(dates([1 end]));
title('Data and Trend GDP Level');
ylabel('Log level');

subplot(2,2,2);
triAvgCycle = filter([1 2 3 2 1], 1, alpha_m(:,3)) ./ 3;
plot(dates(5:end), triAvgCycle(5:end), 'LineWidth', lineWidth);
hold on;
plot(dates(3:3:end), alpha_q(:,3), 'LineWidth', lineWidth);
plot(dates(3:3:end), alpha_hp(:,3), 'LineWidth', lineWidth);
box off
datetick('x');
recessionplot();
xlim(dates([1 end]));
xticklabels([]);
ylim([-.3 .2]);
title('Cycle');

subplot(2,2,3);
triAvgTrend = filter([1 2 3 2 1], 1, alpha_m(:,2)) ./ 3;
plot(dates(5:end), triAvgTrend(5:end), 'LineWidth', lineWidth);
hold on;
plot(dates(3:3:end), alpha_q(:,2), 'LineWidth', lineWidth);
plot(dates(3:3:end), alpha_hp(:,2), 'LineWidth', lineWidth);
box off
datetick('x');
xlim(dates([1 end]));
title('Trend growth rate');
ylabel('Log change');

subplot(2,2,4);
triAvgCycle = filter([1 2 3 2 1], 1, alpha_m(:,3)) ./ 3;
plot(dates(6:3:end), diff(triAvgCycle(3:3:end)), 'LineWidth', lineWidth);
hold on;
plot(dates(6:3:end), diff(alpha_q(:,3)), 'LineWidth', lineWidth);
plot(dates(6:3:end), diff(alpha_hp(:,3)), 'LineWidth', lineWidth);
box off
datetick('x');
recessionplot();
xlim(dates([1 end]));
ylim([-.1 .1]);
title('Cycle growth rate');

print pgmtmfss3_trend_cycle.png -dpng
