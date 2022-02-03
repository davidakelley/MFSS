% Mixed-frequency Laubach & Williams Natural Rate of Interest 
%
% Data are taken from code provided by the Federal Reserve Bank of San Fransisco, modified
% or retreived from FRED to accomodate the monthly frequency.
%
% See Also:
%   pgmtmfss_replication.m - Replication code runner script

load('data_r_star_inputs.mat')

% Estimated parameters from Laubach-Williams code
thetaR = [1.564865401	-0.609755474	-0.055248043	0.574182907	0.369526918	...
  0.041247163	0.002288444	0.03554271	1.414502044	...
  log(0.346675183^2) log(0.760784621^2) log(0.597671856^2)]';

% Calibrated ratios for median-unbiased estimates of variances
lambda_g = 0.01183685;
lambda_z = 0.04195394;

%% Estimate the models
pgmtmfss5_r_star_quarterly
pgmtmfss5_r_star_monthly

%% Decompose r* by data series that determine it
% Monthly model
alphaM = ssMOpt.smooth(YM, XM, WM);
[yContribM, paramContribM, inflContribM, ratesContribM] = ssMOpt.decompose_smoothed(YM, XM, WM);

decompState = size(alphaM,2) - 1;
rGDP = squeeze(yContribM(decompState,1,:)) + paramContribM(decompState,:)';
rInf = sum(squeeze(yContribM(decompState,2,:)), 2) + sum(squeeze(inflContribM(decompState,1:3,:)))';
rR = squeeze(sum(ratesContribM(decompState, :,:),2));
rOil = squeeze(inflContribM(decompState,4,:));
rImport = squeeze(inflContribM(decompState,5,:));

check = sum([rGDP rInf rR rOil rImport], 2) - alphaM(:,decompState);
assert(all(abs(check) < 1e-8));

startPadM = 4;
plotDataM = [alphaM(:,end-1) rGDP rInf rR rOil rImport];

% Quarterly model 
alphaQ = ssQOpt.smooth(YQ, XQ, []);
[yContribQ, paramContribQ, XContribQ] = ssQOpt.decompose_smoothed(YQ, XQ, []);

decompState = size(alphaQ,2);
rGDP = squeeze(yContribQ(decompState,1,:)) + ...
  squeeze(sum(XContribQ(decompState,1:2,:), 2)) + paramContribQ(decompState,:)';
rInf = sum(squeeze(yContribQ(decompState,2,:)), 2) + sum(squeeze(XContribQ(decompState,5:7,:)))';
rR = squeeze(sum(XContribQ(decompState,3:4,:), 2));
rOil = squeeze(XContribQ(decompState,8,:));
rImport = squeeze(XContribQ(decompState,9,:));

check = sum([rGDP rInf rR rOil rImport], 2) - alphaQ(:,decompState);
assert(all(abs(check) < 1e-8));

plotDataQ = [alphaQ(:,decompState) rGDP rInf rR rOil rImport];
startPadQ = 2;

%% Plot comparison of decompositions
figure('Color', ones(1,3));
subplot(2,1,1);
hold on;
plot(datesm(startPadM:end), plotDataM(startPadM:end,1), 'k', 'LineWidth', 2);
plot(datesm(startPadM:end), plotDataM(startPadM:end,2), 'LineWidth', 1.5, 'Color', [0 0.447 0.741]);
plot(datesm(startPadM:end), plotDataM(startPadM:end,3), 'LineWidth', 1.5, 'Color', [0.85 0.325 0.098]);
plot(datesm(startPadM:end), plotDataM(startPadM:end,4), 'LineWidth', 1.5, 'Color', [0.929 0.694 0.125]);
plot(datesm(startPadM:end), plotDataM(startPadM:end,5), 'LineWidth', 1.5, 'Color', [0.494 0.184 0.556]);
plot(datesm(startPadM:end), plotDataM(startPadM:end,6), 'LineWidth', 1.5, 'Color', [0.466 0.674 0.188]);
plot([datesm(startPadM), datesm(end)], [0 0], 'k', 'LineWidth', 0.25);
datetick('x');
xlim([datesm(startPadM), datesm(end)]);
title('Contributions to Monthly r*');
xticklabels([]);
ylabel('percent');

subplot(2,1,2);
hold on;
plot(datesq(startPadQ:end), plotDataQ(startPadQ:end,1), 'k', 'LineWidth', 2);
plot(datesq(startPadQ:end), plotDataQ(startPadQ:end,2), 'LineWidth', 1.5, 'Color', [0 0.447 0.741]);
plot(datesq(startPadQ:end), plotDataQ(startPadQ:end,3), 'LineWidth', 1.5, 'Color', [0.85 0.325 0.098]);
plot(datesq(startPadQ:end), plotDataQ(startPadQ:end,4), 'LineWidth', 1.5, 'Color', [0.929 0.694 0.125]);
plot(datesq(startPadQ:end), plotDataQ(startPadQ:end,5), 'LineWidth', 1.5, 'Color', [0.494 0.184 0.556]);
plot(datesq(startPadQ:end), plotDataQ(startPadQ:end,6), 'LineWidth', 1.5, 'Color', [0.466 0.674 0.188]);
plot([datesq(startPadQ), datesq(end)], [0 0], 'k', 'LineWidth', 0.25);
datetick('x');
xlim([datesq(startPadQ), datesq(end)]);
title('Contributions to Quarterly r*');
ylabel('percent');

print pgmtmfss5_rstar.png -dpng
