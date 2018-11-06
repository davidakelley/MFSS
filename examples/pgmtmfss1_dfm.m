% Mixed-frequency dynamic factor model 
%
% The data in y is a panel of 5 time series: 
%   - The quarterly log-difference of Real Gross Domestic Product
%   - The monthly log-difference of All Employees: Total Nonfarm Payrolls
%   - The monthly log-difference of Real personal income excluding current transfer receipts
%   - The monthly log-difference of Industrial Production Index
%   - The monthly log-difference of Real Manufacturing and Trade Industries Sales
% The monthly series are normalized have a mean of zero and a standard deviation of one. 
% Data retreived from FRED as of August 2, 2018. 

data = load('data_panel.mat');
y = data.y;
dates = data.dates;

%% Code Example 1: Estimating a Mixed Frequency Dynamic Factor model
% y is a panel of 5 time series at a monthly base frequency.
% The 1st series in y is quarterly with observations in the 3rd month of the quarter.
Z = [1; nan(4,1)];
d = [nan; zeros(4,1)];
H = diag(nan(5,1));
T = nan;
Q = nan;

ssE = StateSpaceEstimation(Z, H, T, Q, 'd', d);

LB = ssE.ThetaMapping.LowerBound;
UB = ssE.ThetaMapping.UpperBound;
LB.T(1,1) = -1;
UB.T(1,1) = 1;
ssE.ThetaMapping = ssE.ThetaMapping.addRestrictions(LB, UB);

accum = Accumulator.GenerateRegular(y, {'avg','','','',''}, [3 1 1 1 1]);
ssEA = accum.augmentStateSpaceEstimation(ssE);

ssML = ssEA.estimate(y);
alphaHat = ssML.smooth(y);

%% Plot
figure('Color', ones(1,3)); 

% Monthly data with monthly version of factor. Note that some data more than 6 standard 
% deviations away from their mean are not shown on this plot. 
subplot(2, 1, 1);
hold on
plot(dates, y(:,2:end), '.')
plot(dates, zscore(alphaHat(:,1)), 'k', 'LineWidth', 2)
datetick('x');
recessionplot
xlim(dates([1 end]))
ylim([-6 6])
title('Monthly factor and data (z-scores)');

% Quarterly data (GDP) with the aggregated (quarterly) factor
subplot(2, 1, 2);
hold on
plot(dates, y(:,1), '.')
plot(dates(3:3:end), alphaHat(3:3:end,3) + ssML.d(1), 'k', 'LineWidth', 2)
datetick('x');
recessionplot
xlim(dates([1 end]))
title('Quarterly factor and GDP (annualized growth rate)');

print 'pgmtmfss1_dfm.png' -dpng