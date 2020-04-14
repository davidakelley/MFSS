% 2-variable VAR that generates a daily series from a weekly series
%
% The data in rateLevels and rateChanges are 2 time series: 
%   - The 10-Year Treasury Constant Maturity Rate
%   - The 30-Year Fixed Rate Mortgage Average in the United States
% Data retrieved from FRED as of April 1, 2020. 
%
% See Also:
%   pgmtmfss_replication.m - Replication code runner script

load('data_var_disagg');

%% Code Example 3: Estimating a daily VAR weekly mortgage rate survey
Y = rateLevels; 

% Set up accumulator
cal = [repmat([1; 2; 3; 1; 1;], size(Y,1)/5, 1); 1];
accum = Accumulator(2, cal, ones(size(cal)));

% Set up MFVAR object and estimate
mdl = MFVAR(Y, 1, accum);
mdlE = mdl.estimate();

% Get daily interest rate changes
alpha = mdlE.smooth(mdl.Y);


%% Plot time series of interest rates
% Make a plot of the spread using the daily mortgage rate and the weekly rate
% where the weekly rate is the average of Mon-Wed rate placed on Tuesday.
startInx = 9394;
rateDisaggRS = alpha(:,2);
rateDisaggNan = rateLevels(:,2);

figure('Color', ones(1,3));
plot(dates(startInx:end), rateDisaggRS(startInx:end), 'LineWidth', 2);
hold on;
plot(dates(startInx:end), rateDisaggNan(startInx:end), 'x', 'MarkerSize', 10);
box off;
datetick('x', 'keeplimits')
xlim([dates(startInx), dates(end)]);
legend('Related series disaggregation', 'Observed data', ...
  'Location', 'sw', 'Orientation', 'vertical');
legend boxoff
ylabel('Percent');

print 'pgmtmfss3_var_disagg.png' -dpng
