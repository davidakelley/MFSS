% 4-variable VAR at the monthly, quarterly, and mixed frequencies
%
% The data in yQuarterly and yMixedFreq are 4 time series: 
%   - The log-level of Real Gross Domestic Product
%   - The log-level of Consumer Price Index for All Urban Consumers: All Items
%   - The log-level of Producer Price Index for All Commodities
%   - The level of the effective federal funds rate.
% The data in yMonthly replaces GDP with: 
%   - The log-level of All Employees: Total Nonfarm Payrolls
% Data retrieved from FRED as of October 10, 2018. 
%
% See Also:
%   pgmtmfss_replication.m - Replication code runner script

load('data_var');

nSeries = 4;
varLagsQ = 2;
varLagsM = 6;

%% Code Example 2: Estimating a monthly VAR including quarterly GDP

% Estimate the Mixed-frequency VAR:
accum = Accumulator.GenerateRegular(yMixedFreq, {'AVG', '', '', ''}, [1 1 1 1]);
mfvarMdl = MFVAR(yMixedFreq, varLagsM, accum);
mfvarMdl.tol = 1e-4;
ssOpt = mfvarMdl.estimate();
Lmf = chol(ssOpt.Q, 'lower'); Dmf = eye(nSeries);
ssMFVAR = StateSpace([eye(nSeries) zeros(nSeries, nSeries*(varLagsM-1))], zeros(nSeries), ...
  [ssOpt.T(1:nSeries,1:nSeries*varLagsM,1); ...
  [eye(nSeries*(varLagsM-1)) zeros(nSeries*(varLagsM-1), nSeries)]], Dmf, ...
  'c', ssOpt.c(1:nSeries*varLagsM,1), ...
  'R', [Lmf; zeros(nSeries*(varLagsM-1), nSeries)]);
irfMF = ssMFVAR.impulseState(48);

% Estimate monthly model
yM = yMonthly(varLagsM+1:end,:);
lagM = lagmatrix(yMonthly, 1:varLagsM);
xM = [lagM(varLagsM+1:end,:) ones(size(yM,1),1)];
coefM = (yM' * xM) / (xM' * xM);
phiM = coefM(:,1:end-1);
consM = coefM(:,end);
sigmaM = ((yM' * yM) - coefM * (yM' * xM)') ./ (size(yMonthly,1) - varLagsM - 1); 
Lm = chol(sigmaM, 'lower'); Dm = eye(nSeries);
ssMVAR = StateSpace([eye(nSeries) zeros(nSeries, nSeries*(varLagsM-1))], zeros(nSeries), ...
  [phiM; [eye(nSeries*(varLagsM-1)) zeros(nSeries*(varLagsM-1), nSeries)]], Dm, ...
  'c', [consM; zeros(nSeries*(varLagsM-1),1)], ...
  'R', [Lm; zeros(nSeries*(varLagsM-1), nSeries)]);
irfM = ssMVAR.impulseState(48);

% Estimate quarterly model
yQ = yQuarterly(varLagsQ+1:end,:);
lagQ = lagmatrix(yQuarterly, 1:varLagsQ);
xQ = [lagQ(varLagsQ+1:end,:) ones(size(yQ,1),1)];
coefQ = (yQ' * xQ) / (xQ' * xQ); 
phiQ = coefQ(:,1:end-1);
consQ = coefQ(:,end);
sigmaQ = ((yQ' * yQ) - coefQ * (yQ' * xQ)') ./ (size(yQuarterly,1) - varLagsQ - 1); 
Lq = chol(sigmaQ, 'lower'); Dq = eye(nSeries);
ssQVAR = StateSpace([eye(nSeries) zeros(nSeries, nSeries*(varLagsQ-1))], zeros(nSeries), ...
  [phiQ; [eye(nSeries*(varLagsQ-1)) zeros(nSeries*(varLagsQ-1), nSeries)]], Dq, ...
  'c', [consQ; zeros(nSeries*(varLagsQ-1),1)], ...
  'R', [Lq; zeros(nSeries*(varLagsQ-1), nSeries)]);
irfQ = ssQVAR.impulseState(16);

%% Plot IRFs
% Response of activity (GDP or employment) to shock to funds rate
figure('Color', ones(1,3));

plot(1:48, squeeze(irfMF(1,:,nSeries)), 'LineWidth', 2);
hold on;
plot(3:3:48, squeeze(irfQ(1,:,nSeries)), 'x', 'MarkerSize', 10);
plot(1:48, squeeze(irfM(1,:,nSeries)), 'LineWidth', 2);
box off;
legend('Mixed-frequency (GDP)', 'Quarterly (GDP)', 'Monthly (Employment)', ...
  'Location', 'sw', 'Orientation', 'vertical');
legend boxoff
xlim([1 48])
xlabel('Months');
ylabel('Log change');

print 'pgmtmfss2_var.png' -dpng
