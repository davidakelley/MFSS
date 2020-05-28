% Estimate an ARMA(2,1) model on quarterly GDP

%% Get data
data = load('data_gdp.mat');
y = data.y(6:3:end) - data.y(3:3:end-3);

%% Set up and estimate model
Z = [1 0];
H = 0;
T = [nan, 1; nan, 0];
c = [nan; 0];
R = [1; nan];
Q = nan;
ssE = StateSpaceEstimation(Z, H, T, Q, 'R', R, 'c', c);

% Restrict MA component's magnitude
LB = ssE.ThetaMapping.LowerBound;
LB.R(2,1) = -1;
UB = ssE.ThetaMapping.UpperBound;
UB.R(2,1) = 1;
ssE.ThetaMapping = ssE.ThetaMapping.addRestrictions(LB, UB);

% Estimate
ssML = ssE.estimate(y);

% Generate IRF
irf = ssML.impulseState(16);
plot(irf(1,:)');
