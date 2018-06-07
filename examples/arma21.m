
%% Get data
data = cbd.data('DIFFL(IP)', 'startDate', '1/1/1959');
y = data{:,:};

%% Set up and estimate model
Z = [1 0];
H = 0;
T = [nan, 1; nan, 0];
R = [1; nan];
Q = nan;
ssE = StateSpaceEstimation(Z, H, T, Q, 'R', R);

% Restrict MA component's magnitude
LB = ssE.ThetaMapping.LowerBound;
LB.R(2,1) = -1;
UB = ssE.ThetaMapping.UpperBound;
UB.R(2,1) = 1;
ssE.ThetaMapping = ssE.ThetaMapping.addRestrictions(LB, UB);

% Estimate
ssML = ssE.estimate(y);