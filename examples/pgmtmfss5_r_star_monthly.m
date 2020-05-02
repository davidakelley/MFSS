% Laubach-Williams estimate of the natural rate of interest at the monthly frequency
%
% See Also:
%   pgmtmfss_r_star.m - Mixed-frequency Laubach & Williams Natural Rate of Interest 

%% Data
YM = [100*gdpM(27:end) inflationMA(27:end)];
XM = [inflationMA(26:end-1) (inflationMA(25:end-2)+inflationMA(24:end-3)+inflationMA(23:end-4))/3 ...
  (inflationMA(22:end-5)+inflationMA(21:end-6)+inflationMA(20:end-7)+inflationMA(19:end-8))/4 ...
  oilMA(26:end-1) importMA(27:end)];
XM(isnan(XM)) = 0;
WM = [realrateM(26:end) realrateM(25:end-1)];

datesm = linspace((dates(8))-60, dates(end), size(YM,1))';

%% Model
% The observed data is organized
%     [GDP_t pi_t]'
% The state is organized as 
%     [GDP*_t GDP*_t-1 GDP_t GDP_t-1 g_t g_t-1 z_t z_t-1 r*_t]'
% The exogenous series in the measurement equation are 
%     [pi_t-1 avg1(pi)_t avg2(pi)_t oil_t-1 import_t]'
% The exogenous series in the state equation are 
%     [r_t-1 r_t-2]' 
%
% See Also:
%   pgmtmfss_replication.m - Replication code runner script

syms a1 a2 a3 b1 b2 b3 b4 b5 c sigma2IS sigma2PC sigma2Ystar

Z = [0 0 1 0 zeros(1,5); 
    -b3 0 b3 0 zeros(1,5)];
beta = [zeros(1,5); b1 b2 (1-b1-b2) b4 b5];
H = diag([0 sigma2PC]);

T = [1 0 0 0 1 0 0 0 0; 1 zeros(1,8);
  1-a1 -a2 a1 a2 1-(12*c*a3/2) -12*c*a3/2 -a3/2 -a3/2 0; 0 0 1 0 zeros(1,5);
  zeros(1,4) 1 zeros(1,4); zeros(1,4) 1 zeros(1,4);
  zeros(1,6) 1 zeros(1,2); zeros(1,6) 1 zeros(1,2);
  0 0 0 0 12*c 0 1 0 0];
gamma = [zeros(2); a3/2 a3/2; zeros(6,2)];
R = [1 0 0 0; zeros(1,4);
     1 0 1 0 ; zeros(1,4); 
     0 lambda_g/3 0 0; zeros(1,4);
     0 0 0 lambda_z/(3*a3); zeros(1,4);
     zeros(1,4)];
Q = diag([sigma2Ystar sigma2Ystar sigma2IS sigma2IS]);

ssE = StateSpaceEstimation(Z, H, T, Q, 'beta', beta, 'gamma', gamma, 'R', R);

% Bounds on the slopes of the IS and Phillips curves
ssE.ThetaMapping = ssE.ThetaMapping.addStructuralRestriction(a3, [], -0.0025);
ssE.ThetaMapping = ssE.ThetaMapping.addStructuralRestriction(b3, 0.025, []);

%% Augment
accum = Accumulator.GenerateRegular(YM, {'avg', ''}, [1 0]);
ssEA = accum.augmentStateSpaceEstimation(ssE);

%% Initial state
interpGDP = interp1(3:3:size(gdpM,1), gdpM(3:3:end), 1:size(gdpM))';
lagGDP = 100*interpGDP(25:26)';
gInit = 0.9394933/3;
ystarInit = [809.7823659 809.7823659-gInit];
rstarInit = 12*gInit;
ssE.a0 = [ystarInit, lagGDP, repmat(gInit, [1 2]), zeros(1, 2), rstarInit]';
ssE.P0 = 0.2*eye(9);
ssE.P0(3,3) = 0;
ssE.P0(4,4) = 0;

ssEA.a0 = [ssE.a0; 100*mean(lagGDP)];
ssEA.P0 = blkdiag(ssE.P0, 0.2);

%% Estimate
[ssMOpt, ~, thetaMOpt] = ssEA.estimate(YM, thetaR, XM, WM);
