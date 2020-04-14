% Laubach-Williams estimate of the natural rate of interest
%
% See Also:
%   pgmtmfss_r_star.m - Mixed-frequency Laubach & Williams Natural Rate of Interest 

%% Data
YQ = [100*gdpQ(9:end) inflationQ(9:end)];
XQ = [100*gdpQ(8:end-1) 100*gdpQ(7:end-2) realrateQ(8:end-1) realrateQ(7:end-2) ...
  inflationQ(8:end-1) (inflationQ(7:end-2)+inflationQ(6:end-3)+inflationQ(5:end-4))/3 ...
  (inflationQ(4:end-5)+inflationQ(3:end-6)+inflationQ(2:end-7)+inflationQ(1:end-8))/4 ...
  oilQ(8:end-1) importQ(9:end)];

datesq = dates(9:end);

%% Model
% The observed data is organized
%     [GDP_t pi_t]'
% The state is organized as 
%     [GDP*_t GDP*_t-1 GDP*_t-2 g_t g_t-1 z_t z_t-1 r*_t]'
% The exogenous series in the measurement equation are 
%     [GDP_t-1 GDP_t-2 r_t-1 r_t-2 pi_t-1 avg1(pi)_t avg2(pi)_t oil_t-1 import_t]'

syms a1 a2 a3 b1 b2 b3 b4 b5 c sigma2Ystar sigma2IS sigma2PC

Z = [ 1 -a1 -a2 -c*a3*2 -c*a3*2 -a3/2 -a3/2 0;
  0 -b3  0   0   0   0   0 0];
Beta = [a1 a2 a3/2 a3/2 0   0   0   0   0;
  b3  0   0   0  b1 b2 1-b1-b2 b4 b5];
H = diag([sigma2IS sigma2PC]);

T = [1 0 0 1 0 0 0 0; 1 0 0 0 0 0 0 0; 0 1 0 0 0 0 0 0; ...
  0 0 0 1 0 0 0 0; 0 0 0 1 0 0 0 0; ...
  0 0 0 0 0 1 0 0; 0 0 0 0 0 1 0 0; ...
  0 0 0 4*c 0 1 0 0];
R = [lambda_g 1 0; 0 0 0; 0 0 0; ...
  lambda_g 0 0; 0 0 0; ...
  0 0 lambda_z/a3; 0 0 0; ...
  lambda_g*c*2 0 lambda_z/a3];
Q = diag([sigma2Ystar, sigma2Ystar, sigma2IS]);

ssEQ = StateSpaceEstimation(Z, H, T, Q, 'beta', Beta, 'R', R);

% Set up parameter bounds
% Bounds on slope of the Phillips curve
ssEQ.ThetaMapping = ssEQ.ThetaMapping.addStructuralRestriction(a3, [], -0.0025);
ssEQ.ThetaMapping = ssEQ.ThetaMapping.addStructuralRestriction(b3, 0.025, []);

%% Initial state
ssEQ.a0 = [809.7823659, 808.8428725, 807.9033792, 0.9394933, 0.9394933, 0, 0, 4*0.9394933]';
ssEQ.P0 = [0.7552736	0.2	0	0.2000498	0.2	0 0 0;
  0.2 0.2	0	0 0	0 0 0;
  0	0	0.2	0 0	0 0 0;
  0.2	0	0	0.2	0.2	0 0 0;
  0.2 0	0	0.2 0.2	0 0 0;
  0 0	0	0 0	0.2694159	0.2 0;
  0 0	0	0 0	0.2 0.2 0;
  0 0 0 0 0 0 0 0.2];

%% Estimate
[ssQOpt, ~, thetaOptQ] = ssEQ.estimate(YQ, thetaR, XQ);
