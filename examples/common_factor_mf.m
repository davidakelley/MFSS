% Create accumulator
accum = Accumulator.GenerateRegular(y, {'avg', ''}, [3 1]);

% Put nan values for each estimated quantity
Z = [1; nan(1,1)];
d = zeros(2,1);
H = diag(nan(2,1));
T = nan;
c = 0;
R = 1;
Q = nan;

% Create estimation object
ssE = StateSpaceEstimation(Z, d, H, T, c, R, Q);
ssEA = accum.augmentStateSpaceEstimation(ssE);
% 
% % Create initial values
% Z0 = ones(2,1);
% H0 = eye(2);
% T0 = 1;
% Q0 = 1;
% ss0 = StateSpace(Z0, d, H0, T0, c, R, Q0);
% ss0A = accum.augmentStateSpace(ss0);

% Estimate
ssML = ssEA.estimate(y');   