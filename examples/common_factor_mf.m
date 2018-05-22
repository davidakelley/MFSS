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

ssLB = ssE.ThetaMapping.theta2system(-inf(ssE.ThetaMapping.nTheta,1));
ssLB.T = -1;
ssUB = ssE.ThetaMapping.theta2system(inf(ssE.ThetaMapping.nTheta,1));
ssUB.T = 1;
ssE.ThetaMapping = ssE.ThetaMapping.addRestrictions(ssLB, ssUB);

ssEA = accum.augmentStateSpaceEstimation(ssE);
% ss0A = accum.augmentStateSpace(StateSpace([1;1], [0;0], eye(2), 2, 0, 1, 1));

% Estimate
ssML = ssEA.estimate(y');
