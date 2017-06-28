% Trend-cycle decomposition with generated data

%% Generate true model
% Observation equation: 
%   y = mu + psi + epsilon
%
% Transition: 
%   mu_t = mu_{t-1} + beta_{t-1}
%   beta_t = beta_{t-1} + xi
%   psi_t = rho * f(lambda) + kappa

% theta parameters
rho = .97;
lambda = 2 * pi ./ 24;
sigma_epsilon = .5;
sigma_xi = .001;
sigma_kappa = .5;

% State space parameters
Z = [1 0 1 0];
d = 0;
H = sigma_epsilon; 

T = blkdiag([1, 1; 0 1], rho * [cos(lambda) sin(lambda); -sin(lambda) cos(lambda)]);
c = zeros(4, 1);
R = [zeros(1, 3); eye(3)];
Q = diag([sigma_xi sigma_kappa sigma_kappa]);

% Generate data
ssTrue = StateSpace(Z, d, H, T, c, R, Q);
ssTrue.a0 = [1; .1; 0; 0];
[y, alpha] = generateData(ssTrue, 300);

% Extract signal
ssTrue.a0 = [];
[alphaHat, sOut, fOut] = ssTrue.smooth(y);
alphaHat(:, 1:fOut.dt) = nan;

% Plot
f1 = gcf;
subplot(4, 1, 1);
plot([alpha(1, :)' alphaHat(1, :)']);
title('mu');
subplot(4, 1, 2);
plot([alpha(2, :)' alphaHat(2, :)']);
title('beta');
subplot(4, 1, 3);
plot([alpha(3, :)' alphaHat(3, :)']);
title('psi');

subplot(4, 1, 4);
plot(y);
title('y');


%% Estimate with simulated data

Z = [1, 0, 1, 0];
d = 0;
H = nan;

T = blkdiag([1 1; 0 1], nan(2));
c = zeros(4,1);
R = [zeros(1, 3); eye(3)];
Q = diag(nan(3,1));

ssE = StateSpaceEstimation(Z, d, H, T, c, R, Q);

% The vector theta is going to be organized as 
%   [sigma_epsilon, sigma_xi, sigma_kappa, rho, lambda]'
% We'll get there by creating the transformations off of rho and lambda (as the
% nTheta+1 and nTheta+2 elements w.r.t the default nTheta) then call
% validateThetaMap to get rid of the unused theta elements.
%
% We're adding 3 elements of psi here and 2 elements of theta. Both are based
% off nTheta since that's the current size of both theta and psi.
nTheta = ssE.ThetaMapping.nTheta;
nPsi = nTheta;

% Indexes of which theta elements determine the particular psi element.
ssE.ThetaMapping.PsiIndexes{nPsi+1} = [nTheta+1 nTheta+2];
ssE.ThetaMapping.PsiIndexes{nPsi+2} = [nTheta+1 nTheta+2];
ssE.ThetaMapping.PsiIndexes{nPsi+3} = [nTheta+1 nTheta+2];

% Using the subset of the theta vector, construct the new elements of psi
ssE.ThetaMapping.PsiTransformation{nPsi+1} = @(theta) theta(1) * cos(theta(2));
ssE.ThetaMapping.PsiTransformation{nPsi+2} = @(theta) theta(1) * sin(theta(2));
ssE.ThetaMapping.PsiTransformation{nPsi+3} = @(theta) -theta(1) * sin(theta(2));

% ... and give the gradient of the elements of psi as a vector w.r.t. the theta
% elements we're using.
ssE.ThetaMapping.PsiGradient{nPsi+1} = @(theta) [cos(theta(2)), -theta(1) * sin(theta(2))]';
ssE.ThetaMapping.PsiGradient{nPsi+2} = @(theta) [sin(theta(2)), theta(1) * cos(theta(2))]';
ssE.ThetaMapping.PsiGradient{nPsi+3} = @(theta) [-sin(theta(2)), -theta(1) * cos(theta(2))]';

% We need to get back to theta from psi. The inverse functions to do so must
% take the whole psi vector and an index of which psi elements are used by
% that theta vector (these indexes will be automatically generated). 
%
% Note that if we generated initial values for the theta vector instead of the
% state space parameters, this step would be unnecessary.
ssE.ThetaMapping.PsiInverse{nTheta+1} = @(psi, inx) sqrt(psi(inx(1)).^2 + psi(inx(2)).^2);
ssE.ThetaMapping.PsiInverse{nTheta+2} = @(psi, inx) atan(psi(inx(2)) ./ psi(inx(1)));

% Assign the elements of psi to the state space parameters. We could also modify
% the transofrmation performed on the elements of psi at this point but this
% model doesn't require so.
ssE.ThetaMapping.index.T(3,3) = nPsi + 1;
ssE.ThetaMapping.index.T(3,4) = nPsi + 2;
ssE.ThetaMapping.index.T(4,3) = nPsi + 3;
ssE.ThetaMapping.index.T(4,4) = nPsi + 1;

% We also need to restrict the variance on the cycle to be determined by the
% same element of psi
ssE.ThetaMapping.index.Q(2,2) = ssE.ThetaMapping.index.Q(3,3);

% Remove unneccessary parts of theta and psi 
ssE.ThetaMapping = ssE.ThetaMapping.validateThetaMap();

% Initialization - parameters got orderd by the log of the 3 variances, 
% rho, lambda. Note that this initialization is intentionally quite poor
theta0 = ssE.ThetaMapping.system2theta(ssTrue);

% With the model set up, we're able to estimate the parameters
ssOpt = ssE.estimate(y, theta0);

%% Compare true parameters to estimated parameters
trueTheta = ssE.ThetaMapping.system2theta(ssTrue);
estTheta = ssE.ThetaMapping.system2theta(ssOpt);
disp([trueTheta, estTheta]);

[alphaHatOpt, ~, fOutOpt] = ssOpt.smooth(y);
alphaHat(:, 1:fOutOpt.dt) = nan;

%% Plot
subplot(4, 1, 1);
plot([alpha(1, :)' alphaHat(1, :)' alphaHatOpt(1, :)']);
title('mu');
subplot(4, 1, 2);
plot([alpha(2, :)' alphaHat(2, :)' alphaHatOpt(2, :)']);
title('beta');
subplot(4, 1, 3);
plot([alpha(3, :)' alphaHat(3, :)' alphaHatOpt(3, :)']);
title('psi');

subplot(4, 1, 4);
plot(y);
title('y');