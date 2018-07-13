
addpath 'O:\PROJ_LIB\Presentations\Chartbook\Data\Dataset Creation\cbd'

data = cbd.trim(cbd.data({'DIFFL(GDPC1)', 'DIFFL(PAYEMS)', 'DIFFL(W875RX1)', ...
  'DIFFL(INDPRO)', 'DIFFL(CMRMTSPL)'}, 'dbID', 'FRED'), ...
  'startDate', '1/1/1967', 'endDate', '1/1/2000');
y = data{:,:};

%% Model
N1 = 1;
N2 = 4;
N = N1 + N2;
p = 4;
q = 3;

beta1 = 1;
scalar3 = [1 2 3 2 1] ./ 3;

H = [[beta1.*scalar3 scalar3];
  [nan(N2,1) zeros(N2,4) zeros(N2,5)]];

Fa = [nan(1,p) zeros(1,5-p); eye(4) zeros(4,5-p)];
Fb = [nan(1,q) zeros(1,5-q); eye(4) zeros(4,1)];
F = blkdiag(Fa, Fb);

Sigma1 = diag(nan(2,1));
Sigma2 = blkdiag(0, diag(nan(N2,1)));

G = [1 0; 
  zeros(4,2); 
  0 1;
  zeros(4,2)];

mu = nan(N,1);

ssMM = StateSpaceEstimation(H, Sigma2, F, Sigma1, 'R', G, 'd', mu);
ssMM.a0 = zeros(10,1);
ssMM.P0 = 10*eye(10);
ssMMOpt = ssMM.estimate(y);

alphaMM = ssMMOpt.smooth(y);

%% Our version of their model
Z = [[1; nan(4,1)], zeros(5, 3) [1; zeros(4,1)] zeros(5,3)];
d = nan(5,1);
H = blkdiag(0, diag(nan(4,1)));

T = [nan(1,4), zeros(1,4);
     eye(3), zeros(3,5); 
     zeros(1,4) nan(1,4);
     zeros(3,4) eye(3) zeros(3,1)];

Q = diag(nan(2,1));
R = [1, 0; zeros(3,2); 0 1; zeros(3,2)];

ssE = StateSpaceEstimation(Z, H, T, Q, 'R', R, 'd', d);
accum = Accumulator.GenerateRegular(y, {'avg', '', '', '', ''}, [3 1 1 1 1]);
ssEA = accum.augmentStateSpaceEstimation(ssE);

ssEA.a0 = zeros(ssEA.m,1);
ssEA.P0 = 10*eye(ssEA.m);

%% Estimate
ssML = ssEA.estimate(y);
alphaA = ssML.smooth(y);

%% Plot
plot([alphaMM(:,1) alphaA(:,1)]);
