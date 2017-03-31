% %temp_test.m
%{
addpath(fullfile(favDirs('mfss'), 'test'))
addpath(fullfile(favDirs('mfss'), 'examples'))
%}

% Current limitations: 
%   T must be zeros
%   p must be 1 

timePers = 200;
p = 1; 
m = 2;
g = 1;

rng('shuffle');
% Z = randn(p, m);
Z = ones(p, m);
% d = randn(p, 1);
d = ones(p, 1);
% Hchol = 1 + tril(randn(p));
% Hchol(1:p+1:end) = abs(Hchol(1:p+1:end));
% H = Hchol * Hchol';
% H = diag(diag(H));
H = diag(ones(p, 1));

T = 2 * eye(m) + diag(abs(randn(m,1))) + 0.1 * randn(m);
T = T ./ (abs(max(eig(T))) + 0.3);
% T = 0;
% T = zeros(m);
% T = 0.9;

% c = 0.1 * randn(m, 1);
c = zeros(m, 1);
% R = abs(randn(m, g));
R = 10 * ones(m, g);
% Q = diag(diag(abs(randn(g))));
Q = 0.01 * diag(ones(g,1));

ss = StateSpace(Z, d, H, T, c, R, Q);

rng(1);
[y, alpha] = generateData(ss, timePers);

tm = ThetaMap.ThetaMapAll(ss);
% tm.index.H = diag(diag(tm.index.H));
% tm.index.Z(:) = 0;
% tm.index.d(:) = 0;
% tm.index.H(:) = 0;
% tm.index.T(:) = 0;
% tm.index.c(:) = 0;
% tm.index.R(:) = 0;
% tm = tm.validateThetaMap();

plot(alpha');

tic;
[numeric, numericG] = numericGradient(ss, tm, y, 1e-8);
tocN = toc;

theta = tm.system2theta(ss);
tic;
[ll, grad] = ss.gradient(y, tm, theta);
tocA = toc;

fprintf('Analytic took %3.2f%% of the numeric time.\n', tocA./tocN*100);
diffTab = array2table([numeric grad (grad - numeric) (grad - numeric)./numeric], ...
  'VariableNames', {'Numeric', 'Analytic', 'Diff', 'RelativebDiff'});
diffTab.Variable = tm2matList(tm);
diffTab = diffTab(:,[5 1:4]);
disp(diffTab);

ss = ss.setDefaultInitial();

% ssE = StateSpaceEstimation(1, nan, nan, nan, nan, 1, nan);
% ssOpt = ssE.estimate(y, ss);