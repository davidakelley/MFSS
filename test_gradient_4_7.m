% temp_test.m
% Add the paths the first time you run this:
%{
 addpath(fullfile(pwd, 'test'))
 addpath(fullfile(pwd, 'examples'))
%}

% Current limitations: 
%   All states must be stationary. Haven't gotten to the exact initial yet.

% Suspicions:
%   There's something wrong with T. 
%     Reasoning: When T is nonzero, everything blows up. We can mitigate the
%     problem by making other (Z, RQR', etc.) values small.
%   There's something wrong with G(y_t,i). 
%     Reasoning: With p > 1 the last observation works but for i < p is off.

clear; home; 
fprintf('Testing analytic univariate gradient.\n\n')
p = 1; 
m = 1;
g = 1;

%% Set 1
fprintf('<strong>Set 1 (T = 0) </strong>\n');

% Observation equation
Z = ones(p, m);
d = ones(p, 1);
Hchol = diag(ones(p, 1)) + tril(ones(p));
Hchol(1:p+1:end) = abs(Hchol(1:p+1:end));
H = Hchol * Hchol';
H = diag(diag(H));

% State equation
T = zeros(m);
c = 0 * ones(m, 1);
R = 1 * ones(m, g);
Q = 1 * diag(ones(g,1));

% Set up test
ss = StateSpace(Z, d, H, T, c, R, Q);
test_compareGradient(ss);

return

%% Set 2
fprintf('<strong>Set 2 (T != 0) </strong>\n');
% Observation equation
Z = 2 * ones(p, m);
d = ones(p, 1);
H = 1 * diag(ones(p, 1));

% State equation
T_temp = 2 * eye(m) + diag(abs(ones(m,1))) + 0.1 * ones(m);
T = T_temp ./ (abs(max(eig(T_temp))) + 0.3);
c = 0 * ones(m, 1);
R = 1 * abs(ones(m, g));
Q = 1 * diag(diag(abs(ones(g))));

% Set up test
ss = StateSpace(Z, d, H, T, c, R, Q);
test_compareGradient(ss);
