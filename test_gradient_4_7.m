% temp_test.m
% Add the paths the first time you run this:
%{
 addpath(fullfile(pwd, 'test'))
 addpath(fullfile(pwd, 'examples'))
%}
% Current limitations: 
%   All states must be stationary. Haven't gotten to the exact initial yet.

clear; home; 
fprintf('Testing analytic univariate gradient.\n\n')
p = 3; 
m = 2;
g = 1;

%% Set 1

fprintf('<strong>Set 1 (T = 0) </strong>\n');

% Observation equation
Z = ones(p, m);
d = 0 * ones(p, 1);
Hchol = diag(0.5 * ones(p, 1)) + 0.5 * tril(ones(p));
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

% return

%% Set 2
fprintf('<strong>Set 2 (T != 0) </strong>\n');
% Observation equation
Z = 1 * ones(p, m);
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
