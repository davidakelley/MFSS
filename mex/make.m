% make.m
% Makes the .mex file for StateSpace.m

% David Kelley, 2016

clear; clear functions;  %#ok<CLFUNC>

% Folders
outputFolder = [subsref(strsplit(pwd, 'StateSpace'), ...
  struct('type', '{}', 'subs', {{1}})) 'StateSpace'];
srcFolder = fullfile(outputFolder, 'mex');
blaslib = fullfile(matlabroot, 'extern', 'lib', ...
  computer('arch'), 'microsoft', 'libmwblas.lib');
lapacklib = fullfile(matlabroot, 'extern',  'lib', ...
  computer('arch'), 'microsoft', 'libmwlapack.lib');

% Compile
flags = {'-O', '-largeArrayDims', '-outdir', outputFolder};
mex(flags{:}, fullfile(srcFolder, 'kfilter_uni.cpp'), blaslib, lapacklib);
mex(flags{:}, fullfile(srcFolder, 'ksmoother_uni.cpp'), blaslib, lapacklib);

%% Set up test
g = 1; m = 2; p = 2; timeDim = 600;

Z = [1 0; 1.1 0];
d = zeros(2, 1);
H = [.1 0; 0 .3];
T = [0.6, -0.2; 1 0];
c = [0; 0];
R = [1; -0];
Q = 0.1;

% Generate data
eta = R * Q^(1/2) * randn(g, timeDim);
alpha = nan(m, timeDim);
alpha(:,1) = eta(:,1);
for iT = 2:timeDim
  alpha(:,iT) = T * alpha(:,iT-1) + eta(:,iT);
end

epsilon = H^(1/2) * randn(p, timeDim);
Y = nan(p, timeDim);
for iT = 1:timeDim
  Y(:,iT) = Z * alpha(:,iT) + epsilon(:,iT);
end

ss = StateSpace(Z, d, H, T, c, R, Q, []);

%% Run filter
ss.useMex = false;
[a_m, logl_m, fOut_m] = ss.filter(Y);
ss.useMex = true;
[a, logl, fOut] = ss.filter(Y);

% Assertions
eps = 1e-13;
closeMat = @(x,y) all(all(x - y < eps));
closeCube = @(x,y) all(all(all(x - y < eps)));

assert(closeMat(logl, logl_m));
assert(closeMat(a, a_m));
assert(closeCube(fOut.P, fOut_m.P));
assert(closeMat(fOut.v, fOut_m.v));
assert(closeMat(fOut.F, fOut_m.F));
assert(closeCube(fOut.M, fOut_m.M));
assert(closeCube(fOut.K, fOut_m.K));
assert(closeCube(fOut.L, fOut_m.L));

fprintf('\nUnivariate filter tests pass.\n');

%% Run smoother
ss.useMex = false;
[alpha_m, sOut_m] = ss.smooth(Y);
ss.useMex = true;
[alpha, sOut] = ss.smooth(Y);

% Assertions
eps = 1e-13;
closeMat = @(x,y) all(all(x - y < eps));
closeCube = @(x,y) all(all(all(x - y < eps)));

assert(closeMat(alpha, alpha_m));
assert(closeCube(sOut.eta, sOut_m.eta));
assert(closeMat(sOut.r, sOut_m.r));
assert(closeCube(sOut.N, sOut_m.N));
assert(closeCube(sOut.V, sOut_m.V));
assert(closeCube(sOut.J, sOut_m.J));
assert(closeMat(sOut.logli, sOut_m.logli));

fprintf('Univariate smoother tests pass.\n');

%% Timing
ss.useMex = false;
filter_fn = @() ss.filter(Y); 
mTime_filter = timeit(filter_fn, 3);

ss.useMex = true;
filter_fn = @() ss.filter(Y); 
mexTime_filter = timeit(filter_fn, 3);

ss.useMex = false;
smooth_fn = @() ss.smooth(Y); 
mTime_smooth = timeit(smooth_fn, 2);

ss.useMex = true;
smooth_fn = @() ss.smooth(Y); 
mexTime_smooth = timeit(smooth_fn, 2);

fprintf(['\n' mfilename ' complete.\n']);
disp(' mex function produces same result as .m file.');
disp([' mex filter   takes ' num2str(mexTime_filter/mTime_filter*100) '% of the time as .m file.']);
disp([' mex smoother takes ' num2str(mexTime_smooth/mTime_smooth*100) '% of the time as .m file.']);
