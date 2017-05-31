function make
% Makes the .mex files for StateSpace.m
%
% Currently using armadillo 7.600.

% David Kelley, 2016

fprintf('\nBuilding .mex files for MFSS.\n');

clear mex; %#ok<CLMEX>

% Get folders
thisFile = mfilename('fullpath');
baseDir = thisFile(1:strfind(upper(thisFile), 'MFSS')+4);
outputFolder = fullfile(baseDir, '+mfss_mex');
if ~exist(outputFolder, 'dir')
  mkdir(outputFolder);
end

srcFolder = fullfile(baseDir, 'mex');
blaslib = fullfile(matlabroot, 'extern', 'lib', ...
  computer('arch'), 'microsoft', 'libmwblas.lib');
lapacklib = fullfile(matlabroot, 'extern',  'lib', ...
  computer('arch'), 'microsoft', 'libmwlapack.lib');

% Compile
flags = {'-O', '-outdir', outputFolder};
mex(flags{:}, fullfile(srcFolder, 'filter_uni.cpp'), blaslib, lapacklib);
mex(flags{:}, fullfile(srcFolder, 'smoother_uni.cpp'), blaslib, lapacklib);
fprintf('\n');

% Test
results_uni = runtests(fullfile(baseDir, 'test', 'mex_univariate_test.m'));
results_multi = runtests(fullfile(baseDir, 'test', 'mex_multivariate_test.m'));
% results_grad = runtests(fullfile(baseDir, 'test', 'mex_gradient_test.m'));

% Report
if all(~[results_uni.Failed]) && all(~[results_multi.Failed]) % && all(~[results_grad.Failed])
  fprintf('\nCompleted mex comilation. All tests pass.\n');
end
