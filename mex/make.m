function make
% Makes the .mex files for StateSpace.m

% David Kelley, 2016

fprintf('\nBuilding .mex files for StateSpace.\n');

clear mex; %#ok<CLMEX>

% Get folders
baseDir =  [subsref(strsplit(mfilename('fullpath'), 'StateSpace'), ...
  struct('type', '{}', 'subs', {{1}})) 'StateSpace'];
outputFolder = fullfile(baseDir, '+ss_mex');
if ~exist(outputFolder, 'dir')
  mkdir(outputFolder);
end

srcFolder = fullfile(baseDir, 'mex');
blaslib = fullfile(matlabroot, 'extern', 'lib', ...
  computer('arch'), 'microsoft', 'libmwblas.lib');
lapacklib = fullfile(matlabroot, 'extern',  'lib', ...
  computer('arch'), 'microsoft', 'libmwlapack.lib');

% Compile
flags = {'-O', '-largeArrayDims', '-outdir', outputFolder};
mex(flags{:}, fullfile(srcFolder, 'kfilter_uni.cpp'), blaslib, lapacklib);
mex(flags{:}, fullfile(srcFolder, 'ksmoother_uni.cpp'), blaslib, lapacklib);
mex(flags{:}, fullfile(srcFolder, 'kfilter_multi.cpp'), blaslib, lapacklib);
mex(flags{:}, fullfile(srcFolder, 'ksmoother_multi.cpp'), blaslib, lapacklib);
fprintf('\n');

% Test
results_uni = runtests(fullfile(baseDir, 'test', 'mex_univarite_test.m'));
results_multi = runtests(fullfile(baseDir, 'test', 'mex_multivarite_test.m'));

% Report
if all(~[results_uni.Failed]) && all(~[results_multi.Failed])
  fprintf('\nCompleted mex comilation. All tests pass.\n');
end
