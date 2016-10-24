function result = make
% Makes the .mex file for StateSpace.m

% David Kelley, 2016

clear mex; %#ok<CLMEX>

% Get folders
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
fprintf('\n');

% Test
addpath(fullfile(outputFolder, 'test'));
result = runtests('mex_univarite_test');
rmpath(fullfile(outputFolder, 'test'));