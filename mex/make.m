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

switch computer('arch')
  case 'win64'
    blaslib = fullfile(matlabroot, 'extern', 'lib', ...
      computer('arch'), 'microsoft', 'libmwblas.lib');
    lapacklib = fullfile(matlabroot, 'extern',  'lib', ...
      computer('arch'), 'microsoft', 'libmwlapack.lib');
    
  case {'maci64', 'glnxa64'}
    blaslib = '-lblas';
    lapacklib = '-llapack';
    
end

% Compile
flags = {'-O', '-outdir', outputFolder};
mex(flags{:}, fullfile(srcFolder, 'filter_uni.cpp'), blaslib, lapacklib);
mex(flags{:}, fullfile(srcFolder, 'smoother_uni.cpp'), blaslib, lapacklib);
fprintf('\n');

% Test
results = runtests(fullfile(baseDir, 'test', 'kalman_test.m'));
if all(~[results.Failed])
  fprintf('\nCompleted mex comilation. All tests pass.\n');
end
