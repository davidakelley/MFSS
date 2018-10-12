function result = execTests(tests)
% EXECTESTS runs all of the tests for StateSpace.
% Pass a cell array of strings containing the shortcuts to run subsets of
% the tests. Subsets include 'mex', 'gradient', and 'ml'.

% David Kelley, 2016

defaultTests = {'basic', 'kalman', 'Accumulator', 'ml', 'decomp'};
if nargin == 0
  tests = defaultTests;
end

baseDir = fileparts(fileparts(mfilename('fullpath')));
srcDir = fullfile(baseDir, 'src');
addpath(srcDir);

import matlab.unittest.TestSuite
import matlab.unittest.TestRunner
import matlab.unittest.plugins.CodeCoveragePlugin

%% Run tests
testDir = fullfile(baseDir, 'test');

basicTests = [TestSuite.fromFile(fullfile(testDir, 'AbstractSystem_test.m')), ...
  TestSuite.fromFile(fullfile(testDir, 'AbstractStateSpace_test.m'))];
kalmanTests = TestSuite.fromFile(fullfile(testDir, 'kalman_test.m'));
accumulatorTests = [TestSuite.fromFile(fullfile(testDir, 'Accumulator_test.m')), ...
  TestSuite.fromFile(fullfile(testDir, 'Accumulator_IntegrationTest.m'))];
mlTests = [TestSuite.fromFile(fullfile(testDir, 'estimate_test.m')), ...
  TestSuite.fromFile(fullfile(testDir, 'ThetaMap_test.m')), ...
  TestSuite.fromFile(fullfile(testDir, 'mfvar_test.m'))];
decompTests = TestSuite.fromFile(fullfile(testDir, 'Decompose_test.m'));

alltests = {basicTests kalmanTests accumulatorTests ...
  mlTests decompTests};
selectedTests = alltests(ismember(defaultTests, tests));
suite = [selectedTests{:}];

runner = TestRunner.withTextOutput;
runner.addPlugin(CodeCoveragePlugin.forFolder(srcDir));
result = runner.run(suite);

display(result);