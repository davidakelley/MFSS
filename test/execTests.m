function result = execTests(tests)
% EXECTESTS runs all of the tests for StateSpace. 
% Pass a cell array of strings containing the shortcuts to run subsets of
% the tests. Subsets include 'mex', 'gradient', and 'ml'.

% David Kelley, 2016 

defaultTests = {'basic', 'mex', 'accumulator', 'gradient', 'thetamap', 'ml'};
if nargin == 0
  tests = defaultTests;
end

baseDir =  [subsref(strsplit(mfilename('fullpath'), 'MFSS'), ...
  struct('type', '{}', 'subs', {{1}})) 'MFSS'];
addpath(baseDir);

import matlab.unittest.TestSuite
import matlab.unittest.TestRunner
import matlab.unittest.plugins.CodeCoveragePlugin

%% Run tests
testDir = [baseDir '\test'];

basicTests = TestSuite.fromFile(fullfile(testDir, 'AbstractSystem_test.m'));
mexTests = [TestSuite.fromFile(fullfile(testDir, 'mex_univariate_test.m')), ...
            TestSuite.fromFile(fullfile(testDir, 'mex_multivariate_test.m'))];
accumulatorTests = [TestSuite.fromFile(fullfile(testDir, 'accumulator_test.m')), ...
                    TestSuite.fromFile(fullfile(testDir, 'accumulatorIntegration_test.m'))];
gradientTests = TestSuite.fromFile(fullfile(testDir, 'gradient_test.m'));
thetaMapTests = TestSuite.fromFile(fullfile(testDir, 'thetamap_test.m'));
mlTests = TestSuite.fromFile(fullfile(testDir, 'estimate_test.m'));
          
alltests = {basicTests mexTests accumulatorTests gradientTests thetaMapTests mlTests};
selectedTests = alltests(ismember(defaultTests, tests));
suite = [selectedTests{:}];

runner = TestRunner.withTextOutput;
runner.addPlugin(CodeCoveragePlugin.forFolder(baseDir));
result = runner.run(suite);

display(result);