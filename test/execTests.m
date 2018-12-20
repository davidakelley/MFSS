function result = execTests(tests)
% Run all of the tests for MFSS 
%
% Optional Arguments:
% tests (cell array of strings): 
% 	Contains the shortcuts to run subsets of the tests. Valid options are
% 	'basic', 'kalman', 'Accumulator', 'ml', and 'decomp'. Defaults to {'basic',
% 	'kalman', 'Accumulator', 'ml', 'decomp'}

% David Kelley, 2016-2018

defaultTests = {'basic', 'kalman', 'Accumulator', 'ml', 'decomp'};
if nargin == 0
  tests = defaultTests;
end

baseDir = fileparts(fileparts(mfilename('fullpath')));
srcDir = fullfile(baseDir, 'src');
addpath(srcDir);

%% Run tests
testDir = fullfile(baseDir, 'test');

basicTests = setTests(testDir, {'AbstractSystem', 'AbstractStateSpace'});
kalmanTests = setTests(testDir, {'kalman'});
accumulatorTests = setTests(testDir, {'Accumulator', 'Accumulator_Integration'});
mlTests = setTests(testDir, {'estimate', 'ThetaMap', 'mfvar'});
decompTests = setTests(testDir, {'Decompose'});

alltests = {basicTests kalmanTests accumulatorTests mlTests decompTests};
selectedTests = alltests(ismember(defaultTests, tests));
suite = [selectedTests{:}];

runner = matlab.unittest.TestRunner.withTextOutput;
runner.addPlugin(matlab.unittest.plugins.CodeCoveragePlugin.forFolder(srcDir));
result = runner.run(suite);

display(result);

end

function testSuite = setTests(testDir, testScripts)
% helper function to set tests up
%
% Arguments:
%  testScripts (cell array of strings):  Name of test files without _test.m

import matlab.unittest.TestSuite
import matlab.unittest.TestRunner
import matlab.unittest.plugins.CodeCoveragePlugin

testsCell = cellfun( ...
  @(test) TestSuite.fromFile(fullfile(testDir, [test '_test.m'])), ...
  testScripts, 'Uniform', false);
testSuite = [testsCell{:}];

end
