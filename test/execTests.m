% execTests.m
% EXECTESTS runs all of the tests for the cbd toolbox

% David Kelley, 2015

thisFile = mfilename('fullpath');
baseDir = fileparts(fileparts(thisFile));

addpath(baseDir);
addpath(fullfile(baseDir, 'examples'));

import matlab.unittest.TestSuite
import matlab.unittest.TestRunner
import matlab.unittest.plugins.CodeCoveragePlugin

%% Run tests
suite = TestSuite.fromFolder([baseDir '\test']);
runner = TestRunner.withTextOutput;
runner.addPlugin(CodeCoveragePlugin.forFolder([baseDir '\src']));
result = runner.run(suite);

display(result);