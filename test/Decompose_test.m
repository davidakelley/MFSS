% Test the Kalman weight decompositions
% David Kelley, 2017

classdef Decompose_test < matlab.unittest.TestCase
  
  properties
    data = struct;
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      baseDir = fileparts(fileparts(mfilename('fullpath')));
      addpath(baseDir);
      addpath(fullfile(baseDir, 'examples'));
      
      data_load = load(fullfile(baseDir, 'examples', 'data', 'dk.mat'));
      testCase.data.nile = data_load.nile;
    end
  end
  
  methods (Test)
    function testAR(testCase)
      % Do a stationary test 
      ss = generateARmodel(1, 1, false);
      y = generateData(ss, 150);
      
      [decomp_data, decomp_const] = ss.decompose_filtered(y);
      
      testCase.verifyEqual(decomp_data, y, 'AbsTol', 1e-4);
    end
    
    function testNile(testCase)
      % Use estimated model from DK
      % This is non-stationary and so probably harder than the AR test.
      ss = StateSpace(1, 0, 15099, 1, 0, 1, 1469.1);
      y = testCase.data.nile';
      
      decomp_data = ss.decompose_smoothed(y);
      
      testCase.verifyEqual(decomp_data, y, 'AbsTol', 1e-4);
    end
  end
end