% Test basic functions of AbstractSystem and utility functions
% David Kelley, 2017

classdef AbstractStateSpace_test < matlab.unittest.TestCase
  
  properties
   
  end

  methods(TestClassSetup)
    function setupOnce(testCase) %#ok<MANU>
      baseDir = fileparts(fileparts(mfilename('fullpath')));
      addpath(baseDir);
    end
  end
  
  methods (Test)  
    function testLagsInStateSelf(testCase)
      % Test that a unit loading doesn't mess up LagsInState
      ss = StateSpace(1, 0, 1, 1, 0, 0, 1);
     
      ss = ss.setInvariantTau();
      lags = ss.LagsInState(1);
      testCase.verifyEqual(lags, 0);
    end
    
  end
end