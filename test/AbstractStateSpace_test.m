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
      ss = StateSpace(1, 1, 1, 1);
     
      ss = ss.setInvariantTau();
      lags = ss.LagsInState(1);
      testCase.verifyEqual(lags, 0);
    end
    
    function testTVPInputZ(testCase)
      % Test that we can give a single TVP input and the rest not
      tauVec = [ones(9,1); repmat(2,9,1)];
      Z = struct('Zt', cat(3, [1; 1], [10; 10]), 'tauZ', tauVec);
      H = zeros(2);
      T = 0.9;
      Q = 1;
      
      ss = StateSpace(Z, H, T, Q);
      [y, alpha] = generateData(ss);
      testCase.verifyEqual(y', ...
        repmat(alpha' .* (1 + (tauVec-1).*9), [1 2]))
    end
    
    function testTVPInputQ(testCase)
      % Test that we can give a single TVP input and the rest not
      Z = ones(2,1);
      H = zeros(2);
      T = 0.9;
      tauVec = [ones(500,1); repmat(2,501,1)];
      Q = struct('Qt', cat(3, .1, 10), 'tauQ', tauVec);
      
      ss = StateSpace(Z, H, T, Q);
      [~, alpha] = generateData(ss);
      testCase.verifyGreaterThan(var(alpha(501:end)), var(alpha(1:500)));
    end
    
  end
end
