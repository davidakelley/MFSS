% Tests for StateSpace.gradient
%
% Run with
%{
  testOut = runtests('test/Gradient_test.m');
%}

% David Kelley, 2016

classdef Gradient_test < matlab.unittest.TestCase
  
  properties
    y
    ss
    logl
    gradient
    delta = 1e-5;
  end
  
  methods(TestClassSetup)    
    function setupOnce(testCase)
      addCBD;      
      addpath('C:\Users\g1dak02\Documents\MATLAB\StateSpace');
      data_pull = cbd.data({'DIFF%(IP)', 'FRBCNAIM@SURVEYS'}, ...
        'startDate', '1/1/1980');
      dm_data = cbd.stddm(data_pull);
      testCase.y = dm_data{:,:}';
            
      Z = [1; 1];
      d = zeros(2, 1);
      H = [.5, .4; .4, .5];
      T = 0.9;
      c = 0;
      R = 1;
      Q = 1000;
      testCase.ss = StateSpace(Z, d, H, T, c, R, Q, []);
      testCase.ss = testCase.ss.checkSample(testCase.y);
      testCase.ss = testCase.ss.setDefaultInitial();
      testCase.ss = testCase.ss.generateThetaMap();
      
      [testCase.logl, testCase.gradient] = testCase.ss.gradient(testCase.y);
    end
  end
  
  methods (Test)
    function testZ(testCase)
      theta = testCase.ss.getParamVec();
      grad = nan(size(theta));
      
      for iT = 1:size(theta, 1)
        iTheta = theta;
        iTheta(iT) = iTheta(iT) + testCase.delta;
        
        [ssTest, a0_theta, P0_theta] = testCase.ss.theta2system(iTheta);
        
        [~, loglZ1] = ssTest.filter(testCase.y, a0_theta, P0_theta);
        grad(iT) = (loglZ1 - testCase.logl) ./ testCase.delta;
      end
      
      disp('Percentage errors:');
      disp((grad - testCase.gradient) ./ testCase.gradient * 100);
      testCase.verifyEqual(testCase.gradient, grad, 'RelTol', 0.05);
    end
  end
end
