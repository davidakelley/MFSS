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
  
  methods
    function printNice(testCase, grad)
      paramLen = structfun(@length, testCase.ss.thetaMap.elem);
      systemParam = {'Z', 'd', 'H', 'T', 'c', 'R', 'Q', 'a0', 'P0'}';
      separateNames = arrayfun(@(len, name) repmat(name, [len 1]), paramLen, systemParam, 'Uniform', false);
      nameVec = cat(1, separateNames{:});
      
      out = [{'Param', 'Analytic', 'Numeric', 'Difference'}; ...
        [nameVec num2cell(testCase.gradient) num2cell(grad) num2cell(testCase.gradient - grad)]];
      disp(out);
    end
    
  end
  
  methods(TestClassSetup)    
    function setupOnce(testCase)
      testCase.ss = generateARmodel(4, 2);
      testCase.y = generateData(testCase.ss, 500);
      
      testCase.ss = testCase.ss.checkSample(testCase.y);
      testCase.ss = testCase.ss.setDefaultInitial();
      testCase.ss = testCase.ss.generateThetaMap();
      
      [testCase.logl, testCase.gradient] = testCase.ss.gradient(testCase.y);
    end
  end
  
  methods (Test)
    function testNumeric(testCase)
      theta = testCase.ss.getParamVec();
      grad = nan(size(theta));
      
      [~, logl_fix] = testCase.ss.filter(testCase.y);

      for iT = 1:size(theta, 1)
        iTheta = theta;
        iTheta(iT) = iTheta(iT) + testCase.delta;
        
        [ssTest, a0_theta, P0_theta] = testCase.ss.theta2system(iTheta);
        
        [~, logl_delta] = ssTest.filter(testCase.y, a0_theta, P0_theta);
        grad(iT) = (logl_delta - logl_fix) ./ testCase.delta;
      end
      testCase.printNice(grad);
     
      testCase.verifyEqual(testCase.gradient, grad, 'RelTol', 0.01);
    end
  end
end
