% Tests for StateSpace.gradient
%
% Run with
%{
  testOut = runtests('test/Gradient_test.m');
%}

% David Kelley, 2016

classdef Gradient_test < matlab.unittest.TestCase
  
  properties
    delta = 1e-8;
  end
  
  methods
    function printNice(testCase, ss, analytic, numeric) %#ok<INUSL>
      paramLen = structfun(@length, ss.thetaMap.elem);
      systemParam = {'Z', 'd', 'H', 'T', 'c', 'R', 'Q', 'a0', 'P0'}';
      separateNames = arrayfun(@(len, name) repmat(name, [len 1]), paramLen, systemParam, 'Uniform', false);
      nameVec = cat(1, separateNames{:});
      
      out = [{'Param', 'Analytic', 'Numeric', 'Difference'}; ...
        [nameVec num2cell(analytic) num2cell(numeric) num2cell(analytic - numeric)]];
      disp(out);
    end
    
    function numeric = numericGradient(testCase, ss, y)
      theta = ss.getParamVec();
      numeric = nan(size(theta));
      
      [~, logl_fix] = ss.filter(y);

      for iT = 1:size(theta, 1)
        iTheta = theta;
        iTheta(iT) = iTheta(iT) + testCase.delta;
        
        [ssTest, a0_theta, P0_theta] = ss.theta2system(iTheta);
        
        [~, logl_delta] = ssTest.filter(y, a0_theta, P0_theta);
        numeric(iT) = (logl_delta - logl_fix) ./ testCase.delta;
      end
    end
  end
  
  methods (Test)
    function testNumericMedium(testCase)
      ss = generateARmodel(2, 2);
      y = generateData(ss, 500);
      
      ss = ss.checkSample(y);
      ss = ss.setDefaultInitial();
      ss = ss.generateThetaMap();
      
      tic;
      [~, analytic] = ss.gradient(y);
      time_a = toc;
      tic;
      numeric = testCase.numericGradient(ss, y);
      time_n = toc;
      
      testCase.printNice(ss, analytic, numeric);
      fprintf('Analytic gradient took %3.2f%% of the time as the numeric version.\n', 100*(time_a/time_n));
      testCase.verifyEqual(analytic, numeric, 'RelTol', 0.01);
    end
    
    function testNumericLLM(testCase)
      ss = generateARmodel(1, 2);
      y = generateData(ss, 500);
      
      ss = ss.checkSample(y);
      ss = ss.setDefaultInitial();
      ss = ss.generateThetaMap();
      
      tic;
      [~, analytic] = ss.gradient(y);
      time_a = toc;
      tic;
      numeric = testCase.numericGradient(ss, y);
      time_n = toc;
      fprintf('Analytic gradient took %3.2f%% of the time as the numeric version.\n', 100*(time_a/time_n));
      
      testCase.printNice(ss, analytic, numeric);
      testCase.verifyEqual(analytic, numeric, 'RelTol', 0.01);
    end
    
  end
end
