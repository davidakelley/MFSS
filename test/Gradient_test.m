%% Tests for StateSpace.gradient
%
% Run with
%{
  testOut = runtests('test/gradient_test.m');
%}

% David Kelley, 2016

classdef gradient_test < matlab.unittest.TestCase
  
  properties
    delta = 1e-5;
    tol = 5e-3;
    bbk
  end
  
  methods
    function printNice(testCase, ss, tm, analytic, numeric) %#ok<INUSL>
%       paramLen = structfun(@length, ss.thetaMap.elem);
      systemParam = {'Z', 'd', 'H', 'T', 'c', 'R', 'Q', 'a0', 'P0'}';
      separateNames = arrayfun(@(len, name) repmat(name, [len 1]), tm.nTheta, systemParam, 'Uniform', false);
      nameVec = cat(1, separateNames{:});
      nameVec(~ss.thetaMap.estimated) = [];
      
      out = [{'Param', 'Analytic', 'Numeric', 'Difference', 'Relative'}; ...
        [nameVec num2cell(analytic) num2cell(numeric) num2cell(analytic - numeric) num2cell(analytic ./ numeric -1)]];
      disp(out);
    end
    
    function numeric = numericGradient(testCase, ss, tm, y)
      % Iterate through theta, find changes in likelihood
      
      numeric = nan(tm.nTheta, 1);
      [~, logl_fix] = ss.filter(y);
      
      theta = tm.system2theta(ss);
      for iT = 1:tm.nTheta
        iTheta = theta;
        iTheta(iT) = iTheta(iT) + testCase.delta;
        
        ssTest = tm.theta2system(iTheta);

        [~, logl_delta] = ssTest.filter(y);
        numeric(iT) = (logl_delta - logl_fix) ./ testCase.delta;
      end
    end
  end  
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      [data, ~, dims] = loadFactorModel();
      addpath('C:\Users\g1dak02\Documents\MATLAB\StateSpace');

      % Subset the model to make it more managable:
      dims.nSeries = 92;
      
      y = data.indicators(:, 1:dims.nSeries)';
      y(:, any(isnan(y), 1)) = [];
      
      testCase.bbk = struct('data', data, 'dims', dims, 'y', y);
    end
  end
  
  methods (Test)
    function testGeneratedLLM(testCase)
      p = 1; m = 2; timeDim = 500;
      ss = generateARmodel(p, m, false);
      y = generateData(ss, timeDim);
      
      ss = ss.checkSample(y);
      ss = ss.setDefaultInitial();
      
      ss.usingDefaulta0 = false;
      ss.usingDefaultP0 = false;
      ss.P0 = eye(ss.m) * 1e7;
      tm = ThetaMap.ThetaMapAll(ss);
      
      tic;
      [~, analytic] = ss.gradient(y, tm);
      time_a = toc;
      tic;
      numeric = testCase.numericGradient(ss, tm, y);
      time_n = toc;
      fprintf(['\nModel: %d series, %d states, t = %d\n' ...
        'Analytic gradient took %3.2f%% of the time as the numeric version.\n'],...
        p, m, timeDim, 100*(time_a/time_n));
      
      testCase.printNice(ss, tm, analytic, numeric);
      testCase.verifyEqual(analytic, numeric, 'RelTol', testCase.tol);
    end
    
    function testGeneratedMedium(testCase)
      p = 4; m = 4; timeDim = 500;
      ss = generateARmodel(p, m, false);
      y = generateData(ss, timeDim);
      
      ss = ss.checkSample(y);
      ss = ss.setDefaultInitial();
      ss.usingDefaulta0 = false;
      ss.usingDefaultP0 = false;
      ss.P0 = eye(ss.m) * 1e7;
      
      tm = ThetaMap.ThetaMapAll(ss);
      
      tic;
      [~, analytic] = ss.gradient(y, tm);
      time_a = toc;
      tic;
      numeric = testCase.numericGradient(ss, tm, y);
      time_n = toc;
      
      testCase.printNice(ss, tm, analytic, numeric);
      fprintf(['\nModel: %d series, %d states, t = %d\n' ...
        'Analytic gradient took %3.2f%% of the time as the numeric version.\n'],...
        p, m, timeDim, 100*(time_a/time_n));
      testCase.verifyEqual(analytic, numeric, 'RelTol', testCase.tol);
    end
    
    function testGeneratedLarge(testCase)
      p = 20; m = 2; timeDim = 500;
      ss = generateARmodel(p, m, false);
      y = generateData(ss, timeDim);
      
      ss = ss.checkSample(y);
      ss = ss.setDefaultInitial();
      ss.usingDefaulta0 = false;
      ss.usingDefaultP0 = false;
      ss.P0 = eye(ss.m) * 1e7;
      
      tm = ThetaMap.ThetaMapAll(ss);
      
      tic;
      [~, analytic] = ss.gradient(y, tm);
      time_a = toc;
      tic;
      numeric = testCase.numericGradient(ss, tm, y);
      time_n = toc;
      
      testCase.printNice(ss, tm, analytic, numeric);
      fprintf(['\nModel: %d series, %d states, t = %d\n' ...
        'Analytic gradient took %3.2f%% of the time as the numeric version.\n'],...
        p, m, timeDim, 100*(time_a/time_n));
      testCase.verifyEqual(analytic, numeric, 'RelTol', testCase.tol);
     end
    
    function testFactorModel(testCase)
      % Set up state
      rnfacs = 2;
      nSeries = testCase.bbk.dims.nSeries;
      nlags = testCase.bbk.dims.nlags;
      
      Z = [nan(nSeries, rnfacs) zeros(nSeries, rnfacs * (nlags-1))];
      d = zeros(nSeries, 1);
      H = eye(nSeries);
      H(H == 1) = nan;
      
      T = [nan(rnfacs, rnfacs * nlags);
        eye(rnfacs * (nlags-1)) zeros(rnfacs)];
      c = zeros(rnfacs * nlags, 1);
      R = [eye(rnfacs); zeros(rnfacs * (nlags-1), rnfacs)];
      Q = nan(rnfacs);
      
      ss = StateSpaceEstimation(Z, d, H, T, c, R, Q, []);
      ss.usingDefaulta0 = false;
      ss.a0 = zeros(ss.m, 1);
      ss.usingDefaultP0 = false;
      ss.P0 = eye(ss.m) * 1e7;
           
      % Initial values
      [Z(:, 1:rnfacs), f0] = pca(testCase.bbk.y', 'NumComponents', rnfacs);
      f0(any(isnan(f0), 2), :) = [];
      
      H = diag(var(testCase.bbk.y' - f0 * Z(:, 1:rnfacs)'));
      
      y_var = f0(nlags+1:end, :);
      assert(nlags == 2);
      x = [f0(2:end-1, :) f0(1:end-2, :)];
      
      yTx = y_var' * x;
      xTx = x' * x;
      yTy = y_var' * y_var;
      
      T(1:rnfacs, :) = yTx / xTx;
      Q = (yTy - yTx / xTx * yTx') ./ size(testCase.bbk.y, 1);
      
      ss0 = StateSpace(Z, d, H, T, c, R, Q, []);
      
      % Set up 
      ss0 = ss0.checkSample(testCase.bbk.y);
      ss0 = ss0.setDefaultInitial();
      
      tm = ThetaMap.ThetaMapAll(ss0);
      
      % Test
      tic;
      [~, analytic] = ss0.gradient(testCase.bbk.y, tm);
      time_a = toc;
      tic;
      numeric = testCase.numericGradient(ss0, tm, testCase.bbk.y);
      time_n = toc;

      testCase.printNice(ss, tm, analytic, numeric);
      fprintf(['\nModel: %d series, %d states, t = %d (diagonal H)\n' ...
        'Analytic gradient took %3.2f%% of the time as the numeric version.\n'],...
        nSeries, rnfacs, length(testCase.bbk.y), 100*(time_a/time_n));
      testCase.verifyEqual(analytic, numeric, 'RelTol', 0.02);
    end
  end
end
