%% Tests for StateSpace.gradient
%
% Run with
%{
  testOut = runtests('test/Gradient_test.m');
%}

% David Kelley, 2016

classdef gradient_test < matlab.unittest.TestCase
  
  properties
    delta = 1e-5;
    tol = 5e-3;
    bbk
  end
  
  methods
    function printNice(testCase, ss, analytic, numeric) %#ok<INUSL>
      paramLen = structfun(@length, ss.thetaMap.elem);
      systemParam = {'Z', 'd', 'H', 'T', 'c', 'R', 'Q', 'a0', 'P0'}';
      separateNames = arrayfun(@(len, name) repmat(name, [len 1]), paramLen, systemParam, 'Uniform', false);
      nameVec = cat(1, separateNames{:});
      nameVec(~ss.thetaMap.estimated) = [];
      
      out = [{'Param', 'Analytic', 'Numeric', 'Difference', 'Relative'}; ...
        [nameVec num2cell(analytic) num2cell(numeric) num2cell(analytic - numeric) num2cell(analytic ./ numeric -1)]];
      disp(out);
    end
    
    function numeric = numericGradient(testCase, ss, y)
      theta = ss.getParamVec();
      nTheta = sum(ss.thetaMap.estimated);
      numeric = nan(nTheta, 1);
      
      [~, logl_fix] = ss.filter(y);

      thetaInds = find(ss.thetaMap.estimated);
      for iT = 1:nTheta
        iTheta = theta;
        iTheta(thetaInds(iT)) = iTheta(thetaInds(iT)) + testCase.delta;
        
        [ssTest, a0_theta, P0_theta] = ss.theta2system(iTheta);
        if isempty(a0_theta) && ~ss.usingDefaulta0
          a0_theta = ss.a0;
        end
        if isempty(P0_theta) && ~ss.usingDefaultP0
          P0_theta = ss.P0;
        end
        
        [~, logl_delta] = ssTest.filter(y, a0_theta, P0_theta);
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
      ss = ss.generateThetaMap();
      ss.thetaMap.estimated(:) = 1;

      ss.usingDefaulta0 = false;
      ss.usingDefaultP0 = false;
      ss.P0 = eye(ss.m) * 1e7;
      
      tic;
      [~, analytic] = ss.gradient(y);
      time_a = toc;
      tic;
      numeric = testCase.numericGradient(ss, y);
      time_n = toc;
      fprintf(['\nModel: %d series, %d states, t = %d\n' ...
        'Analytic gradient took %3.2f%% of the time as the numeric version.\n'],...
        p, m, timeDim, 100*(time_a/time_n));
      
%       testCase.printNice(ss, analytic, numeric);
      testCase.verifyEqual(analytic, numeric, 'RelTol', testCase.tol);
    end
    
    function testGeneratedMedium(testCase)
      p = 4; m = 4; timeDim = 500;
      ss = generateARmodel(p, m, false);
      y = generateData(ss, timeDim);
      
      ss = ss.checkSample(y);
      ss = ss.setDefaultInitial();
      ss = ss.generateThetaMap();
      ss.thetaMap.estimated(:) = 1;
      
      ss.usingDefaulta0 = false;
      ss.usingDefaultP0 = false;
      ss.P0 = eye(ss.m) * 1e7;
      
      tic;
      [~, analytic] = ss.gradient(y);
      time_a = toc;
      tic;
      numeric = testCase.numericGradient(ss, y);
      time_n = toc;
      
%       testCase.printNice(ss, analytic, numeric);
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
      ss = ss.generateThetaMap();
      ss.thetaMap.estimated(:) = 1;
      
      ss.usingDefaulta0 = false;
      ss.usingDefaultP0 = false;
      ss.P0 = eye(ss.m) * 1e7;
      
      tic;
      [~, analytic] = ss.gradient(y);
      time_a = toc;
      tic;
      numeric = testCase.numericGradient(ss, y);
      time_n = toc;
      
%       testCase.printNice(ss, analytic, numeric);
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
      
      ss = StateSpace(Z, d, H, T, c, R, Q, []);
      ss.usingDefaulta0 = false;
      ss.usingDefaultP0 = false;
      ss.P0 = eye(ss.m) * 1e7;
      
      ss0 = ss;
      
      % Initial values
      [ss0.Z(:, 1:rnfacs), f0] = pca(testCase.bbk.y', 'NumComponents', rnfacs);
      f0(any(isnan(f0), 2), :) = [];
      
      ss0.H = diag(var(testCase.bbk.y' - f0 * ss0.Z(:, 1:rnfacs)'));
      
      y_var = f0(nlags+1:end, :);
      assert(nlags == 2);
      x = [f0(2:end-1, :) f0(1:end-2, :)];
      
      yTx = y_var' * x;
      xTx = x' * x;
      yTy = y_var' * y_var;
      
      ss0.T(1:rnfacs, :) = yTx / xTx;
      ss0.Q = (yTy - yTx / xTx * yTx') ./ size(testCase.bbk.y, 1);
      
      % Set up 
      ss0 = ss0.checkSample(testCase.bbk.y);
      ss0 = ss0.setDefaultInitial();
      
      ss.a0 = ss0.a0;
      ss.P0 = ss0.P0;
      
      ss = ss.generateThetaMap();
      ss.Z = ss0.Z; ss.H = ss0.H; 
      ss.T = ss0.T; ss.Q = ss0.Q;
      
      % Test
      tic;
      [~, analytic] = ss.gradient(testCase.bbk.y);
      time_a = toc;
      tic;
      numeric = testCase.numericGradient(ss, testCase.bbk.y);
      time_n = toc;

%       testCase.printNice(ss, analytic, numeric);
      fprintf(['\nModel: %d series, %d states, t = %d (diagonal H)\n' ...
        'Analytic gradient took %3.2f%% of the time as the numeric version.\n'],...
        nSeries, rnfacs, length(testCase.bbk.y), 100*(time_a/time_n));
      testCase.verifyEqual(analytic, numeric, 'RelTol', 0.02);
    end
  end
end
