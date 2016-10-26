% Tests for StateSpace.gradient
%
% Run with
%{
  testOut = runtests('test/Gradient_test.m');
%}

% David Kelley, 2016

classdef Gradient_factor_model_test < matlab.unittest.TestCase
  
  properties
    y
    ss
    logl
    gradient
    delta = 1e-5;
    analyticTime
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      
      ssDir = pwd;
      cd('C:\Users\g1dak02\Documents\MATLAB\bbk');
      [opts, dims, paths] = estimationOptions();
      opts.targets = {'GDPH'};
      [opts, dims] = getTargetSpec(opts, dims);
      [data, opts, dims] = loadData(opts, dims, paths);
      cd(ssDir);
      
      resetPath;
      addpath('C:\Users\g1dak02\Documents\MATLAB\StateSpace');

      dims.nSeries = 2;
      
      testCase.y = data.indicators(:, 1:dims.nSeries)';
      testCase.y(:, any(isnan(testCase.y), 1)) = [];
      
      %% Set up state
      dims.rnfacs = 2;
      Z = [nan(dims.nSeries, dims.rnfacs) zeros(dims.nSeries, dims.rnfacs * (dims.nlags-1))];
      d = zeros(dims.nSeries, 1);
      H = eye(dims.nSeries);
      H(H == 1) = nan;
      
      T = [nan(dims.rnfacs, dims.rnfacs * dims.nlags);
        eye(dims.rnfacs * (dims.nlags-1)) zeros(dims.rnfacs)];
      c = zeros(dims.rnfacs * dims.nlags, 1);
      R = [eye(dims.rnfacs); zeros(dims.rnfacs * (dims.nlags-1), dims.rnfacs)];
      Q = nan(dims.rnfacs);
      
      testCase.ss = StateSpace(Z, d, H, T, c, R, Q, []);
      
      %% Initial values
      [testCase.ss.Z(:, 1:dims.rnfacs), f0] = pca(testCase.y', 'NumComponents', dims.rnfacs);
      f0(any(isnan(f0), 2), :) = [];
      
      testCase.ss.H = diag(var(testCase.y' - f0 * testCase.ss.Z(:, 1:dims.rnfacs)'));
      
      y_var = f0(dims.nlags+1:end, :);
      assert(dims.nlags == 2);
      x = [f0(2:end-1, :) f0(1:end-2, :)];
      
      yTx = y_var' * x;
      xTx = x' * x;
      yTy = y_var' * y_var;
      
      testCase.ss.T(1:dims.rnfacs, :) = yTx / xTx;
      testCase.ss.Q = (yTy - yTx / xTx * yTx') ./ size(testCase.y, 1);
      
      %% Set up 
      testCase.ss = testCase.ss.checkSample(testCase.y);
      testCase.ss = testCase.ss.setDefaultInitial();
      testCase.ss = testCase.ss.generateThetaMap();
      
      tic;
      [testCase.logl, testCase.gradient] = testCase.ss.gradient(testCase.y);
      testCase.analyticTime = toc;
    end
  end
  
  methods (Test)
    function testZ(testCase)
      theta = testCase.ss.getParamVec();
      grad = nan(size(theta));
      
      tic;
      for iT = 1:size(theta, 1)
        iTheta = theta;
        iTheta(iT) = iTheta(iT) + testCase.delta;
        
        [ssTest, a0_theta, P0_theta] = testCase.ss.theta2system(iTheta);
        
        [~, loglZ1] = ssTest.filter(testCase.y, a0_theta, P0_theta);
        grad(iT) = (loglZ1 - testCase.logl) ./ testCase.delta;
      end
      numericalTime = toc;
      
      disp('Percentage errors:');
      disp((grad - testCase.gradient) ./ testCase.gradient * 100);
      fprintf('Analytic gardient took %3.2f%% of the time.\n', ...
        100 * (testCase.analyticTime / numericalTime));
      testCase.verifyEqual(testCase.gradient, grad, 'RelTol', 0.05);
    end
  end
end
