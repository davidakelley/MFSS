% Test univariate mex filter with an AR model.
% Assumes that the Matlab version of the univariate filter/smoother are
% correct.

% David Kelley, 2016

classdef estimate_test < matlab.unittest.TestCase
  
  properties
    data = struct;
    tol_DK = 1e-2;    % Test v. Drubin-Koopman
    tol_grad = 1e-5;   % Tets against gradient version
    bbk
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      % Factor model data
      [bbk_data, ~, dims] = loadFactorModel();
      % Subset the model to make it more managable:
      dims.nSeries = 92;
      y = bbk_data.indicators(:, 1:dims.nSeries)';
      y(:, any(isnan(y), 1)) = [];
      testCase.bbk = struct('data', bbk_data, 'dims', dims, 'y', y);
      
      % Load data
      testDir = [subsref(strsplit(pwd, 'StateSpace'), ...
        struct('type', '{}', 'subs', {{1}})) 'StateSpace\test\data'];
      dataStr = fileread(fullfile(testDir, 'Nile.dat'));
      lineBreaks = strfind(dataStr, sprintf('\n'));
      dataStr(1:lineBreaks(1)) = [];
      testCase.data.nile = sscanf(dataStr, '%d');

      addpath('C:\Users\g1dak02\Documents\MATLAB\StateSpace');
    end
  end
  
  methods (Test)
    
    function testNile(testCase)
      Z = 1;
      d = 0;
      H = nan;
      T = 1;
      c = 0;
      R = 1;
      Q = nan;
      
      ss = StateSpace(Z, d, H, T, c, R, Q, []);
      ss0 = ss;
      ss0.H = 1000;
      ss0.Q = 1000;
      ss.verbose = false;
      
      ssE = ss.estimate(testCase.data.nile', ss0);
      
      % Using values from Dubrin & Koopman (2012), p. 37
      testCase.verifyEqual(ssE.H, 15099, 'RelTol', testCase.tol_DK);
      testCase.verifyEqual(ssE.Q, 1469.1, 'RelTol', testCase.tol_DK);
    end
    
    function testNileGradient(testCase)
      Z = 1;
      d = 0;
      H = nan;
      T = 1;
      c = 0;
      R = 1;
      Q = nan;
      
      ss = StateSpace(Z, d, H, T, c, R, Q, []);
      ss0 = ss;
      ss0.H = 1000;
      ss0.Q = 1000;
      
      ss.useGrad = true;
      ss.verbose = false;
      ssE = ss.estimate(testCase.data.nile', ss0);
      ss.useGrad = false;
      ssE_ng = ss.estimate(testCase.data.nile', ss0);
      
      % Using values from Dubrin & Koopman (2012), p. 37
      testCase.verifyEqual(ssE.H, ssE_ng.H, 'RelTol', testCase.tol_grad);
      testCase.verifyEqual(ssE.Q, ssE_ng.Q, 'RelTol',  testCase.tol_grad);
    end
    
    function testMatlab(testCase)
      % Test against Matlab's native implementation of state space models
      Z = 1;
      d = 0;
      H = nan;
      T = 1;
      c = 0;
      R = 1;
      Q = nan;
      
      ss = StateSpace(Z, d, H, T, c, R, Q, []);
      ss0 = ss;
      ss0.H = 1000;
      ss0.Q = 1000;
      
      ss.useGrad = true;
      ss.verbose = false;
      ssE = ss.estimate(testCase.data.nile', ss0);
      
      A = 1; B = nan; C = 1; D = nan;
      mdl = ssm(A, B, C, D);
      estmdl = estimate(mdl, testCase.data.nile, [1000; 1000]);

      testCase.verifyEqual(ssE.H, estmdl.D^2, 'RelTol', testCase.tol_DK);
      testCase.verifyEqual(ssE.Q, estmdl.B^2, 'RelTol',  testCase.tol_DK);
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
      
      % Test
      ssE = ss.estimate(testCase.bbk.y, ss0);
      
    end
  end
end