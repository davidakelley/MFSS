% Test univariate mex filter with an AR model.
% Assumes that the Matlab version of the univariate filter/smoother are
% correct.

% David Kelley, 2016

classdef estimate_test < matlab.unittest.TestCase
  
  properties
    data = struct;
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      % Load data
      baseDir =  [subsref(strsplit(mfilename('fullpath'), 'MFSS'), ...
        struct('type', '{}', 'subs', {{1}})) 'MFSS'];
      addpath(baseDir);
      addpath(fullfile(baseDir, 'examples'));

      data_dk = load(fullfile(baseDir, 'examples', 'data', 'dk.mat'));
      testCase.data.nile = data_dk.nile;
    end
  end
  
  methods(TestClassTeardown)
    function closeFigs(testCase) %#ok<MANU>
      close all;
    end
  end
  
  methods (Test)
    function testNile(testCase)
      nile = testCase.data.nile';
      
      Z = 1;
      d = 0;
      H = nan;
      T = 1;
      c = 0;
      R = 1;
      Q = nan;
      
      ss = StateSpaceEstimation(Z, d, H, T, c, R, Q);
      
      H0 = 1000;
      P0 = 1000;
      ss0 = StateSpace(Z, d, H0, T, c, R, P0);
      
      ss.useAnalyticGrad = false;
      ssE = ss.estimate(nile, ss0);
      
      % Using values from Dubrin & Koopman (2012), p. 37
      testCase.verifyEqual(ssE.H, 15099, 'RelTol', 1e-2);
      testCase.verifyEqual(ssE.Q, 1469.1, 'RelTol', 1e-2);
    end
    
    function testNileKappa(testCase)
      nile = testCase.data.nile';
      
      Z = 1;
      d = 0;
      H = nan;
      T = 1;
      c = 0;
      R = 1;
      Q = nan;
      
      ss = StateSpaceEstimation(Z, d, H, T, c, R, Q);
      ss.a0 = 0;
      ss.P0 = 1e6;
      
      H0 = 1000;
      P0 = 1000;
      ss0 = StateSpace(Z, d, H0, T, c, R, P0);
      ss0.a0 = 0;
      ss0.P0 = 1e6;
      
      ssE = ss.estimate(nile, ss0);
      
      % Using values from Dubrin & Koopman (2012), p. 37
      testCase.verifyEqual(ssE.H, 15099, 'RelTol', 1e-2);
      testCase.verifyEqual(ssE.Q, 1469.1, 'RelTol', 1e-2);
    end
    
    function testNileGradient(testCase)
      nile = testCase.data.nile';

      Z = 1;
      d = 0;
      H = nan;
      T = 1;
      c = 0;
      R = 1;
      Q = nan;
      
      ss = StateSpaceEstimation(Z, d, H, T, c, R, Q);
      
      H0 = 1000;
      Q0 = 1000;
      ss0 = StateSpace(Z, d, H0, T, c, R, Q0);

      ss.useAnalyticGrad = false;
      ssE_ng = ss.estimate(nile, ss0);
      
      ss.useAnalyticGrad = true;
      ssE = ss.estimate(nile, ss0);

      testCase.verifyEqual(ssE.H, ssE_ng.H, 'RelTol', 5e-4);
      testCase.verifyEqual(ssE.Q, ssE_ng.Q, 'RelTol',  5e-4);
    end
    
    function testMatlab(testCase)
      % Test against Matlab's native implementation of state space models
      nile = testCase.data.nile';
      
      Z = 1;
      d = 0;
      H = nan;
      T = 1;
      c = 0;
      R = 1;
      Q = nan;
      
      ss = StateSpaceEstimation(Z, d, H, T, c, R, Q);
      
      H0 = 1000;
      Q0 = 1000;
      ss0 = StateSpace(Z, d, H0, T, c, R, Q0);
      
      ss.useAnalyticGrad = false;
      ssE = ss.estimate(nile, ss0);
      
      A = 1; B = nan; C = 1; D = nan;
      mdl = ssm(A, B, C, D);
      estmdl = estimate(mdl, nile', [1000; 1000]);

      testCase.verifyEqual(ssE.H, estmdl.D^2, 'RelTol', 1e-2);
      testCase.verifyEqual(ssE.Q, estmdl.B^2, 'RelTol',  1e-2);
    end
    
    function testGeneratedSmallGradientZero(testCase)
      assumeFail(testCase); % Filter by assumption

      p = 2; m = 1; timeDim = 500;
      ssTrue = generateARmodel(p, m-1, false);
      y = generateData(ssTrue, timeDim);
      
      % Estimated system
      Z = [[1; nan(p-1, 1)] zeros(p, m-1)];
      d = zeros(p, 1);
      H = nan(p, p);
      
      T = [nan(1, m); [eye(m-1) zeros(m-1, 1)]];
      c = zeros(m, 1);
      R = zeros(m, 1); R(1, 1) = 1;
      Q = nan;
      
      ssE = StateSpaceEstimation(Z, d, H, T, c, R, Q);
      
      % Initialization
      pcaWeight = pca(y');
      Z0 = ssE.Z;
      Z0(:,1) = pcaWeight(:, 1);
      res = pcares(y', 1);      
      H0 = cov(res);
      T0 = ssE.T;
      T0(isnan(T0)) = 0.5./m;
      Q0 = 1;
      ss0 = StateSpace(Z0, d, H0, T0, c, R, Q0);

      [~, ~, grad] = ssE.estimate(y, ss0);
      testCase.verifyLessThanOrEqual(abs(grad), 5e-4);
    end
    
    function testBounds(testCase)
      p = 2; m = 1; timeDim = 500;
      ssTrue = generateARmodel(p, m-1, false);
      y = generateData(ssTrue, timeDim);
      
      % Estimated system
      Z = [[1; nan(p-1, 1)] zeros(p, m-1)];
      d = zeros(p, 1);
      H = nan(p, p);
      
      T = [nan(1, m); [eye(m-1) zeros(m-1, 1)]];
      c = zeros(m, 1);
      R = zeros(m, 1); R(1, 1) = 1;
      Q = nan;
      
      % Bounds: constrain 0 < T < 1
      Zlb = Z; Zlb(:) = -Inf;
      dlb = d; dlb(:) = -Inf;
      Hlb = H; Hlb(:) = 0;
      Tlb = T; Tlb(:) = -1;
      clb = c; clb(:) = -Inf;
      Rlb = R; Rlb(:) = -Inf;
      Qlb = Q; Qlb(:) = 0;
      ssLB = StateSpace(Zlb, dlb, Hlb, Tlb, clb, Rlb, Qlb);
      
      Zub = Z; Zub(:) = Inf;
      dub = d; dub(:) = Inf;
      Hub = H; Hub(:) = Inf;
      Tub = T; Tub(:) = 1;
      cub = c; cub(:) = Inf;
      Rub = R; Rub(:) = Inf;
      Qub = Q; Qub(:) = Inf;
      ssUB = StateSpace(Zub, dub, Hub, Tub, cub, Rub, Qub);

      ss = StateSpaceEstimation(Z, d, H, T, c, R, Q, ...
        'LowerBound', ssLB, 'UpperBound', ssUB);
     
      ss0 = ssTrue;
      ss0.T = 0.2;
      
      [ssE, ~, ~] = ss.estimate(y, ss0);
      testCase.verifyLessThanOrEqual(ssE.T, Tub);
      testCase.verifyGreaterThanOrEqual(ssE.T, Tlb);      
    end
  end
end