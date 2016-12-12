% Test univariate mex filter with an AR model.
% Assumes that the Matlab version of the univariate filter/smoother are
% correct.

% David Kelley, 2016

classdef estimate_test < matlab.unittest.TestCase
  
  properties
    data = struct;
    bbk
    deai
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      % Load data
      baseDir =  [subsref(strsplit(mfilename('fullpath'), 'MFSS'), ...
        struct('type', '{}', 'subs', {{1}})) 'MFSS'];
      addpath(baseDir);
      
      testCase.bbk = load(fullfile(baseDir, 'examples', 'data', 'bbk_data.mat'));
      y = testCase.bbk.data.indicators';
      y(:, any(isnan(y), 1)) = [];
      testCase.bbk.y = y;
      
      data_dk = load(fullfile(baseDir, 'examples', 'data', 'dk.mat'));
      testCase.data.nile = data_dk.nile;
      
      testCase.deai = load(fullfile(baseDir, 'examples', 'data', 'deai.mat'));
    end
  end
  
  methods(TestClassTeardown)
    function closeFigs(testCase) %#ok<MANU>
      close all;
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
      
      ss = StateSpaceEstimation(Z, d, H, T, c, R, Q);
      
      H0 = 1000;
      Q0 = 1000;
      ss0 = StateSpace(Z, d, H0, T, c, R, Q0);
      
      nile = testCase.data.nile';
      ssE = ss.estimate(nile, ss0);
      
      % Using values from Dubrin & Koopman (2012), p. 37
      testCase.verifyEqual(ssE.H, 15099, 'RelTol', 1e-2);
      testCase.verifyEqual(ssE.Q, 1469.1, 'RelTol', 1e-2);
      
      [~, ssE_grad] = ssE.gradient(nile, ss.ThetaMapping);
      testCase.verifyLessThan(abs(ssE_grad), 5e-4);
    end
    
    function testNileGradient(testCase)
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

      ss.useGrad = false;
      ssE_ng = ss.estimate(testCase.data.nile', ss0);
      
      ss.useGrad = true;
      ssE = ss.estimate(testCase.data.nile', ss0);

      % Using values from Dubrin & Koopman (2012), p. 37
      testCase.verifyEqual(ssE.H, ssE_ng.H, 'RelTol', 5e-4);
      testCase.verifyEqual(ssE.Q, ssE_ng.Q, 'RelTol',  5e-4);
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
      
      ss = StateSpaceEstimation(Z, d, H, T, c, R, Q);
      
      H0 = 1000;
      Q0 = 1000;
      ss0 = StateSpace(Z, d, H0, T, c, R, Q0);
      
      ss.useGrad = false;
      ssE = ss.estimate(testCase.data.nile', ss0);
      
      A = 1; B = nan; C = 1; D = nan;
      mdl = ssm(A, B, C, D);
      estmdl = estimate(mdl, testCase.data.nile, [1000; 1000]);

      testCase.verifyEqual(ssE.H, estmdl.D^2, 'RelTol', 1e-2);
      testCase.verifyEqual(ssE.Q, estmdl.B^2, 'RelTol',  1e-2);
    end
    
    function testGeneratedSmall(testCase)
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
      
      ss = StateSpaceEstimation(Z, d, H, T, c, R, Q);
      
      % Initialization
      pcaWeight = pca(y');
      Z0 = ss.Z;
      Z0(:,1) = pcaWeight(:, 1);
      res = pcares(y', 1);      
      H0 = cov(res);
      T0 = ss.T;
      T0(isnan(T0)) = 0.5./m;
      Q0 = 1;
      ss0 = StateSpace(Z0, d, H0, T0, c, R, Q0);

      [ssE, ~, grad] = ss.estimate(y, ss0);
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
      Tlb = T; Tlb(:) = 0.1;
      clb = c; clb(:) = -Inf;
      Rlb = R; Rlb(:) = -Inf;
      Qlb = Q; Qlb(:) = 0;
      ssLB = StateSpace(Zlb, dlb, Hlb, Tlb, clb, Rlb, Qlb);
      
      Zub = Z; Zub(:) = Inf;
      dub = d; dub(:) = Inf;
      Hub = H; Hub(:) = Inf;
      Tub = T; Tub(:) = 0.3;
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
    
    %{
    function testFactorModel(testCase)
      % Set up state
      rnfacs = 2;
      nSeries = 4; %testCase.bbk.dims.nSeries;
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
      [ss0.Z(:, 1:rnfacs), f0] = pca(testCase.bbk.y(1:nSeries, :)', 'NumComponents', rnfacs);
      f0(any(isnan(f0), 2), :) = [];
      
      ss0.H = diag(var(testCase.bbk.y(1:nSeries, :)' - f0 * ss0.Z(:, 1:rnfacs)'));
      
      y_var = f0(nlags+1:end, :);
      assert(nlags == 2);
      x = [f0(2:end-1, :) f0(1:end-2, :)];
      
      yTx = y_var' * x;
      xTx = x' * x;
      yTy = y_var' * y_var;
      
      ss0.T(1:rnfacs, :) = yTx / xTx;
      ss0.Q = (yTy - yTx / xTx * yTx') ./ size(testCase.bbk.y(1:nSeries, :), 1);
      
      % Test
      ssE = ss.estimate(testCase.bbk.y(1:nSeries, :), ss0);
      
    end
    
    function testDetroit(testCase)
      ss0 = StateSpace(testCase.deai.Z, testCase.deai.d, testCase.deai.H, ...
        testCase.deai.T, testCase.deai.c, testCase.deai.R, testCase.deai.Q, ...
        testCase.deai.Harvey);
      
      estZ = testCase.deai.Z;
      estZ(:,1) = nan;
      estT = testCase.deai.T;
      estT(estT~=0 & estT~=1) = nan;
      estQ = testCase.deai.Q;
      estQ(estQ~=0 & estQ~=1) = nan;
      ss = StateSpace(estZ, testCase.deai.d, testCase.deai.H, ...
        estT, testCase.deai.c, testCase.deai.R, estQ, ...
        testCase.deai.Harvey);
      
      ssE = ss.estimate(testCase.deai.Y, ss0);
      
    end
    %}
  end
end