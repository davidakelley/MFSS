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
    deai
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
      
      baseDir =  [subsref(strsplit(mfilename('fullpath'), 'StateSpace'), ...
        struct('type', '{}', 'subs', {{1}})) 'StateSpace'];
      testCase.deai = load(fullfile(baseDir, 'test', 'data', 'deai.mat'));


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
      
      ss = StateSpaceEstimation(Z, d, H, T, c, R, Q, []);
      ss.useGrad = false;
      
      H0 = 1000;
      Q0 = 1000;
      ss0 = StateSpace(Z, d, H0, T, c, R, Q0, []);
      
      ssE = ss.estimate(testCase.data.nile', ss0);
      
      % Using values from Dubrin & Koopman (2012), p. 37
      testCase.verifyEqual(ssE.H, 15099, 'RelTol', testCase.tol_DK);
      testCase.verifyEqual(ssE.Q, 1469.1, 'RelTol', testCase.tol_DK);
      
      [~, ssE_grad] = ssE.gradient(testCase.data.nile', ss.ThetaMapping);
      testCase.verifyLessThan(abs(ssE_grad), ss.tol);
    end
    
    function testNileGradient(testCase)
      Z = 1;
      d = 0;
      H = nan;
      T = 1;
      c = 0;
      R = 1;
      Q = nan;
      
      ss = StateSpaceEstimation(Z, d, H, T, c, R, Q, []);
      
      H0 = 1000;
      Q0 = 1000;
      ss0 = StateSpace(Z, d, H0, T, c, R, Q0, []);

      ss.useGrad = false;
      ssE_ng = ss.estimate(testCase.data.nile', ss0);
      
      ss.useGrad = true;
      ssE = ss.estimate(testCase.data.nile', ss0);

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
      
      ss = StateSpaceEstimation(Z, d, H, T, c, R, Q, []);
      
      H0 = 1000;
      Q0 = 1000;
      ss0 = StateSpace(Z, d, H0, T, c, R, Q0, []);
      
      ss.useGrad = false;
      ssE = ss.estimate(testCase.data.nile', ss0);
      
      A = 1; B = nan; C = 1; D = nan;
      mdl = ssm(A, B, C, D);
      estmdl = estimate(mdl, testCase.data.nile, [1000; 1000]);

      testCase.verifyEqual(ssE.H, estmdl.D^2, 'RelTol', testCase.tol_DK);
      testCase.verifyEqual(ssE.Q, estmdl.B^2, 'RelTol',  testCase.tol_DK);
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
      
      ss = StateSpaceEstimation(Z, d, H, T, c, R, Q, []);
      
      % Initialization
      pcaWeight = pca(y');
      Z0 = ss.Z;
      Z0(:,1) = pcaWeight(:, 1);
      res = pcares(y', 1);      
      H0 = cov(res);
      T0 = ss.T;
      T0(isnan(T0)) = 0.5./m;
      Q0 = 1;
      ss0 = StateSpace(Z0, d, H0, T0, c, R, Q0, []);

      [~, ~, grad] = ss.estimate(y, ss0);
      testCase.verifyLessThan(abs(grad), ss.tol);
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
      ssLB = ss;
      ssLB.Z(:) = nan;
      ssLB.d(:) = nan;
      ssLB.H(:) = nan;
      ssLB.T(:) = 0;
      ssLB.c(:) = nan;
      ssLB.R(:) = nan;
      ssLB.Q(:) = nan;
      ssUB = ssLB;
      
      ssUB.T = 1;
      ssUB.H(:) = Inf;

      ss = StateSpaceEstimation(Z, d, H, T, c, R, Q, [], ...
        'LowerBound', ssLB, 'UpperBound', ssUB);

      [ssE, ~, grad] = ss.estimate(y, ssTrue, ...
        'LowerBound', ssLB, 'UpperBound', ssUB);
      testCase.verifyLessThan(abs(grad), ssE.tol);
      
    end
%     
%     function testFactorModel(testCase)
%       % Set up state
%       rnfacs = 2;
%       nSeries = 4; %testCase.bbk.dims.nSeries;
%       nlags = testCase.bbk.dims.nlags;
%       
%       Z = [nan(nSeries, rnfacs) zeros(nSeries, rnfacs * (nlags-1))];
%       d = zeros(nSeries, 1);
%       H = eye(nSeries);
%       H(H == 1) = nan;
%       
%       T = [nan(rnfacs, rnfacs * nlags);
%         eye(rnfacs * (nlags-1)) zeros(rnfacs)];
%       c = zeros(rnfacs * nlags, 1);
%       R = [eye(rnfacs); zeros(rnfacs * (nlags-1), rnfacs)];
%       Q = nan(rnfacs);
%       
%       ss = StateSpace(Z, d, H, T, c, R, Q, []);
%       ss0 = ss;
%       
%       % Initial values
%       [ss0.Z(:, 1:rnfacs), f0] = pca(testCase.bbk.y(1:nSeries, :)', 'NumComponents', rnfacs);
%       f0(any(isnan(f0), 2), :) = [];
%       
%       ss0.H = diag(var(testCase.bbk.y(1:nSeries, :)' - f0 * ss0.Z(:, 1:rnfacs)'));
%       
%       y_var = f0(nlags+1:end, :);
%       assert(nlags == 2);
%       x = [f0(2:end-1, :) f0(1:end-2, :)];
%       
%       yTx = y_var' * x;
%       xTx = x' * x;
%       yTy = y_var' * y_var;
%       
%       ss0.T(1:rnfacs, :) = yTx / xTx;
%       ss0.Q = (yTy - yTx / xTx * yTx') ./ size(testCase.bbk.y(1:nSeries, :), 1);
%       
%       % Test
%       ssE = ss.estimate(testCase.bbk.y(1:nSeries, :), ss0);
%       
%     end
%     
%     function testDetroit(testCase)
%       ss0 = StateSpace(testCase.deai.Z, testCase.deai.d, testCase.deai.H, ...
%         testCase.deai.T, testCase.deai.c, testCase.deai.R, testCase.deai.Q, ...
%         testCase.deai.Harvey);
%       
%       estZ = testCase.deai.Z;
%       estZ(:,1) = nan;
%       estT = testCase.deai.T;
%       estT(estT~=0 & estT~=1) = nan;
%       estQ = testCase.deai.Q;
%       estQ(estQ~=0 & estQ~=1) = nan;
%       ss = StateSpace(estZ, testCase.deai.d, testCase.deai.H, ...
%         estT, testCase.deai.c, testCase.deai.R, estQ, ...
%         testCase.deai.Harvey);
%       
%       ssE = ss.estimate(testCase.deai.Y, ss0);
%       
%     end
  end
end