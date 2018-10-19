% Test univariate mex filter with an AR model.
% Assumes that the Matlab version of the univariate filter/smoother are
% correct.

% David Kelley, 2016

classdef mfvar_test < matlab.unittest.TestCase
  
  properties
    data = struct;
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      % Load data
      baseDir = fileparts(fileparts(mfilename('fullpath')));
      addpath(baseDir);
      addpath(fullfile(baseDir, 'examples'));

      data_load = load(fullfile(baseDir, 'examples', 'data', 'dk.mat'));
      testCase.data.nile = data_load.nile;
    end
  end
  
  methods(TestClassTeardown)
    function closeFigs(testCase) %#ok<MANU>
      close all force;
    end
  end
  
  methods (Test)
    %% Test that the V and J computation from StateSpace is accurate
    function testVJ_noT(testCase)
      ss = StateSpace([1; 1; 1], eye(3), 0, 1);
      y = generateData(ss, 100);
      
      [~, sOut, fOut] = ss.smooth(y);      
      ss = ss.setDefaultInitial();
      ss = ss.prepareFilter(y, [], []);
      sOut.N = cat(3, sOut.N, zeros(size(sOut.N, 1)));
      [V, J] = ss.getErrorVariances(y, fOut, sOut);
      
      testCase.verifyEqual(V, 0.25*ones(1,1,100));
      testCase.verifyEqual(J, zeros(1,1,100));
    end
    
    function testVJ_AR1(testCase)
      ss = StateSpace(1, 1, 0.5, 1);
      y = generateData(ss, 1);
      ss.a0 = 0;
      ss.P0 = 1;
      
      [~, sOut, fOut] = ss.smooth(y);      
      ss = ss.setDefaultInitial();
      ss = ss.prepareFilter(y, [], []);
      sOut.N = cat(3, sOut.N, zeros(size(sOut.N, 1)));
      [ssV, ssJ] = ss.getErrorVariances(y, fOut, sOut);
      
      % Compute in multivariate for simplicity
      P1 = ss.T^2 * ss.P0 + ss.Q;
      F1 = P1 + ss.Q;
      N1 = 1./F1;
      V1 = P1 - P1 * N1 * P1;
      K1 = ss.T * P1 * ss.Z' / F1;
      L1 = ss.T - K1 * ss.Z;
      % P2 = (ss.T^2 * P1 + ss.Q);
      % N2 = 1./(P2 + ss.Q);
      J = P1 * L1'; %* (eye(ss.m) - N2 * P2); % j = t + 1
      
      testCase.verifyEqual(ssV, V1);
      testCase.verifyEqual(ssJ, J);
    end
    
    function testVJ_VAR2(testCase)
      p = 2; 
      lags = 2;
      
      phi2T = @(phi) [phi; eye(p*(lags-1)) zeros(p*(lags-1), p)];
      
      phiRaw = 0.2*randn(p, p*lags) + [.5*eye(p) zeros(p)];
      phi = phiRaw - ...
        [eye(p)*(max(abs(eig(phi2T(phiRaw))))-1) zeros(p, p*(lags-1))];
      const = randn(p,1);
      sigmaRaw = randn(p);
      sigma = eye(p) + 0.5 * (sigmaRaw + sigmaRaw');
      
      ss = StateSpace([eye(p) zeros(p,p*(lags-1))], zeros(p), ...
        phi2T(phi), sigma, 'c', [const; zeros(p*(lags-1),1)], ...
        'R', [eye(p); zeros(p*(lags-1),p)]);
      y = generateData(ss, 1);
      ss = ss.setDefaultInitial;
      
      [~, sOut, fOut] = ss.smooth(y);      
      ss = ss.setDefaultInitial();
      ss = ss.prepareFilter(y, [], []);
      sOut.N = cat(3, sOut.N, zeros(size(sOut.N, 1)));
      [ssV, ssJ] = ss.getErrorVariances(y, fOut, sOut);
      
      % Compute in multivariate for simplicity
      P1 = ss.T * ss.P0 * ss.T' + ss.R * ss.Q * ss.R';
      F1 = ss.Z * P1 * ss.Z' + ss.H;
      N1 = ss.Z' / F1 * ss.Z;
      V1 = P1 - P1 * N1 * P1;
      K1 = ss.T * P1 * ss.Z' / F1;
      L1 = ss.T - K1 * ss.Z;
      J = P1 * L1';

      testCase.verifyEqual(ssV, V1, 'AbsTol', 5e-14);
      testCase.verifyEqual(ssJ, J, 'AbsTol', 5e-14);
    end
    
    %% Integration tests of MFVAR
    function testEM_AR1_improve(testCase)
      % Test that the the EM always improved the likelihood
      nile = testCase.data.nile;
      varE = MFVAR(nile, 1);
      testCase.verifyWarningFree(@varE.estimate);
    end
    
    function testEM_AR1(testCase)
      % Test that the EM is working by comparing it to general ML estimation
      
      nile = testCase.data.nile;
      
      % Estimate MFVAR
      varE = MFVAR(nile, 1);
      varOpt = varE.estimate();
      [~, llEM] = varOpt.filter(nile);
      
      % Estimate general state space optimization
      ssE = StateSpaceEstimation(1, 0, nan, nan, 'c', nan);
      ssE.a0 = varOpt.a0;
      ssE.P0 = varOpt.P0;
      ssOpt = ssE.estimate(nile, varOpt);
      [~, llOpt] = ssOpt.filter(nile);
      
      testCase.verifyEqual(llEM, llOpt, 'AbsTol', 1e-2);
    end
    
  end
end