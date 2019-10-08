% Test non-Kalman filter methods of StateSpace

% David Kelley, 2019

classdef StateSpace_test < matlab.unittest.TestCase
  
  properties
    data = struct;
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      % Load data
      baseDir = fileparts(fileparts(mfilename('fullpath')));
      addpath(baseDir);
      addpath(fullfile(baseDir, 'examples'));

      data_load = load(fullfile(baseDir, 'examples', 'durbin_koopman.mat'));
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
      J = P1 * L1'; 
      
      testCase.verifyEqual(ssV, V1);
      testCase.verifyEqual(ssJ, J);
    end
    
    function testVJ_VAR2(testCase)
      p = 2; 
      lags = 2;
      
      [y, ss] = mfvar_test.generateVAR(p, lags, 1);
      
      ss = ss.setDefaultInitial;
      
      [~, sOut, fOut] = ss.smooth(y);      
      ss = ss.setDefaultInitial();
      [ss, yPrep] = ss.prepareFilter(y, [], []);
      sOut.N = cat(3, sOut.N, zeros(size(sOut.N, 1)));
      [ssV, ssJ] = ss.getErrorVariances(yPrep, fOut, sOut);
      
      % Compute in multivariate for simplicity
      P1 = ss.T * ss.P0 * ss.T' + ss.R * ss.Q * ss.R';
      F1 = ss.Z * P1 * ss.Z' + ss.H;
      N1 = ss.Z' / F1 * ss.Z;
      V1 = P1 - P1 * N1 * P1;
      K1 = ss.T * P1 * ss.Z' / F1;
      L1 = ss.T - K1 * ss.Z;
      J = P1 * L1';

      testCase.verifyEqual(ssV, V1, 'AbsTol', 5e-10);
      testCase.verifyEqual(ssJ, J, 'AbsTol', 5e-10);
    end
  end
  
end
