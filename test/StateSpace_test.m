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
    
    function testVJ_VAR2_missing_long(testCase)
      p = 3; 
      lags = 2;
      nT = 150;
      
      [y, ss] = mfvar_test.generateVAR(p, lags, nT);
      y(1:100, 1) = nan;
      
      [~, sOut, fOut] = ss.smooth(y);      
      ss = ss.setDefaultInitial();
      [ssPrep, yPrep] = ss.prepareFilter(y, [], []);
      sOut.N = cat(3, sOut.N, zeros(size(sOut.N, 1)));
      [ssV, ssJ] = ssPrep.getErrorVariances(yPrep, fOut, sOut);
      
      [P, N, V, J] = StateSpace_test.kalmanMultiVariances(ss, lags, y);
      
      testCase.verifyEqual(fOut.P, P, 'AbsTol', 5e-10);
      testCase.verifyEqual(sOut.N, N, 'AbsTol', 5e-10);
      testCase.verifyEqual(sOut.V, ssV, 'AbsTol', 5e-10);
      testCase.verifyEqual(sOut.V, V, 'AbsTol', 5e-10);
      testCase.verifyEqual(ssJ, J, 'AbsTol', 5e-10);
    end
    
    function testVJ_VAR2_missing_accum_long(testCase)
      p = 3; 
      lags = 2;
      seed = 7;
      
      [y, ss] = mfvar_test.generateVAR(p, lags, 51, seed);
      aggY = y;
      aggY(:, 2) = Accumulator_test.aggregateY(y(:, 2), 3, 'avg');
      
      aggY(1:45,3) = nan;
      
      [~, sOut, fOut] = ss.smooth(aggY);      
      ss = ss.setDefaultInitial();
      [ssPrep, yPrep] = ss.prepareFilter(aggY, [], []);
      sOut.N = cat(3, sOut.N, zeros(size(sOut.N, 1)));
      [ssV, ssJ] = ssPrep.getErrorVariances(yPrep, fOut, sOut);
      
      [P, N, V, J] = StateSpace_test.kalmanMultiVariances(ss, lags, aggY);
      
      testCase.verifyEqual(fOut.P, P, 'AbsTol', 5e-10);
      testCase.verifyEqual(sOut.N, N, 'AbsTol', 5e-10);
      testCase.verifyEqual(sOut.V, ssV, 'AbsTol', 5e-10);
      testCase.verifyEqual(sOut.V, V, 'AbsTol', 5e-10);
      testCase.verifyEqual(ssJ, J, 'AbsTol', 5e-10);
    end
    
  end
  
  methods(Static)
    function [P, N, V, J, L] = kalmanMultiVariances(ss, nLags, y)
      % Multivariate Kalman filter variance matrixes
      pl = ss.p * nLags;
      nT = size(y,1);
      
      P = zeros(pl, pl, nT+1);
      F = zeros(ss.p, ss.p, nT);
      L = zeros(pl, pl, nT);
      
      N = zeros(pl, pl, nT+1);
      V = zeros(pl, pl, nT);
      J = zeros(pl, pl, nT);
      
      wInd = ~isnan(y(1,:));
      W = eye(ss.p);
      W = W(wInd,:);

      P(:,:,1) = ss.T * ss.P0 * ss.T' + ss.R * ss.Q * ss.R';
      
      WZ = W * ss.Z;
      
      F(wInd,wInd,1) = WZ * P(:,:,1) * WZ' + W * ss.H * W';
      N(:,:,1) = WZ' / F(wInd,wInd,1) * WZ;
      
      V(:,:,1) = P(:,:,1) - P(:,:,1) * N(:,:,1) * P(:,:,1);
      K = ss.T * P(:,:,1) * WZ' / F(wInd,wInd,1);
      L(:,:,1) = ss.T - K * WZ;
      J(:,:,1) = P(:,:,1) * L(:,:,1)';
      
      P(:,:,2) = ss.T * P(:,:,1) * L(:,:,1)' + ss.R * ss.Q * ss.R';
      
      for iT = 2:nT
        wInd = ~isnan(y(iT,:));
        W = eye(ss.p);
        W = W(wInd,:);

        WZ = W * ss.Z;

        F(wInd,wInd,iT) = WZ * P(:,:,iT) * WZ' + W * ss.H * W';
        
        K = ss.T * P(:,:,iT) * WZ' / F(wInd,wInd,iT);
        L(:,:,iT) = ss.T - K * WZ;
        
        P(:,:,iT+1) = ss.T * P(:,:,iT) * L(:,:,iT)' + ss.R * ss.Q * ss.R';
      end
      
      % N(:,:,T+1) is zeros.
      for iT = nT:-1:1
        wInd = ~isnan(y(iT,:));
        W = eye(ss.p);
        W = W(wInd,:);

        WZ = W * ss.Z;
        
        N(:,:,iT) = WZ' / F(wInd,wInd,iT) * WZ + L(:,:,iT)' * N(:,:,iT+1) * L(:,:,iT);
        V(:,:,iT) = P(:,:,iT) - P(:,:,iT) * N(:,:,iT) * P(:,:,iT);
        J(:,:,iT) = P(:,:,iT) * L(:,:,iT)' * (eye(pl) - N(:,:,iT+1) * P(:,:,iT+1));
      end
      
    end
  end
  
end
