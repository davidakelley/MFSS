% Test thetamap  with an AR model.
% Assumes that the Matlab version of the theta mapping method are
% correct.

% David Kelley & Bill Kluender, 2016-2017

classdef ThetaMap_test < matlab.unittest.TestCase
  
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
      addpath(fullfile(baseDir, 'examples'));
      
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
    
    %% Tests converting between theta vectors and StateSpaces
    function thetaSystemThetaSimple(testCase)
      % Test that we can go from a theta to a system and back.
      Z = 1;
      d = 0;
      H = 3;
      T = 1;
      c = 0;
      R = 1;
      Q = 5;
      
      % Create ThetaMap
      ss = StateSpace(Z, d, H, T, c, R, Q);
      tm = ThetaMap.ThetaMapAll(ss);
      
      % Convert from theta -> ss -> thetaMap
      theta = [2 0 2.0986 1 0 1 2.6094]';
      ss = tm.theta2system(theta);
      thetaNew = tm.system2theta(ss);
      
      testCase.verifyEqual(theta, thetaNew, 'AbsTol', 1e-16);
    end
    
    function systemThetaSystemSimple(testCase)
      % Test that we can go from a StateSpace to a theta vector and back.
      Z = 1;
      d = 0;
      H = 3;
      T = 1;
      c = 0;
      R = 1;
      Q = 5;
      
      % Create ThetaMap
      ss = StateSpace(Z, d, H, T, c, R, Q);
      tm = ThetaMap.ThetaMapAll(ss);
      
      % Convert from StateSpace -> theta -> StateSpace
      theta = tm.system2theta(ss);
      ssNew = tm.theta2system(theta);
      
      testCase.verifyEqual(ss.Z, ssNew.Z, 'AbsTol', 1e-16);
      testCase.verifyEqual(ss.d, ssNew.d, 'AbsTol', 1e-16);
      testCase.verifyEqual(ss.H, ssNew.H, 'AbsTol', 1e-16);
      testCase.verifyEqual(ss.T, ssNew.T, 'AbsTol', 1e-16);
      testCase.verifyEqual(ss.c, ssNew.c, 'AbsTol', 1e-16);
      testCase.verifyEqual(ss.R, ssNew.R, 'AbsTol', 1e-16);
      testCase.verifyEqual(ss.Q, ssNew.Q, 'AbsTol', 1e-16);
      testCase.verifyEqual(ss.usingDefaulta0, ssNew.usingDefaulta0);
      testCase.verifyEqual(ss.usingDefaultP0, ssNew.usingDefaultP0);
    end
    
    function thetaSystemTheta_a0P0(testCase)
      % Theta -> StateSpace -> Theta with explicit initial values
      Z = 1;
      d = 0;
      H = 3;
      T = 1;
      c = 0;
      R = 1;
      Q = 5;
      
      % Set up StateSpace with initial values
      ss = StateSpace(Z, d, H, T, c, R, Q);
      ss = ss.setInvariantTau;
      ss = ss.setDefaultInitial;
      ss.usingDefaulta0 = false;
      ss.usingDefaultP0 = false;

      % Create ThetaMap
      tm = ThetaMap.ThetaMapAll(ss);
      
      % Theta -> StateSpace -> Theta
      theta = [2 0 2.0986 1 0 1 2.6094 1 1]';
      ss = tm.theta2system(theta);
      thetaNew = tm.system2theta(ss);
      
      testCase.verifyEqual(theta, thetaNew, 'AbsTol', 1e-16);
    end
    
    function systemThetaSystem_a0P0(testCase)
      % StateSpace -> Theta -> StateSpace with explicit initial values
      Z = 1;
      d = 0;
      H = 3;
      T = 1;
      c = 0;
      R = 1;
      Q = 5;
      
      % Set up StateSpace with initial values
      ss = StateSpace(Z, d, H, T, c, R, Q);
      ss = ss.setInvariantTau;
      ss = ss.setDefaultInitial;
      ss.usingDefaulta0 = false;
      ss.usingDefaultP0 = false;

      % Create ThetaMap
      tm = ThetaMap.ThetaMapAll(ss);
      
      % StateSpace -> Theta -> StateSpace
      theta = tm.system2theta(ss);
      ssNew = tm.theta2system(theta);
      
      testCase.verifyEqual(ss.Z, ssNew.Z, 'AbsTol', 1e-16);
      testCase.verifyEqual(ss.d, ssNew.d, 'AbsTol', 1e-16);
      testCase.verifyEqual(ss.H, ssNew.H, 'AbsTol', 1e-16);
      testCase.verifyEqual(ss.T, ssNew.T, 'AbsTol', 1e-16);
      testCase.verifyEqual(ss.c, ssNew.c, 'AbsTol', 1e-16);
      testCase.verifyEqual(ss.R, ssNew.R, 'AbsTol', 1e-16);
      testCase.verifyEqual(ss.Q, ssNew.Q, 'AbsTol', 1e-16);
      testCase.verifyEqual(ss.a0, ssNew.a0, 'AbsTol', 1e-16);
      testCase.verifyEqual(ss.P0, ssNew.P0, 'AbsTol', 1e-9);
    end
    
    %% Test that the map obeys bounds
    function testLowerBound(testCase)
      % Test that we get an error with a StateSpace with elements below the
      % lower bound of a ThetaMap
      error();
    end
    
    function testUpperBound(testCase)
      % Test that we get an error with a StateSpace with elements above the
      % upper bound of a ThetaMap
      error();
    end
    
    %% Test that we can edit a ThetaMap and remove elements from theta
    function testRemoveThetaElements(testCase)
      % Test that we can zero out an element of index and let validate collapse
      % the theta vector.
      
      % Create ThetaMap
      p = 2; m = 1; 
      ssGen = generateARmodel(p, m, false);
      tm = ThetaMap.ThetaMapAll(ssGen);
      
      % Edit and validate
      tmNew = tm;
      tmNew.index.T(2, :) = 0;
      tmNew.fixed.T(2, :) = [1 0];
      tmNew = tmNew.validateThetaMap();
      
      testCase.verifyEqual(tmNew.nTheta, tm.nTheta - 2);      
    end
    
  end
end