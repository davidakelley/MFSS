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
    function testThetaBounds(testCase)
      % Test that we get the established bounds with -Inf or Inf theta elements.
      p = 3; m = 1; 
      ss = generateARmodel(p, m, false);
      tm = ThetaMap.ThetaMapAll(ss);
      
      ssLB = ss.setAllParameters(-Inf);
      ssLB.T(1, 1) = -1;
      
      ssUB = ss.setAllParameters(Inf);
      ssUB.T(1, 1) = 1;
      
      tm = tm.addRestrictions(ssLB, ssUB);
      
      theta = tm.system2theta(ss);
      
      % Test that ThetaMap obeys lower bounds
      ssMin = tm.theta2system(repmat(-1e16, size(theta)));
      testCase.verifyEqual(ssMin.T(1,1), -1);
      
      % Test that ThetaMap obeys upper bounds
      ssMax = tm.theta2system(repmat(1e16, size(theta)));
      testCase.verifyEqual(ssMax.T(1,1), 1);
    end
    
    function testStateSpaceBounds(testCase)
      % Test that trying to get a theta from a system that voilates the bounds
      % results in an error.
      p = 3; m = 1; 
      ss = generateARmodel(p, m, false);
      tm = ThetaMap.ThetaMapAll(ss);
      
      ssLB = ss.setAllParameters(-Inf);
      ssLB.T(1, 1) = 0;
      
      ssUB = ss.setAllParameters(Inf);
      ssUB.T(1, 1) = 1;
      
      tm = tm.addRestrictions(ssLB, ssUB);
      
      ssTooLow = ss;
      ssTooLow.T(1,1) = -0.5;
      testCase.verifyError(@() tm.system2theta(ssTooLow), 'system2theta:LBound');
      
      ssTooHigh = ss;
      ssTooHigh.T(1,1) = 2;
      testCase.verifyError(@() tm.system2theta(ssTooHigh), 'system2theta:UBound');
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
    
    %% Basic parameters gradient computation
    function testParamGradZ(testCase)
      p = 2; m = 1; 
      ss = generateARmodel(p, m, false);

      ssE = StateSpaceEstimation(nan(size(ss.Z)), ss.d, ss.H, ss.T, ss.c, ss.R, ss.Q);
      tm = ssE.ThetaMapping;
      theta = tm.system2theta(ss);
      
      G = tm.parameterGradients(theta);
      
      testCase.verifyEqual(G.Z, eye(4));
    end
    
    function testParamGradd(testCase)
      p = 2; m = 1; 
      ss = generateARmodel(p, m, false);

      ssE = StateSpaceEstimation(ss.Z, nan(size(ss.d)), ss.H, ss.T, ss.c, ss.R, ss.Q);
      tm = ssE.ThetaMapping;
      theta = tm.system2theta(ss);
      
      G = tm.parameterGradients(theta);
      
      testCase.verifyEqual(G.d, eye(2));
    end
    
    function testParamGradH(testCase)
      p = 2; m = 1; 
      ss = generateARmodel(p, m, false);
      ss.H = 0.333 * eye(2);
      
      ssE = StateSpaceEstimation(ss.Z, ss.d, nan(size(ss.H)), ss.T, ss.c, ss.R, ss.Q);
      tm = ssE.ThetaMapping;
      theta = tm.system2theta(ss);
      
      G = tm.parameterGradients(theta);
      
      testCase.verifyEqual(G.H(1,1), ss.H(1), 'AbsTol', 1e-14);
      testCase.verifyEqual(G.H(2, 2:3), ones(1, 2));
      testCase.verifyEqual(G.H(3,4), ss.H(2,2), 'AbsTol', 1e-14);
    end
    
    function testParamGradT(testCase)
      p = 2; m = 1; 
      ss = generateARmodel(p, m, false);

      ssE = StateSpaceEstimation(ss.Z, ss.d, ss.H, nan(size(ss.T)), ss.c, ss.R, ss.Q);
      tm = ssE.ThetaMapping;
      theta = tm.system2theta(ss);
      
      G = tm.parameterGradients(theta);
      
      testCase.verifyEqual(G.T, eye(4));
    end
    
    function testParamGradc(testCase)
      p = 2; m = 1; 
      ss = generateARmodel(p, m, false);

      ssE = StateSpaceEstimation(ss.Z, ss.d, ss.H, ss.T, nan(size(ss.c)), ss.R, ss.Q);
      tm = ssE.ThetaMapping;
      theta = tm.system2theta(ss);
      
      G = tm.parameterGradients(theta);
      
      testCase.verifyEqual(G.c, eye(2));
    end
    
    function testParamGradR(testCase)
      p = 2; m = 1; 
      ss = generateARmodel(p, m, false);

      ssE = StateSpaceEstimation(ss.Z, ss.d, ss.H, ss.T, ss.c, nan(size(ss.R)), ss.Q);
      tm = ssE.ThetaMapping;
      theta = tm.system2theta(ss);
      
      G = tm.parameterGradients(theta);
      
      testCase.verifyEqual(G.R, eye(2));
    end
    
    function testParamGradQ(testCase)
      p = 2; m = 1; 
      ss = generateARmodel(p, m, false);

      ssE = StateSpaceEstimation(ss.Z, ss.d, ss.H, ss.T, ss.c, ss.R, nan(size(ss.Q)));
      tm = ssE.ThetaMapping;
      theta = tm.system2theta(ss);
      
      G = tm.parameterGradients(theta);
      
      testCase.verifyEqual(G.Q, ss.Q, 'AbsTol', 1e-14);
    end
    
    %% Initial value gradient computation
    function testParamGrada0(testCase)
      % Test that Ga0 output from ThetaMap.initialuValuesGradients is correct
      p = 2; m = 1;
      ss = generateARmodel(p, m, false);
      ss.c(1) = 0.1;
      tm = ThetaMap.ThetaMapAll(ss);
      theta = tm.system2theta(ss);
      G = tm.parameterGradients(theta);
      
      Ga0analytic = tm.initialValuesGradients(ss, G);
      
      % Compute numeric gradient by adding a small number to theta and remaking
      % the system, then computing the initial state.
      ss = ss.setInvariantTau;
      ss = ss.setDefaultInitial;
      basea0 = ss.a0;      
      epsilon = 1e-8;
      Ga0numeric = nan(size(Ga0analytic));
      for iT = 1:tm.nTheta
        iTheta = theta;
        iTheta(iT) = theta(iT) + epsilon;
        ssNew = tm.theta2system(iTheta);
        ssNew = ssNew.setInvariantTau;
        ssNew = ssNew.setDefaultInitial();
        Ga0numeric(iT, :) = (ssNew.a0 - basea0) ./ epsilon;
      end
      
      testCase.verifyEqual(Ga0analytic, Ga0numeric, 'AbsTol', 1e-8);
    end
    
    function testParamGradP0(testCase)
      % Test that GP0 output from ThetaMap.initialuValuesGradients is correct
      p = 2; m = 1; 
      ss = generateARmodel(p, m, false);
      ss.c(1) = 0.1;
      tm = ThetaMap.ThetaMapAll(ss);
      theta = tm.system2theta(ss);
      G = tm.parameterGradients(theta);
      
      [~, GP0analytic] = tm.initialValuesGradients(ss, G);
      
      % Compute numeric gradient by adding a small number to theta and remaking
      % the system, then computing the initial state.
      ss = ss.setInvariantTau;
      ss = ss.setDefaultInitial;
      baseP0 = ss.P0;      
      epsilon = 1e-8;
      GP0numeric = nan(size(GP0analytic));
      for iT = 1:tm.nTheta
        iTheta = theta;
        iTheta(iT) = theta(iT) + epsilon;
        ssNew = tm.theta2system(iTheta);
        ssNew = ssNew.setInvariantTau;
        ssNew = ssNew.setDefaultInitial();
        GP0numeric(iT, :) = reshape((ssNew.P0 - baseP0), [], 1) ./ epsilon;
      end
      
      testCase.verifyEqual(GP0analytic, GP0numeric, 'AbsTol', 1e-8);
    end
  
    function testParamGradDiffuse(testCase)
      % Test that Ga0 output from ThetaMap.initialuValuesGradients is correct
      p = 2; m = 1;
      ss = generateARmodel(p, m, false);
      ss.T(1, :) = [1 0.2];
      tm = ThetaMap.ThetaMapAll(ss);
      theta = tm.system2theta(ss);
      G = tm.parameterGradients(theta);
      
      [Ga0, GP0] = tm.initialValuesGradients(ss, G);
      
      testCase.verifyEqual(Ga0, zeros(size(Ga0)));
      testCase.verifyEqual(GP0, zeros(size(GP0)));
    end
  end
end