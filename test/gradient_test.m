%% Tests for StateSpace.gradient
%
% Run with
% runtests('test/gradient_test.m');

% David Kelley, 2016

classdef gradient_test < matlab.unittest.TestCase
  
  properties
    ar2obs2
    ar2obs2data
    ar2obs2tm
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      baseDir =  [subsref(strsplit(mfilename('fullpath'), 'MFSS'), ...
        struct('type', '{}', 'subs', {{1}})) 'MFSS'];
      addpath(baseDir);
      addpath(fullfile(baseDir, 'examples'));
    
      ss = generateARmodel(2, 2, false);
      y = generateData(ss, 50);
      testCase.ar2obs2 = ss;
      testCase.ar2obs2data = y;
    end
  end
  
  methods (Test)
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
      
      testCase.verifyEqual(Ga0analytic, Ga0numeric, 'AbsTol', 1e-6);
    end
    
    function testParamGradQ0(testCase)
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
      baseP0 = ss.Q0;      
      epsilon = 1e-8;
      GP0numeric = nan(size(GP0analytic));
      for iT = 1:tm.nTheta
        iTheta = theta;
        iTheta(iT) = theta(iT) + epsilon;
        ssNew = tm.theta2system(iTheta);
        ssNew = ssNew.setInvariantTau;
        ssNew = ssNew.setDefaultInitial();
        GP0numeric(iT, :) = reshape((ssNew.Q0 - baseP0), [], 1) ./ epsilon;
      end
      
      testCase.verifyEqual(GP0analytic, GP0numeric, 'AbsTol', 1e-6);
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
    
    %% Loglikelihood gradients for single parameters
    function testGradZ(testCase)
      ss = testCase.ar2obs2;
      y = testCase.ar2obs2data;
      
      ssE = StateSpaceEstimation(nan(size(ss.Z)), ss.d, ss.H, ss.T, ss.c, ss.R, ss.Q);
      tm = ssE.ThetaMapping;
      
      [~, analytic] = ss.gradient(y, tm);
      numeric = numericGradient(ss, tm, y, 1e-8);
      
      testCase.verifyEqual(analytic, numeric, 'AbsTol', 1e-4);
    end
    
    function testGradd(testCase)
      ss = testCase.ar2obs2;
      y = testCase.ar2obs2data;
      
      ssE = StateSpaceEstimation(ss.Z, nan(size(ss.d)), ss.H, ss.T, ss.c, ss.R, ss.Q);
      tm = ssE.ThetaMapping;
      
      [~, analytic] = ss.gradient(y, tm);
      numeric = numericGradient(ss, tm, y, 1e-8);
      
      testCase.verifyEqual(analytic, numeric, 'AbsTol', 1e-4);
    end
    
    function testGradH(testCase)
      ss = testCase.ar2obs2;
      y = testCase.ar2obs2data;
      
      ssE = StateSpaceEstimation(ss.Z, ss.d, nan(size(ss.H)), ss.T, ss.c, ss.R, ss.Q);
      tm = ssE.ThetaMapping;
      
      [~, analytic] = ss.gradient(y, tm);
      numeric = numericGradient(ss, tm, y, 1e-8);
      
      testCase.verifyEqual(analytic, numeric, 'AbsTol', 1e-4);
    end
    
    function testGradT(testCase)
      ss = testCase.ar2obs2;
      y = testCase.ar2obs2data;
      
      ssE = StateSpaceEstimation(ss.Z, ss.d, ss.H, nan(size(ss.T)), ss.c, ss.R, ss.Q);
      tm = ssE.ThetaMapping;
      
      [~, analytic] = ss.gradient(y, tm);
      numeric = numericGradient(ss, tm, y, 1e-8);
      
      testCase.verifyEqual(analytic, numeric, 'AbsTol', 1e-4);
    end
    
    function testGradc(testCase)
      ss = testCase.ar2obs2;
      y = testCase.ar2obs2data;
      
      ssE = StateSpaceEstimation(ss.Z, ss.d, ss.H, ss.T, nan(size(ss.c)), ss.R, ss.Q);
      tm = ssE.ThetaMapping;
      
      [~, analytic] = ss.gradient(y, tm);
      numeric = numericGradient(ss, tm, y, 1e-8);
      
      testCase.verifyEqual(analytic, numeric, 'AbsTol', 1e-4);
    end
    
    function testGradR(testCase)
      ss = testCase.ar2obs2;
      y = testCase.ar2obs2data;
      
      ssE = StateSpaceEstimation(ss.Z, ss.d, ss.H, ss.T, ss.c, nan(size(ss.R)), ss.Q);
      tm = ssE.ThetaMapping;
      
      [~, analytic] = ss.gradient(y, tm);
      numeric = numericGradient(ss, tm, y, 1e-8);
      
      testCase.verifyEqual(analytic, numeric, 'AbsTol', 1e-4);
    end
    
    function testGradQ(testCase)
      ss = testCase.ar2obs2;
      y = testCase.ar2obs2data;
      
      ssE = StateSpaceEstimation(ss.Z, ss.d, ss.H, ss.T, ss.c, ss.R, nan(size(ss.Q)));
      tm = ssE.ThetaMapping;
      
      [~, analytic] = ss.gradient(y, tm);
      numeric = numericGradient(ss, tm, y, 1e-8);
      
      testCase.verifyEqual(analytic, numeric, 'AbsTol', 1e-4);
    end
    
    %% Likelihood gradients for parameters at estimated optima
    function testGraddEstim(testCase)
      ss = testCase.ar2obs2;
      y = testCase.ar2obs2data;
      
      ssE = StateSpaceEstimation(ss.Z, nan(size(ss.d)), ss.H, ss.T, ss.c, ss.R, ss.Q);
      tm = ssE.ThetaMapping;
      ssE.diagnosticPlot = false;
      
      ss1 = ssE.estimate(y, ss);
      
      [~, analytic] = ss1.gradient(y, tm);
      numeric = numericGradient(ss1, tm, y, 1e-8);
      
      testCase.verifyEqual(analytic, numeric, 'AbsTol', 1e-4);
    end

  end
end
