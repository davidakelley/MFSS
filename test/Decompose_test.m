% Test the Kalman weight decompositions for the filter and smoother
% David Kelley, 2017

classdef Decompose_test < matlab.unittest.TestCase
  properties
    data = struct;
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      baseDir = fileparts(fileparts(mfilename('fullpath')));
      addpath(baseDir);
      addpath(fullfile(baseDir, 'examples'));
      
      data_load = load(fullfile(baseDir, 'examples', 'data', 'dk.mat'));
      testCase.data.nile = data_load.nile;
    end
  end
  
  methods (Test)
    %% Filter weight tests
    function testAR1_a(testCase)
      % Stationary univariate AR(2) test
      ss = generateARmodel(1, 1, false);
      y = generateData(ss, 150);
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered(y);
      reconstruct_a = squeeze(decomp_data) + decomp_const;
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11);
    end
    
    function testAR1M_a(testCase)
      % Stationary multivariate AR(1) test
      ss = generateARmodel(3, 0, false);
      y = generateData(ss, 150);
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered(y);
      dataEff = reshape(sum(decomp_data, 2), [1 151]);
      reconstruct_a = dataEff + decomp_const;
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11);
    end
    
    function testAR1Mc_a(testCase)
      % Stationary multivariate correlated errors AR(1) test
      ss = generateARmodel(3, 0, true);
      y = generateData(ss, 150);
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered(y);
      dataEff = reshape(sum(decomp_data, 2), [1 151]);
      reconstruct_a = dataEff + decomp_const;
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11);
    end
    
    function testNile_a(testCase)
      % Use estimated model from DK
      % This is non-stationary.
      ss = StateSpace(1, 0, 15099, 1, 0, 1, 1469.1);
      y = testCase.data.nile';
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered(y);
      reconstruct_a = squeeze(decomp_data)';
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11);
      testCase.verifyEqual(decomp_const, zeros(1, 101));
    end
    
    function testARpM_a(testCase)
      % Do a multivariate non-stationary AR(p) test
      ss = generateARmodel(2, 3, true);
      ss.T(1,1) = 1.001 - sum(ss.T(1,2:end));
      y = generateData(ss, 150);
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered(y);
      dataEff = reshape(sum(decomp_data, 2), size(a));
      reconstruct_a = dataEff + decomp_const;
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11);
    end
    
    function testARpM_const_a(testCase)
      % Do a multivariate non-stationary AR(p) test with constants
      ss = generateARmodel(5, 3, false);
      ss.d = [-1; 2; .5; -.2; 0];
      ss.c = [-1; .5; 0; 0];
      ss.T(1,1) = 1.01 - sum(ss.T(1,2:end));
      y = generateData(ss, 150);
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered(y);
      dataEff = squeeze(sum(decomp_data, 2));
      reconstruct_a = dataEff + decomp_const;
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11);
    end
    
    function testARpM_a0_a(testCase)
      % Do a multivariate non-stationary AR(p) test with explicit a0
      ss = generateARmodel(5, 3, false);
      ss.a0 = [10; 9.9; 9.9; 9.9];
      ss.T(1,1) = 1.01 - sum(ss.T(1,2:end));
      y = generateData(ss, 150);
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered(y);
      dataEff = squeeze(sum(decomp_data, 2));
      reconstruct_a = dataEff + decomp_const;
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11);
    end
    
    function testARpTVP_a(testCase)
      % Do a time-varrying parameters test
      ss = generateARmodel(2, 2, true);
      ss.Q = 1;
      
      ss = ss.setTimeVarrying(12);
      ss = ss.setInvariantTau();
      ss.tau.T = [repmat([1; 2], [6 1]); 1];
      ss.T(:,:,2) = ss.T;
      ss.T(1,1,2) = -ss.T(1,1,2);
      
      y = generateData(ss, 12);

      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered(y);
      dataEff = reshape(sum(decomp_data, 2), size(a));
      reconstruct_a = dataEff + decomp_const;
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11);
    end
    
    function testARp_a_oddZ(testCase)
      % Do a test that I'm not sure why its failing
      % It was failing due to L being incorrectly specified.
      ss = generateARmodel(2, 1, true);
      % Neccessary to have one obs load off the second state
      ss.Z = [1 0; 0 1];
      
      % Not neccessary but makes differences easier to see.
      ss.T(1,:) = [1 0]; 
      ss.H = diag(ones(2,1));
      ss.Q = 1;
      ss.a0 = [1; 0];
      ss.P0 = eye(2);
      
      y = nan(2, 2);
      y(:, 1) = [0; 0];
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered(y);
      reconstruct_a = reshape(sum(decomp_data, 2), size(a)) + decomp_const;
  
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11);
    end
    
    %% Simple tests of the weights on r_t
    % These are really unneccessary if the tests on alpha succeed but are
    % hopefully valuable while developing the rest of the smoother weights.
    function testSimple_r(testCase)
      % Stationary multivariate AR(1) test
      ss = generateARmodel(1, 0, false);
      y = generateData(ss, 6);
      y = nan(size(y));
      y(3) = 100;
      
      % Get actual r values
      [~, sOut] = ss.smooth(y);
      r = sOut.r;
      
      % Prep from decompose_smoothed
      ss.validateKFilter();
      ss = ss.checkSample(y);
      ssMulti = ss; 
      [ssUni, yUni, C] = ss.prepareFilter(y);
      
      [~, sOut, fOut] = ssUni.smooth(yUni);
      [wa, wac, wad, waa0] = ssUni.filter_weights(yUni, fOut, ssMulti, C);
      omegar = ssUni.smoother_weights_r(yUni, fOut, wa, wad, wac, waa0, sOut);
      r_reconstruct = reshape(sum(sum(omegar, 2), 4), size(r));
      
      testCase.verifyEqual(r, r_reconstruct, 'AbsTol', 1e-11);
    end
    
    function testAR1M_r_T0(testCase)
      % Stationary multivariate AR(1) test
      ss = generateARmodel(3, 0, true);
      y = generateData(ss, 6);
      y = nan(size(y));
      y(:,3) = [100; 0; 0];
      % y(1,3) = 100;
      ss.T = 0;
      
      % Get actual r values
      [~, sOut] = ss.smooth(y);
      r = sOut.r;
      
      % Get decomposition 
      % Prep from decompose_smoothed
      ss.validateKFilter();
      ss = ss.checkSample(y);
      ssMulti = ss; 
      [ssUni, yUni, C] = ss.prepareFilter(y);
      [~, sOut, fOut] = ssUni.smooth(yUni);
      % Decompose
      [wa, wac, wad, waa0] = ssUni.filter_weights(yUni, fOut, ssMulti, C);
      omegar = ssUni.smoother_weights_r(yUni, fOut, wa, wad, wac, waa0, sOut);
      r_reconstruct = reshape(sum(sum(omegar, 2), 4), size(r));
      
      % Test
      testCase.verifyEqual(r, r_reconstruct, 'AbsTol', 1e-11);
    end
    
    %% Smoother weight tests
    function testSimple_alpha(testCase)
      % Stationary multivariate AR(1) test
      ss = generateARmodel(1, 0, false);
      y = generateData(ss, 150);
      y = zeros(size(y));
      y(2) = 100;
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'AbsTol', 1e-11);
    end
    
    function testAR1_alpha(testCase)
      % Stationary univariate AR(2) test
      ss = generateARmodel(1, 1, false);
      y = generateData(ss, 150);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      reconstruct_alpha = reshape(sum(decomp_data, 2), size(alpha)) + decomp_const;
      
      testCase.verifyEqual(reconstruct_alpha, alpha, 'AbsTol', 1e-11);
    end
    
    function testAR1M_alpha(testCase)
      % Stationary multivariate AR(1) test
      ss = generateARmodel(3, 0, true);
      y = generateData(ss, 10);
      y = zeros(size(y));
      y(:,4) = [100; 0; 0];
      % y(:,5) = [0; 0; 100];
      ss.T = 0;
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'AbsTol', 1e-11);
    end
    
    function testAR1Mc_alpha(testCase)
      % Stationary multivariate correlated errors AR(1) test
      ss = generateARmodel(3, 0, true);
      y = generateData(ss, 150);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'AbsTol', 1e-11);
    end
    
    function testNile_alpha(testCase)
      % Use estimated model from DK
      % This is non-stationary.
      ss = StateSpace(1, 0, 15099, 1, 0, 1, 1469.1);
      y = testCase.data.nile';
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      reconstruct_alpha = squeeze(decomp_data)';
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'AbsTol', 1e-11);
      testCase.verifyEqual(decomp_const, zeros(1, 100));
    end
    
    function testARpM_alpha(testCase)
      % Do a multivariate non-stationary AR(p) test
      ss = generateARmodel(2, 3, true);
      ss.T(1,1) = 1.001 - sum(ss.T(1,2:end));
      y = generateData(ss, 150);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'AbsTol', 1e-11);
    end
    
    function testARpM_const_alpha(testCase)
      % Do a multivariate non-stationary AR(p) test with constants
      ss = generateARmodel(5, 3, false);
      %ss.d = [-1; 2; .5; -.2; 0];
  
      % FIXME: include c
      ss.c = [-1; .5; 0; 0];

      % FIXME: replace 1.01 instead of .99
      ss.T(1,1) = .9 - sum(ss.T(1,2:end));
      y = generateData(ss, 150);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'AbsTol', 1e-11);
    end
    
    function testARpM_a0_alpha(testCase)
      % Do a multivariate non-stationary AR(p) test with explicit a0
      ss = generateARmodel(5, 3, false);
      ss.a0 = [10; 9.9; 9.9; 9.9];
      ss.T(1,1) = 1.01 - sum(ss.T(1,2:end));
      y = generateData(ss, 150);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'AbsTol', 1e-11);
    end
  end
end