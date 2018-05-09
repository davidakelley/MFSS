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
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testAR1M_a(testCase)
      % Stationary multivariate AR(1) test
      ss = generateARmodel(3, 0, false);
      y = generateData(ss, 150);
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered(y);
      dataEff = reshape(sum(decomp_data, 2), [1 151]);
      reconstruct_a = dataEff + decomp_const;
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testAR1Mc_a(testCase)
      % Stationary multivariate correlated errors AR(1) test
      ss = generateARmodel(3, 0, true);
      y = generateData(ss, 150);
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered(y);
      dataEff = reshape(sum(decomp_data, 2), [1 151]);
      reconstruct_a = dataEff + decomp_const;
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testNile_a(testCase)
      % Use estimated model from DK
      % This is non-stationary.
      ss = StateSpace(1, 0, 15099, 1, 0, 1, 1469.1);
      y = testCase.data.nile';
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered(y);
      reconstruct_a = squeeze(decomp_data)';
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11, 'AbsTol', 1e-12);
      testCase.verifyEqual(decomp_const, zeros(1, 101));
    end
    
    function testARpM_a(testCase)
      % Do a multivariate non-stationary AR(p) test
      ss = generateARmodel(5, 3, false);
      ss.T(1,1) = 1.001 - sum(ss.T(1,2:end));
      y = generateData(ss, 150);
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered(y);
      dataEff = reshape(sum(decomp_data, 2), size(a));
      reconstruct_a = dataEff + decomp_const;
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11, 'AbsTol', 1e-12);
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
      dataEff = reshape(sum(decomp_data, 2), size(a));
      reconstruct_a = dataEff + decomp_const;
      
      testCase.verifyEqual(reconstruct_a, a, 'RelTol', 1e-13, 'AbsTol', 1e-12);
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
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11, 'AbsTol', 1e-12);
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
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11, 'AbsTol', 1e-12);
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
  
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    %% Tests of the weights for r_t
    function test_vu(testCase)
      % Test we can reconstruct v^u using the quantities generated for the r recursion
      % Nonstationary model with > 2 observation series.
      ss = generateARmodel(4, 3, true);
      ss.Z = [ones(4,1) zeros(4,3)];
      ss.H = eye(4);
      ss.Q = 1;
      ss.T(1,1) = 1.001 - sum(ss.T(1,2:end));
      y = generateData(ss, 10);
      
      % Get actual r values
      ss.useMex = false;
      [~, ~, fOut] = ss.filter(y);
      
      % Get decomposition 
      % Prep from decompose_smoothed
      ss.validateKFilter();
      ss = ss.checkSample(y);
      [ssUni, yUni, C] = ss.prepareFilter(y);
      
      comp = ssUni.build_smoother_weight_parts(yUni, fOut);

      v_reconstruct = nan(size(fOut.v));
      for iT = 1:ssUni.n
        v_reconstruct(:,iT) = comp.Ay(:,:,iT) * C(:,:,ssUni.tau.H(iT)) * y(:,iT) ...
          - comp.Ay(:,:,iT) * C(:,:,ssUni.tau.H(iT)) * ssUni.d(:,ssUni.tau.d(iT)) ...
          - comp.Aa(:,:,iT) * fOut.a(:,iT);
      end
      
      testCase.verifyEqual(fOut.v, v_reconstruct, 'AbsTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testAR11_r(testCase)
      % Stationary multivariate AR(1) test
      ss = generateARmodel(1, 0, false);
      y = generateData(ss, 6);
      y = nan(size(y));
      y(3) = 100;
      
      % Get actual r values
      [~, sOut, fOut] = ss.smooth(y);
      r = sOut.r;
      
      % Prep from decompose_smoothed
      ss.validateKFilter();
      ss = ss.checkSample(y);
      ssMulti = ss; 
      [ssUni, yUni, C] = ss.prepareFilter(y);
      
      fWeights = ssUni.filter_weights(yUni, fOut, ssMulti, C);
      rWeights = ssUni.r_weights(yUni, fOut, fWeights, ssMulti, C);
      r_reconstruct = zeros(size(r));
      for iT = 1:ss.n
        if any(~cellfun(@isempty, rWeights.y(iT,:)))
          r_reconstruct(:,iT) = sum(cat(3, ...
            rWeights.y{iT, ~cellfun(@isempty, rWeights.y(iT,:))}), 3);
        end
      end
      
      testCase.verifyEqual(r, r_reconstruct, 'AbsTol', 1e-11, 'AbsTol', 1e-12);
    end

    function testLLM_diffuse_r(testCase)
      % Local level model test
      ss = generateARmodel(1, 0, true);
      ss.H = 1;
      ss.Q = 1;
      ss.T(1,:) = 1;
      y = generateData(ss, 100);
      
      % Get actual r values
      ss.useMex = false;
      [~, sOut, fOut] = ss.smooth(y);
      r = sOut.r;
      r1 = sOut.r1;
      
      % Get decomposition 
      % Prep from decompose_smoothed
      ss.validateKFilter();
      ss = ss.checkSample(y);
      ssMulti = ss;
      [ssUni, yUni, C] = ss.prepareFilter(y);
      
      % Decompose
      fWeights = ssUni.filter_weights(yUni, fOut, ssMulti, C);
      [rWeights, r1Weights] = ssUni.r_weights(yUni, fOut, fWeights, ssMulti, C);
      r_reconstruct = zeros(size(r));
      r1_reconstruct = zeros(size(r1));
      for iT = 1:ss.n
        if any(~cellfun(@isempty, rWeights.y(iT,:)))
          r_reconstruct(:,iT) = sum(cat(3, ...
            rWeights.y{iT, ~cellfun(@isempty, rWeights.y(iT,:))}), 3);
        end
      end
      for iT = 1:fOut.dt
        if any(~cellfun(@isempty, r1Weights.y(iT,:)))
          r1_reconstruct(:,iT) = sum(cat(3, ...
            r1Weights.y{iT, ~cellfun(@isempty, r1Weights.y(iT,:))}), 3);
        end
      end
      
      % Test
      testCase.verifyEqual(r_reconstruct, r, 'AbsTol', 1e-11, 'AbsTol', 1e-12);
      testCase.verifyEqual(r1_reconstruct, r1, 'AbsTol', 1e-11, 'AbsTol', 1e-12);
    end

    function testARp1_diffuse_r(testCase)
      % Nonstationary AR(p) test
      rng(123);
      
      ss = generateARmodel(1, 4, true);
      ss.H = 1;
      ss.Q = 1;
      ss.T(1,1) = 1.0001 - sum(ss.T(1,2:end));
      y = generateData(ss, 100);
      
      % Get actual r values
      ss.useMex = false;
      [~, sOut, fOut] = ss.smooth(y);
      r = sOut.r;
      r1 = sOut.r1;
      
      % Get decomposition 
      % Prep from decompose_smoothed
      ss.validateKFilter();
      ss = ss.checkSample(y);
      ssMulti = ss;
      [ssUni, yUni, C] = ss.prepareFilter(y);
      
      % Decompose
      fWeights = ssUni.filter_weights(yUni, fOut, ssMulti, C);
      [rWeights, r1Weights] = ssUni.r_weights(yUni, fOut, fWeights, ssMulti, C);
      r_reconstruct = zeros(size(r));
      r1_reconstruct = zeros(size(r1));
      for iT = 1:ss.n
        if any(~cellfun(@isempty, rWeights.y(iT,:)))
          r_reconstruct(:,iT) = sum(cat(3, ...
            rWeights.y{iT, ~cellfun(@isempty, rWeights.y(iT,:))}), 3);
        end
      end
      for iT = 1:fOut.dt
        if any(~cellfun(@isempty, r1Weights.y(iT,:)))
          r1_reconstruct(:,iT) = sum(cat(3, ...
            r1Weights.y{iT, ~cellfun(@isempty, r1Weights.y(iT,:))}), 3);
        end
      end
      
      % Test
      testCase.verifyEqual(r_reconstruct, r, 'RelTol', 1e-10, 'AbsTol', 1e-12);
      testCase.verifyEqual(r1_reconstruct, r1, 'RelTol', 1e-10, 'AbsTol', 1e-12);
    end
    
    function testAR1M_diffuse_r(testCase)
      % Nonstationary AR(1) test
      ss = generateARmodel(3, 0, true);
      ss.Z = ones(3,1);
      ss.H = eye(3);
      ss.Q = 1;
      ss.T(1,1) = 1;
      y = generateData(ss, 2);
      
      % Get actual r values
      ss.useMex = false;
      [~, sOut, fOut] = ss.smooth(y);
      r = sOut.r;
      r1 = sOut.r1;
      
      % Get decomposition 
      % Prep from decompose_smoothed
      ss.validateKFilter();
      ss = ss.checkSample(y);
      ssMulti = ss;
      [ssUni, yUni, C] = ss.prepareFilter(y);
      
      % Decompose
      fWeights = ssUni.filter_weights(yUni, fOut, ssMulti, C);
      [rWeights, r1Weights] = ssUni.r_weights(yUni, fOut, fWeights, ssMulti, C);
      r_reconstruct = zeros(size(r));
      r1_reconstruct = zeros(size(r1));
      for iT = 1:ss.n
        if any(~cellfun(@isempty, rWeights.y(iT,:)))
          r_reconstruct(:,iT) = sum(sum(cat(3, ...
            rWeights.y{iT, ~cellfun(@isempty, rWeights.y(iT,:))}), 3), 2);
        end
      end
      for iT = 1:fOut.dt
        if any(~cellfun(@isempty, r1Weights.y(iT,:)))
          r1_reconstruct(:,iT) = sum(sum(cat(3, ...
            r1Weights.y{iT, ~cellfun(@isempty, r1Weights.y(iT,:))}), 3), 2);
        end
      end
      
      % Test
      testCase.verifyEqual(r_reconstruct, r, 'RelTol', 1e-11, 'AbsTol', 1e-12);
      testCase.verifyEqual(r1_reconstruct, r1, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end    

    function testARpM_diffuse_r(testCase)
      % Nonstationary AR(p) test
      ss = generateARmodel(4, 1, true);
      ss.T(1,1) = 1.001 - sum(ss.T(1,2:end));
      y = generateData(ss, 10);
      
      % Get actual r values
      ss.useMex = false;
      [~, sOut, fOut] = ss.smooth(y);
      r = sOut.r;
      r1 = sOut.r1;
      
      % Get decomposition 
      % Prep from decompose_smoothed
      ss.validateKFilter();
      ss = ss.checkSample(y);
      ssMulti = ss;
      [ssUni, yUni, C] = ss.prepareFilter(y);

      % Decompose
      fWeights = ssUni.filter_weights(yUni, fOut, ssMulti, C);
      [rWeights, r1Weights] = ssUni.r_weights(yUni, fOut, fWeights, ssMulti, C);
      r_reconstruct = zeros(size(r));
      r1_reconstruct = zeros(size(r1));
      for iT = 1:ss.n
        if any(~cellfun(@isempty, rWeights.y(iT,:)))
          r_reconstruct(:,iT) = sum(sum(cat(3, ...
            rWeights.y{iT, ~cellfun(@isempty, rWeights.y(iT,:))}), 3), 2);
        end
      end
      for iT = 1:fOut.dt
        if any(~cellfun(@isempty, r1Weights.y(iT,:)))
          r1_reconstruct(:,iT) = sum(sum(cat(3, ...
            r1Weights.y{iT, ~cellfun(@isempty, r1Weights.y(iT,:))}), 3), 2);
        end
      end
      
      % Test
      testCase.verifyEqual(r_reconstruct, r, 'RelTol', 1e-11, 'AbsTol', 1e-12);
      testCase.verifyEqual(r1_reconstruct, r1, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    %% Smoother weight tests
    function testSimple_alpha(testCase)
      % Stationary multivariate AR(1) test
      ss = generateARmodel(1, 0, false);
      y = generateData(ss, 50);
      y = zeros(size(y));
      y(2) = 100;
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testAR1_alpha(testCase)
      % Stationary univariate AR(2) test
      ss = generateARmodel(1, 1, false);
      y = generateData(ss, 50);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      reconstruct_alpha = reshape(sum(decomp_data, 2), size(alpha)) + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testAR1_multivariate_simple_alpha(testCase)
      % Stationary multivariate AR(1) test
      ss = generateARmodel(3, 0, true);
      y = generateData(ss, 10);
      y = zeros(size(y));
      y(:,4) = [100; 0; 0];
      ss.T = 0;
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testAR1_fullH_alpha(testCase)
      % Stationary multivariate correlated errors AR(1) test
      ss = generateARmodel(3, 0, true);
      y = generateData(ss, 50);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testNile_alpha(testCase)
      % Use estimated model from DK
      ss = StateSpace(1, 0, 15099, 1, 0, 1, 1469.1);
      y = testCase.data.nile';
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      reconstruct_alpha = squeeze(decomp_data)';
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
      testCase.verifyEqual(decomp_const, zeros(1, 100));
    end
    
    function testARpM_alpha(testCase)
      % Do a multivariate non-stationary AR(p) test
      ss = generateARmodel(2, 3, true);
      y = generateData(ss, 50);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testAR11_diffuse_alpha(testCase)
      % Do a multivariate non-stationary AR(p) test
      ss = generateARmodel(1, 0, true);
      ss.T(1,1) = 1.001;
      y = generateData(ss, 50);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
        
    function testAR1M_diffuse_alpha(testCase)
      % Do a multivariate non-stationary AR(p) test
      ss = generateARmodel(4, 0, true);
      ss.T(1,1) = 1.001;
      y = generateData(ss, 50);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testARpM_diffuse_alpha(testCase)
      % Do a multivariate non-stationary AR(p) test
      ss = generateARmodel(4, 3, true);
      ss.T(1,1) = 1.1 - sum(ss.T(1,2:end));
      y = generateData(ss, 50);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testARp_uni_d_alpha(testCase)
      % Do a multivariate non-stationary AR(p) test with constants
      ss = generateARmodel(1, 3, false);
      ss.d = 4; 
      y = generateData(ss, 50);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testARpM_d_alpha(testCase)
      % Do a multivariate non-stationary AR(p) test with constants
      ss = generateARmodel(5, 3, false);
      ss.d = [-1; 2; .5; -.2; 0];
      y = generateData(ss, 50);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testARpM_uni_c_alpha(testCase)
      % Do a multivariate non-stationary AR(p) test with constants
      ss = generateARmodel(1, 0, false);
      ss.c = .4;

      y = generateData(ss, 50);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;

      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testARpM_c_alpha(testCase)
      % Do a multivariate non-stationary AR(p) test with constants
      ss = generateARmodel(5, 3, false);
      ss.c = [-1; .5; 0; 12];
      
      y = generateData(ss, 50);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;

      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testARpM_diffuse_const_alpha(testCase)
      % Do a multivariate non-stationary AR(p) test with constants
      rng(123);
      
      ss = generateARmodel(5, 3, false);
      ss.d = [-1; 2; .5; -.2; 0];
      ss.c = [-1; .5; 0; 0];
      ss.T(1,1) = 1.002 - sum(ss.T(1,2:end));
      y = generateData(ss, 50);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testARpM_uni_a0_alpha(testCase)
      % Do a multivariate stationary AR(p) test with explicit a0
      ss = generateARmodel(5, 0, false);
      ss.a0 = 10;

      y = generateData(ss, 50);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;

      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testARpM_a0_alpha(testCase)
      % Do a multivariate stationary AR(p) test with explicit a0
      ss = generateARmodel(1, 3, false);
      ss.a0 = [100; 0; 0; 0];
      ss.T(1,:) = [.4 .3 .2 .05];
      
      y = generateData(ss, 50);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
            
      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testARpM_diffuse_a0_alpha(testCase)
      % Do a multivariate non-stationary AR(p) test with explicit a0
      rng(123);
      
      ss = generateARmodel(5, 3, false);      
      ss.a0 = [10; 9.9; 9.9; 9.9];
      ss.T(1,1) = 1.01 - sum(ss.T(1,2:end));
      y = generateData(ss, 50);
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
    function testARpM_diffuse_nan_alpha(testCase)
      % Do a multivariate non-stationary AR(p) test
      ss = generateARmodel(4, 3, true);
      ss.T(1,1) = 1.1 - sum(ss.T(1,2:end));
      y = generateData(ss, 50);
      y(2,:) = nan;
      
      alpha = ss.smooth(y);
      [decomp_data, decomp_const] = ss.decompose_smoothed(y);
      dataEff = reshape(sum(decomp_data, 2), size(alpha));
      reconstruct_alpha = dataEff + decomp_const;
      
      testCase.verifyEqual(alpha, reconstruct_alpha, 'RelTol', 1e-11, 'AbsTol', 1e-12);
    end
    
  end
end