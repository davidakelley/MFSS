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
      [decomp_data, decomp_const] = ss.decompose_filtered_uni(y);
      reconstruct_a = squeeze(decomp_data) + decomp_const;
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-8);
    end

    function testAR1M_a(testCase)
      % Stationary multivariate AR(1) test
      ss = generateARmodel(3, 0, false);
      y = generateData(ss, 150);
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered_uni(y);
      reconstruct_a = sum(squeeze(decomp_data(1,:,:)) + decomp_const(1,:), 1);
      
      testCase.verifyEqual(a(1,:), reconstruct_a, 'AbsTol', 1e-8);
    end
      
    function testAR1Mc_a(testCase)
      % Stationary multivariate correlated errors AR(1) test
      ss = generateARmodel(3, 0, true);
      y = generateData(ss, 150);
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered_uni(y);
      reconstruct_a = sum(squeeze(decomp_data(1,:,:)) + decomp_const(1,:), 1);
      
      testCase.verifyEqual(a(1,:), reconstruct_a, 'AbsTol', 1e-8);
    end
      
    function testNile_a(testCase)
      % Use estimated model from DK
      % This is non-stationary.
      ss = StateSpace(1, 0, 15099, 1, 0, 1, 1469.1);
      y = testCase.data.nile';
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered_uni(y);
      reconstruct_a = squeeze(decomp_data)';
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-8);
      testCase.verifyEqual(decomp_const, zeros(1, 101));
    end
    
    function testARpM_a(testCase)
      % Do a multivariate non-stationary AR(p) test
      ss = generateARmodel(2, 3, true);
      ss.T(1,1) = 1.001 - sum(ss.T(1,2:end));
      y = generateData(ss, 150);
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered_uni(y);
      reconstruct_a = sum(squeeze(decomp_data(1,:,:)) + decomp_const(1,:), 1);
      
      testCase.verifyEqual(a(1,:), reconstruct_a, 'AbsTol', 1e-8);
    end
    
    function testARpM_const_a(testCase)
      % Do a multivariate non-stationary AR(p) test with constants
      ss = generateARmodel(5, 3, false);
      ss.d = [-1; 2; .5; -.2; 0];
      ss.c = [-1; .5; 0; 0];
      ss.T(1,1) = 1.01 - sum(ss.T(1,2:end));
      y = generateData(ss, 150);
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered_uni(y);
      dataEff = squeeze(sum(decomp_data, 2));
      reconstruct_a = dataEff + decomp_const;
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-8);
    end
    
     function testARpM_a0_a(testCase)
      % Do a multivariate non-stationary AR(p) test with explicit a0
      ss = generateARmodel(5, 3, false);
      ss.a0 = [10; 9.9; 9.9; 9.9];
      ss.T(1,1) = 1.01 - sum(ss.T(1,2:end));
      y = generateData(ss, 150);
      
      a = ss.filter(y);
      [decomp_data, decomp_const] = ss.decompose_filtered_uni(y);
      dataEff = squeeze(sum(decomp_data, 2));
      reconstruct_a = dataEff + decomp_const;
      
      testCase.verifyEqual(a, reconstruct_a, 'AbsTol', 1e-8);
    end
    
    %% Smoother weight tests
    
  end
end