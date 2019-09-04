% Test DFM model class

% David Kelley, 2019

classdef MFDFM_test < matlab.unittest.TestCase
  properties
    data = struct;
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      % Load data
      baseDir = fileparts(fileparts(mfilename('fullpath')));
      data_load_nile = load(fullfile(baseDir, 'examples', 'durbin_koopman.mat'));
      testCase.data.nile = data_load_nile.nile;
      data_load_panel = load(fullfile(baseDir, 'examples', 'data_panel.mat'));
      testCase.data.panel = data_load_panel.y;
    end
  end
  
  methods (Test)
    %% Integration tests of MFDFM
    function testEM_AR1_single_improve(testCase)
      % Test that the the EM always improved the likelihood
      % Note that this is the same as MFVAR
      nile = testCase.data.nile;
      dfmE = MFDFM(nile, 1, 1);
      testCase.verifyWarningFree(@dfmE.estimate);
    end
    
    function testEM_AR1_panel_sf_improve(testCase)
      % Test that the the EM always improved the likelihood
      % Note that this is the same as MFVAR
      y = testCase.data.panel(:,2:end);
      dfmE = MFDFM(y, 1, 1);
      testCase.verifyWarningFree(@dfmE.estimate);
    end
    
    function testEM_AR1_panel_mf_improve(testCase)
      % Test that the the EM always improved the likelihood
      % Note that this is the same as MFVAR
      y = testCase.data.panel;
      accum = Accumulator.GenerateRegular(y, {'avg'}, 3);

      dfmE = MFDFM(y, 1, 1, accum);
      testCase.verifyWarningFree(@dfmE.estimate);
    end
    
    function testEM_VAR2_accum(testCase)
      p = 10; 
      nFac = 1;
      nLags = 2;
      
      y = MFDFM_test.generateDFM(p, nFac, nLags, 250)';
      aggY = y;
      aggY(:, 2) = Accumulator_test.aggregateY(y(:, 2), 3, 'avg');
      accum = Accumulator.GenerateRegular(aggY, {'', 'avg'}, [1 3]);

      varE = MFDFM(aggY, nFac, nLags, accum);
      testCase.verifyWarningFree(@varE.estimate);
    end    
  end
  
  methods (Static)    
    function [y, ss] = generateDFM(p, nFac, nLags, n)
      % Generate a set of DFM parameters. 
      % Note that the system is not identified for estimation
      % 
      % Not intended for large systems (will be slow with many series)
      
      [~, ~, phi] = MFVAR_test.generateVAR(nFac, nLags, 1);
      phi2T = @(phi) [phi; eye(nFac*(nLags-1)) zeros(nFac*(nLags-1), nFac)];
      
      const = abs(randn(nFac,1));
      sigma = eye(nFac);
      
      loadings = [randn(p, nFac) zeros(p,nFac*(nLags-1))];
      measVar = 0.1 * eye(p);
      
      ss = StateSpace(loadings, measVar, ...
        phi2T(phi), sigma, 'c', [const; zeros(nFac*(nLags-1),1)], ...
        'R', [eye(nFac); zeros(nFac*(nLags-1),nFac)]);
      
      y = generateData(ss, n);
    end    
  end
end
