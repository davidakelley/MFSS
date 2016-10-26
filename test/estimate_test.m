% Test univariate mex filter with an AR model.
% Assumes that the Matlab version of the univariate filter/smoother are
% correct.

% David Kelley, 2016

classdef estimate_test < matlab.unittest.TestCase
  
  properties
    data = struct;
    tol_DK = 1e-2;    % Test v. Drubin-Koopman
    tol_grad = 1e-5;   % Tets against gradient version
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      addpath('C:\Users\g1dak02\Documents\MATLAB\StateSpace');
      
      % Load data
      testDir = [subsref(strsplit(pwd, 'StateSpace'), ...
        struct('type', '{}', 'subs', {{1}})) 'StateSpace\test\data'];
      dataStr = fileread(fullfile(testDir, 'Nile.dat'));
      lineBreaks = strfind(dataStr, sprintf('\n'));
      dataStr(1:lineBreaks(1)) = [];
      testCase.data.nile = sscanf(dataStr, '%d');
    end
  end
  
  methods (Test)
    
    function testNile(testCase)
      Z = 1;
      d = 0;
      H = nan;
      T = 1;
      c = 0;
      R = 1;
      Q = nan;
      
      ss = StateSpace(Z, d, H, T, c, R, Q, []);
      ss0 = ss;
      ss0.H = 1000;
      ss0.Q = 1000;
      
      ssE = ss.estimate(testCase.data.nile', ss0);
      
      % Using values from Dubrin & Koopman (2012), p. 37
      testCase.verifyEqual(ssE.H, 15099, 'RelTol', testCase.tol_DK);
      testCase.verifyEqual(ssE.Q, 1469.1, 'RelTol', testCase.tol_DK);
    end
    
    function testNileGradient(testCase)
      Z = 1;
      d = 0;
      H = nan;
      T = 1;
      c = 0;
      R = 1;
      Q = nan;
      
      ss = StateSpace(Z, d, H, T, c, R, Q, []);
      ss0 = ss;
      ss0.H = 1000;
      ss0.Q = 1000;
      
      ss.useGrad = true;
      ssE = ss.estimate(testCase.data.nile', ss0);
      ss.useGrad = false;
      ssE_ng = ss.estimate(testCase.data.nile', ss0);
      
      % Using values from Dubrin & Koopman (2012), p. 37
      testCase.verifyEqual(ssE.H, ssE_ng.H, 'RelTol', testCase.tol_grad);
      testCase.verifyEqual(ssE.Q, ssE_ng.Q, 'RelTol',  testCase.tol_grad);
    end
  end
end