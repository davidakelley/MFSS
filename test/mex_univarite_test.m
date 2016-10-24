% Test univariate mex filter with an AR model.
% Assumes that the Matlab version of the univariate filter/smoother are
% correct.

% David Kelley, 2016

classdef mex_univarite_test < matlab.unittest.TestCase
  
  properties
    Y
    ssM
    ssMex
  end
  
  methods
    function ssOut = generateARmodel(testCase, p, m) %#ok<INUSL>
      g = 1;
      
      % Observation equation
      Z = rand(p, m);
      d = zeros(p, 1) * 0.3;
      H = diag(rand(p, 1));
      
      % State equation
      weightsAR = 2 * rand(1, m-1) - 1;
      T = [weightsAR ./ sum(weightsAR) * .9, 0; 
            eye(m-1), zeros(m-1, 1)];
      c = zeros(m, 1);
      R = [1; zeros(m-1, 1)];
      Q = diag(rand(g));
      
      ssOut = StateSpace(Z, d, H, T, c, R, Q, []);
    end
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      % Set up test
      timeDim = 600;
      g = 1; m = 2; p = 10; 
      ss = testCase.generateARmodel(p, m);
      
      % Generate data
      eta = ss.R * ss.Q^(1/2) * randn(g, timeDim);
      alpha = nan(m, timeDim);
      alpha(:,1) = eta(:,1);
      for iT = 2:timeDim
        alpha(:,iT) = ss.T * alpha(:,iT-1) + eta(:,iT);
      end
      
      epsilon = ss.H^(1/2) * randn(p, timeDim);
      testCase.Y = nan(p, timeDim);
      for iT = 1:timeDim
        testCase.Y(:,iT) = ss.Z * alpha(:,iT) + epsilon(:,iT);
      end
      
      testCase.ssMex = ss;
      testCase.ssM = testCase.ssMex;
      testCase.ssM.useMex = false;
    end
  end
  
  methods (Test)
    function testFilter(testCase)
      % Run filter
      [a_m, logl_m, fOut_m] = testCase.ssM.filter(testCase.Y);
      [a, logl, fOut] = testCase.ssMex.filter(testCase.Y);
      
      % Assertions
      allowedError = 1e-13;

      testCase.verifyEqual(logl, logl_m, 'RelTol', allowedError);
      testCase.verifyEqual(a, a_m, 'RelTol', allowedError);
      testCase.verifyEqual(fOut.P, fOut_m.P, 'RelTol', allowedError);
      testCase.verifyEqual(fOut.v, fOut_m.v, 'RelTol', allowedError);
      testCase.verifyEqual(fOut.F, fOut_m.F, 'RelTol', allowedError);
      testCase.verifyEqual(fOut.M, fOut_m.M, 'RelTol', allowedError);
      testCase.verifyEqual(fOut.K, fOut_m.K, 'RelTol', allowedError);
      testCase.verifyEqual(fOut.L, fOut_m.L, 'RelTol', allowedError);      
    end
    
    function testSmoother(testCase)
      % Run smoother
      [alpha_m, sOut_m] = testCase.ssM.smooth(testCase.Y);
      [alpha, sOut] = testCase.ssMex.smooth(testCase.Y);
      
      % Assertions
      allowedError = 1e-13;

      testCase.verifyEqual(alpha, alpha_m, 'RelTol', allowedError);
      testCase.verifyEqual(sOut.eta, sOut_m.eta, 'RelTol', allowedError);
      testCase.verifyEqual(sOut.r, sOut_m.r, 'RelTol', allowedError);
      testCase.verifyEqual(sOut.N, sOut_m.N, 'RelTol', allowedError);
      testCase.verifyEqual(sOut.V, sOut_m.V, 'RelTol', allowedError);
      testCase.verifyEqual(sOut.J, sOut_m.J, 'RelTol', allowedError);
      testCase.verifyEqual(sOut.logli, sOut_m.logli, 'RelTol', allowedError);
    end
    
    function testTiming(testCase)
      %% Timing
      filter_fn = @() testCase.ssM.filter(testCase.Y);
      mTime_filter = timeit(filter_fn, 3);
      
      filter_fn = @() testCase.ssMex.filter(testCase.Y);
      mexTime_filter = timeit(filter_fn, 3);
      
      smooth_fn = @() testCase.ssM.smooth(testCase.Y);
      mTime_smooth = timeit(smooth_fn, 2);
      
      smooth_fn = @() testCase.ssMex.smooth(testCase.Y);
      mexTime_smooth = timeit(smooth_fn, 2);
      
      fprintf('\nMex timing (%d observables, %d states, t = %d):\n', ...
        testCase.ssM.p, testCase.ssM.m, size(testCase.Y, 2));
      fprintf(' mex filter takes %3.2f%% of the time as the .m version.\n', ...
        mexTime_filter/mTime_filter*100);
      fprintf(' mex smoother takes %3.2f%% of the time as the .m version.\n', ...
        mexTime_smooth/mTime_smooth*100);
    end
  end
end