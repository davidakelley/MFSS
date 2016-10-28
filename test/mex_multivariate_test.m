% Test univariate mex filter with an AR model.
% Assumes that the Matlab version of the univariate filter/smoother are
% correct.

% David Kelley, 2016

classdef mex_multivariate_test < matlab.unittest.TestCase
  
  properties
    Y
    ssM
    ssMex
    allowedError = 1e-11;
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      addpath('C:\Users\g1dak02\Documents\MATLAB\StateSpace');

      % Set up test
      ss = generateARmodel(10, 2, false);
      
      testCase.Y = generateData(ss, 600);
      
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
      testCase.verifyEqual(logl, logl_m, 'AbsTol', testCase.allowedError);
      testCase.verifyEqual(a, a_m, 'AbsTol', testCase.allowedError);
      testCase.verifyEqual(fOut.P, fOut_m.P, 'AbsTol', testCase.allowedError);
      testCase.verifyEqual(fOut.v, fOut_m.v, 'AbsTol', testCase.allowedError);
      testCase.verifyEqual(fOut.F, fOut_m.F, 'AbsTol', testCase.allowedError);
      testCase.verifyEqual(fOut.M, fOut_m.M, 'AbsTol', testCase.allowedError);
      testCase.verifyEqual(fOut.K, fOut_m.K, 'AbsTol', testCase.allowedError);
      testCase.verifyEqual(fOut.L, fOut_m.L, 'AbsTol', testCase.allowedError);      
      testCase.verifyEqual(fOut.w, fOut_m.w, 'AbsTol', testCase.allowedError);      
      testCase.verifyEqual(fOut.Finv, fOut_m.Finv, 'AbsTol', testCase.allowedError);      
    end
    
    function testSmoother(testCase)
      % Run smoother
      [alpha_m, sOut_m] = testCase.ssM.smooth(testCase.Y);
      [alpha, sOut] = testCase.ssMex.smooth(testCase.Y);
      
      % Assertions
      testCase.verifyEqual(alpha, alpha_m, 'AbsTol', testCase.allowedError);
      testCase.verifyEqual(sOut.eta, sOut_m.eta, 'AbsTol', testCase.allowedError);
      testCase.verifyEqual(sOut.epsilon, sOut_m.epsilon, 'AbsTol', testCase.allowedError);
      testCase.verifyEqual(sOut.r, sOut_m.r, 'AbsTol', testCase.allowedError);
      testCase.verifyEqual(sOut.N, sOut_m.N, 'AbsTol', testCase.allowedError);
      testCase.verifyEqual(sOut.a0tilde, sOut_m.a0tilde, 'AbsTol', testCase.allowedError);
      testCase.verifyEqual(sOut.V, sOut_m.V, 'AbsTol', testCase.allowedError);
      testCase.verifyEqual(sOut.J, sOut_m.J, 'AbsTol', testCase.allowedError);
      testCase.verifyEqual(sOut.logli, sOut_m.logli, 'AbsTol', testCase.allowedError);
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