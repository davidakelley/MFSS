% Test univariate mex filter with an AR model.
% Assumes that the Matlab version of the univariate filter/smoother are
% correct.

% David Kelley, 2016

classdef mex_multivariate_test < matlab.unittest.TestCase
  
  properties
    Y
    ss
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      baseDir =  [subsref(strsplit(mfilename('fullpath'), 'MFSS'), ...
        struct('type', '{}', 'subs', {{1}})) 'MFSS'];
      addpath(baseDir);
      addpath(fullfile(baseDir, 'examples'));

      % Set up test
      testCase.ss = generateARmodel(10, 2, false);
      testCase.Y = generateData(testCase.ss, 600);
    end
  end
  
  methods (Test)
    function testFilter(testCase)
      % Run filter
      testCase.ss.useMex(false);
      [a_m, logl_m, fOut_m] = testCase.ss.filter(testCase.Y);
      testCase.ss.useMex(true);
      [a, logl, fOut] = testCase.ss.filter(testCase.Y);
      
      % Assertions
      testCase.verifyEqual(logl, logl_m, 'AbsTol', 1e-11);
      testCase.verifyEqual(a, a_m, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.P, fOut_m.P, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.v, fOut_m.v, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.F, fOut_m.F, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.M, fOut_m.M, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.K, fOut_m.K, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.L, fOut_m.L, 'AbsTol', 1e-11);
    end
    
    function testSmoother(testCase)
      % Run smoother
      testCase.ss.useMex(false);
      [alpha_m, sOut_m] = testCase.ss.smooth(testCase.Y);
      testCase.ss.useMex(true);
      [alpha, sOut] = testCase.ss.smooth(testCase.Y);
      
      % Assertions
      testCase.verifyEqual(alpha, alpha_m, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.eta, sOut_m.eta, 'AbsTol', 1e-11);
%       testCase.verifyEqual(sOut.epsilon, sOut_m.epsilon, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.r, sOut_m.r, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.N, sOut_m.N, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.a0tilde, sOut_m.a0tilde, 'AbsTol', 1e-11);
%       testCase.verifyEqual(sOut.V, sOut_m.V, 'AbsTol', 1e-11);
%       testCase.verifyEqual(sOut.J, sOut_m.J, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.logli, sOut_m.logli, 'AbsTol', 1e-11);
    end
    
    function testTiming(testCase)
      %% Timing
      testCase.ss.useMex(false);
      filter_fn = @() testCase.ss.filter(testCase.Y);
      mTime_filter = timeit(filter_fn, 3);
      
      testCase.ss.useMex(true);
      filter_fn = @() testCase.ss.filter(testCase.Y);
      mexTime_filter = timeit(filter_fn, 3);
      
      testCase.ss.useMex(false);
      smooth_fn = @() testCase.ss.smooth(testCase.Y);
      mTime_smooth = timeit(smooth_fn, 2);
      
      testCase.ss.useMex(true);
      smooth_fn = @() testCase.ss.smooth(testCase.Y);
      mexTime_smooth = timeit(smooth_fn, 2);

      fprintf('\nMex timing (%d observables, %d states, t = %d):\n', ...
        testCase.ss.p, testCase.ss.m, size(testCase.Y, 2));
      fprintf(' mex filter takes %3.2f%% of the time as the .m version.\n', ...
        mexTime_filter/mTime_filter*100);
      fprintf(' mex smoother takes %3.2f%% of the time as the .m version.\n', ...
        mexTime_smooth/mTime_smooth*100);
    end
  end
end