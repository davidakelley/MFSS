% Test univariate mex filter with an AR model.
% Assumes that the Matlab version of the univariate filter/smoother are
% correct.

% David Kelley, 2016

classdef mex_multivariate_test < matlab.unittest.TestCase
  
  properties
    Y
    model
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      baseDir = fileparts(fileparts(mfilename('fullpath')));
      addpath(baseDir);
      addpath(fullfile(baseDir, 'examples'));

      % Set up test
      ss = generateARmodel(10, 2, false);
      data = generateData(ss, 600);
      
      testCase.model = ss;
      testCase.Y = data;
    end
  end
  
  methods (Test)
    function testFilter(testCase)
      data = testCase.Y;
      ss = testCase.model;
      
      % Run filter
      ss.useMex(false);
      [a_m, logl_m, fOut_m] = ss.filter(data);
      ss.useMex(true);
      [a, logl, fOut] = ss.filter(data);
      
      % Assertions
      testCase.verifyEqual(logl, logl_m, 'AbsTol', 1e-9);
      testCase.verifyEqual(a, a_m, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.P, fOut_m.P, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.v, fOut_m.v, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.F, fOut_m.F, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.K, fOut_m.K, 'AbsTol', 1e-11);
    end
    
    function testFilterDiffuseInitial(testCase)
      data = testCase.Y;
      ss = testCase.model;
      
      ss = ss.setDefaultInitial();
      P0 = blkdiag(ss.P0(1:end-1, 1:end-1), Inf);
      ss.P0 = P0;
      
      % Run filter
      ss.useMex(false);
      [a_m, logl_m, fOut_m] = ss.filter(data);
      ss.useMex(true);
      [a, logl, fOut] = ss.filter(data);
      
      % Assertions
      testCase.verifyEqual(logl, logl_m, 'AbsTol', 1e-9);
      testCase.verifyEqual(a, a_m, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.P, fOut_m.P, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.Pd, fOut_m.Pd, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.v, fOut_m.v, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.F, fOut_m.F, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.Fd, fOut_m.Fd, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.K, fOut_m.K, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.Kd, fOut_m.Kd, 'AbsTol', 1e-11);
    end
    
    function testSmoother(testCase)
      data = testCase.Y;
      ss = testCase.model;
      
      % Run smoother
      ss.useMex(false);
      [alpha_m, sOut_m] = ss.smooth(data);
      ss.useMex(true);
      [alpha, sOut] = ss.smooth(data);
      
      % Assertions
      testCase.verifyEqual(alpha, alpha_m, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.eta, sOut_m.eta, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.r, sOut_m.r, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.N, sOut_m.N, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.a0tilde, sOut_m.a0tilde, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.logli, sOut_m.logli, 'AbsTol', 1e-9);
    end
    
    function testSmootherDiffuse(testCase)

      ss = generateARmodel(2, 1, false);
      ss.Z = eye(2);
      ss.T = [1 0.01; 0 0.95];
       
      y = generateData(ss, 500);
      y(1,1:100) = nan;
      
      % Run smoother
      ss.useMex(false);
      [alpha_m, sOut_m, fOut_m] = ss.smooth(y);
      ss.useMex(true);
      [alpha, sOut, fOut] = ss.smooth(y);
      
      % Assertions
      testCase.verifyEqual(fOut_m.dt, fOut.dt);
      testCase.verifyEqual(fOut_m.a, fOut.a);
      testCase.verifyEqual(fOut_m.P, fOut.P);
      testCase.verifyEqual(fOut_m.v, fOut.v);
      testCase.verifyEqual(fOut_m.K, fOut.K);
      testCase.verifyEqual(fOut_m.Kd, fOut.Kd);
      
      testCase.verifyEqual(alpha, alpha_m, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.eta, sOut_m.eta, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.r, sOut_m.r, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.N, sOut_m.N, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.V, sOut_m.V, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.a0tilde, sOut_m.a0tilde, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.logli, sOut_m.logli, 'AbsTol', 1e-9);
    end
    
    function testTiming(testCase)
      data = testCase.Y;
      ss = testCase.model;
      
      %% Timing
      ss.useMex = false;
      filter_fn = @() ss.filter(data);
      mTime_filter = timeit(filter_fn, 3);
      
      ss.useMex = true;
      filter_fn = @() ss.filter(data);
      mexTime_filter = timeit(filter_fn, 3);
      
      ss.useMex = false;
      smooth_fn = @() ss.smooth(data);
      mTime_smooth = timeit(smooth_fn, 2);
      
      ss.useMex = true;
      smooth_fn = @() ss.smooth(data);
      mexTime_smooth = timeit(smooth_fn, 2);

      fprintf('\nMex timing (%d observables, %d states, t = %d):\n', ...
        ss.p, ss.m, size(data, 2));
      fprintf(' mex filter takes %3.2f%% of the time as the .m version.\n', ...
        mexTime_filter/mTime_filter*100);
      fprintf(' mex smoother takes %3.2f%% of the time as the .m version.\n', ...
        mexTime_smooth/mTime_smooth*100);
    end
  end
end