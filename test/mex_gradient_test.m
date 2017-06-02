% Test mex gradient function .
% Assumes that the Matlab version of the univariate filter/smoother are
% correct.

% David Kelley, 2016

classdef mex_gradient_test < matlab.unittest.TestCase
  methods(TestClassSetup)
    function setupOnce(testCase) %#ok<MANU>
      baseDir = fileparts(fileparts(mfilename('fullpath')));
      addpath(baseDir);
      addpath(fullfile(baseDir, 'examples'));
    end
  end
  
  methods (Test)
    function testGradSmallUni(testCase)
      ss0 = generateARmodel(2, 2, true);
      data = generateData(ss0, 600);
      tm = ThetaMap.ThetaMapAll(ss0);
      
      % Run gradient
      ss0.useMex(false);
      [logl_m, grad_m] = ss0.gradient(data, tm);
      ss0.useMex(true);
      [logl, grad] = ss0.gradient(data, tm);
      
      % Assertions
      testCase.verifyEqual(logl, logl_m, 'AbsTol', 1e-11);
      testCase.verifyEqual(grad, grad_m, 'AbsTol', 1e-11);
    end
    
    function testGradLargeMulti(testCase)
      ss0 = generateARmodel(15, 2, false);
      data = generateData(ss0, 600);
      tm = ThetaMap.ThetaMapAll(ss0);
      
      % Run gradient
      ss0.useMex(false);
      [logl_m, grad_m] = ss0.gradient(data, tm);
      ss0.useMex(true);
      [logl, grad] = ss0.gradient(data, tm);
      
      % Assertions
      testCase.verifyEqual(logl, logl_m, 'AbsTol', 2e-11);
      testCase.verifyEqual(grad, grad_m, 'AbsTol', 5e-11);
    end
    
    function testTiming(testCase) %#ok<MANU>
      %% Timing
      ss = generateARmodel(10, 2, false);
      data = generateData(ss, 600);
      tm = ThetaMap.ThetaMapAll(ss);
            
      ss.useMex(false);
      filter_fn = @() ss.gradient(data, tm);
      mTime_filter = timeit(filter_fn, 2);
      
      ss.useMex(true);
      filter_fn = @() ss.gradient(data, tm);
      mexTime_filter = timeit(filter_fn, 2);

      fprintf('\nMex timing (%d observables, %d states, t = %d):\n', ...
        ss.p, ss.m, size(data, 2));
      fprintf(' mex gradient takes %3.2f%% of the time as the .m version.\n', ...
        mexTime_filter/mTime_filter*100);
    end
  end
end