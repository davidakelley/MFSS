% Test univariate mex filter with an AR model.
% Assumes that the Matlab version of the univariate filter/smoother are
% correct.

% David Kelley, 2016

classdef kalman_test < matlab.unittest.TestCase
  
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
      ss = generateARmodel(10, 2, true);
      data = generateData(ss, 600);
      
      testCase.model = ss;
      testCase.Y = data;
    end
  end
  
  methods (Test)
    function testFiniteFilter(testCase)
      ss = generateARmodel(10, 2, true);
      data = generateData(ss, 600);
      
      % Run filter
      [a, logl] = ss.filter(data);
      
      testCase.verifyThat(a, matlab.unittest.constraints.IsFinite);
      testCase.verifyThat(logl, matlab.unittest.constraints.IsFinite);
    end
    
    function testFiniteSmoother(testCase)
      ss = generateARmodel(10, 2, true);
      data = generateData(ss, 600);
      
      % Run filter
      alpha = ss.smooth(data);
      testCase.verifyThat(alpha, matlab.unittest.constraints.IsFinite);
    end
    
    function testNoObsErrSmoother(testCase)
      ss = generateARmodel(1, 2, true);
      ss.H(:) = 0;
      
      data = generateData(ss, 600);
      
      % Run smoother
      alpha = ss.smooth(data);
      testCase.verifyEqual(alpha(1,:), data, 'AbsTol', 1e-15);
    end  
    
    function testNoObsErrSmootherExogenous(testCase)
      ssAR = generateARmodel(1, 2, true);
      ss = StateSpace(ssAR.Z, 0, ssAR.T, ssAR.Q, 'R', ssAR.R, 'beta', .5);
     
      [data, ~, x] = generateData(ss, 600);
      
      % Run smoother with exogenous data
      alpha = ss.smooth(data, x);
      testCase.verifyEqual(alpha(1,:), data - ss.beta * x, 'AbsTol', 1e-15);
    end 
    
    %% Test mex v. Matlab code
    function testUniFilter(testCase)
      ss = generateARmodel(10, 2, true);
      data = generateData(ss, 600);
      
      % Run filter
      ss.useMex = false;
      [a_m, logl_m, fOut_m] = ss.filter(data);
      ss.useMex = true;
      [a, logl, fOut] = ss.filter(data);
      
      % Assertions
      testCase.verifyEqual(logl, logl_m, 'AbsTol', 1e-9);
      testCase.verifyEqual(a, a_m, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.P, fOut_m.P, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.v, fOut_m.v, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.F, fOut_m.F, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.K, fOut_m.K, 'AbsTol', 1e-11);
    end
    
    function testUniSmoother(testCase)
      ss = generateARmodel(10, 2, true);
      data = generateData(ss, 600);
      
      % Run smoother
      ss.useMex = false;
      [alpha_m, sOut_m] = ss.smooth(data);
      ss.useMex = true;
      [alpha, sOut] = ss.smooth(data);
      
      % Assertions
      testCase.verifyEqual(alpha, alpha_m, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.eta, sOut_m.eta, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.r, sOut_m.r, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.N, sOut_m.N, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.logli, sOut_m.logli, 'AbsTol', 1e-9);
    end
    
    function testMultiFilter(testCase)
      ss = generateARmodel(10, 2, false);
      data = generateData(ss, 600);
      
      % Run filter
      ss.useMex = false;
      [a_m, logl_m, fOut_m] = ss.filter(data);
      ss.useMex = true;
      [a, logl, fOut] = ss.filter(data);
      
      % Assertions
      testCase.verifyEqual(logl, logl_m, 'AbsTol', 1e-9);
      testCase.verifyEqual(a, a_m, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.P, fOut_m.P, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.v, fOut_m.v, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.F, fOut_m.F, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.K, fOut_m.K, 'AbsTol', 1e-11);
    end
    
    function testMultiFilterDiffuseInitial(testCase)
      ss = generateARmodel(10, 2, false);
      data = generateData(ss, 600);
      
      ss = ss.setDefaultInitial();
      P0 = blkdiag(ss.P0(1:end-1, 1:end-1), Inf);
      ss.P0 = P0;
      
      % Run filter
      ss.useMex = false;
      [a_m, logl_m, fOut_m] = ss.filter(data);
      ss.useMex = true;
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
    
    function testMultiSmoother(testCase)
      ss = generateARmodel(10, 2, false);
      data = generateData(ss, 600);
      
      % Run smoother
      ss.useMex = false;
      [alpha_m, sOut_m] = ss.smooth(data);
      ss.useMex = true;
      [alpha, sOut] = ss.smooth(data);
      
      % Assertions
      testCase.verifyEqual(alpha, alpha_m, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.eta, sOut_m.eta, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.r, sOut_m.r, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.N, sOut_m.N, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.a0tilde, sOut_m.a0tilde, 'AbsTol', 1e-11);
      testCase.verifyEqual(sOut.logli, sOut_m.logli, 'AbsTol', 1e-9);
    end
    
    function testMultiSmootherDiffuse(testCase)

      ss = generateARmodel(2, 1, false);
      ss.Z = eye(2);
      ss.T = [1 0.01; 0 0.95];
       
      y = generateData(ss, 500);
      y(1,1:100) = nan;
      
      % Run smoother
      ss.useMex = false;
      [alpha_m, sOut_m, fOut_m] = ss.smooth(y);
      ss.useMex = true;
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
    
    function testExogFilter(testCase)
      ssAR = generateARmodel(2, 1, false);
      ssAR.Z = eye(2);
      ssAR.T = [1 0.01; 0 0.95];
       
      ss = StateSpace(ssAR.Z, ssAR.H, ssAR.T, ssAR.Q, 'R', ssAR.R, 'beta', [1; .2]);
      
      [y, ~, x] = generateData(ss, 500);
      
      % Run filter
      ss.useMex = false;
      [a_m, logl_m, fOut_m] = ss.filter(y, x);
      ss.useMex = true;
      [a, logl, fOut] = ss.filter(y, x);
      
      % Assertions
      testCase.verifyEqual(logl, logl_m, 'AbsTol', 1e-9);
      testCase.verifyEqual(a, a_m, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.P, fOut_m.P, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.v, fOut_m.v, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.F, fOut_m.F, 'AbsTol', 1e-11);
      testCase.verifyEqual(fOut.K, fOut_m.K, 'AbsTol', 1e-11);
    end
    
    function testExogSmoother(testCase)
      ssAR = generateARmodel(2, 1, false);
      ssAR.Z = eye(2);
      ssAR.T = [1 0.01; 0 0.95];
       
      ss = StateSpace(ssAR.Z, ssAR.H, ssAR.T, ssAR.Q, 'R', ssAR.R, 'beta', [1, .3; .2 1]);
      
      [y, ~, x] = generateData(ss, 500);
      
      % Run smoother
      ss.useMex = false;
      [alpha_m, sOut_m, fOut_m] = ss.smooth(y, x);
      ss.useMex = true;
      [alpha, sOut, fOut] = ss.smooth(y, x);
      
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
    
    function testExogFilterBetadEquiv(testCase)
      ssAR = generateARmodel(2, 1, false);
      ssAR.Z = eye(2);
      ssAR.T = [1 0.01; 0 0.95];
       
      ss = StateSpace(ssAR.Z, ssAR.H, ssAR.T, ssAR.Q, 'R', ssAR.R, 'beta', [1; .2]);
      
      N = 500;
      [y, ~, x] = generateData(ss, N);
      [a, logl, fOut] = ss.filter(y, x);
      
      % Run same model but putting the beta*x part in d.
      ss2 = StateSpace(struct('Zt', ssAR.Z, 'tauZ', ones(N,1)), ...
        struct('Ht', ssAR.H, 'tauH', ones(N,1)), ...
        struct('Tt', ssAR.T, 'tauT', ones(N+1,1)), ...
        struct('Qt', ssAR.Q, 'tauQ', ones(N+1,1)), ...
        'R', struct('Rt', ssAR.R, 'tauR', ones(N+1,1)), ...
        'd', struct('dt', [1; .2] * x, 'taud', (1:N)'));
      [a_2, logl_2, fOut_2] = ss2.filter(y);
      
      % Assertions
      testCase.verifyEqual(logl, logl_2, 'AbsTol', 1e-12);
      testCase.verifyEqual(a, a_2, 'AbsTol', 1e-12);
      testCase.verifyEqual(fOut.P, fOut_2.P, 'AbsTol', 1e-12);
      testCase.verifyEqual(fOut.v, fOut_2.v, 'AbsTol', 1e-12);
      testCase.verifyEqual(fOut.F, fOut_2.F, 'AbsTol', 1e-12);
      testCase.verifyEqual(fOut.K, fOut_2.K, 'AbsTol', 1e-12);
    end
    
    function testExogSmootherBetadEquiv(testCase)
      ssAR = generateARmodel(2, 1, false);
      ssAR.Z = eye(2);
      ssAR.T = [1 0.01; 0 0.95];
       
      ss = StateSpace(ssAR.Z, ssAR.H, ssAR.T, ssAR.Q, 'R', ssAR.R, 'beta', [1; .2]);
      
      N = 500;
      [y, ~, x] = generateData(ss, N);
      [alpha, sOut, fOut] = ss.smooth(y, x);

      % Run same model but putting the beta*x part in d.
      ss2 = StateSpace(struct('Zt', ssAR.Z, 'tauZ', ones(N,1)), ...
        struct('Ht', ssAR.H, 'tauH', ones(N,1)), ...
        struct('Tt', ssAR.T, 'tauT', ones(N+1,1)), ...
        struct('Qt', ssAR.Q, 'tauQ', ones(N+1,1)), ...
        'R', struct('Rt', ssAR.R, 'tauR', ones(N+1,1)), ...
        'd', struct('dt', [1; .2] * x, 'taud', (1:N)'));
      [alpha_2, sOut_2, fOut_2] = ss2.smooth(y);
      
      % Assertions
      testCase.verifyEqual(fOut_2.dt, fOut.dt);
      testCase.verifyEqual(fOut_2.a, fOut.a, 'AbsTol', 1e-13);
      testCase.verifyEqual(fOut_2.P, fOut.P, 'AbsTol', 1e-13);
      testCase.verifyEqual(fOut_2.v, fOut.v, 'AbsTol', 1e-13);
      testCase.verifyEqual(fOut_2.K, fOut.K, 'AbsTol', 1e-13);
      testCase.verifyEqual(fOut_2.Kd, fOut.Kd, 'AbsTol', 1e-13);
      
      testCase.verifyEqual(alpha, alpha_2, 'AbsTol', 1e-13);
      testCase.verifyEqual(sOut.eta, sOut_2.eta, 'AbsTol', 1e-13);
      testCase.verifyEqual(sOut.r, sOut_2.r, 'AbsTol', 1e-13);
      testCase.verifyEqual(sOut.N, sOut_2.N, 'AbsTol', 1e-13);
      testCase.verifyEqual(sOut.V, sOut_2.V, 'AbsTol', 1e-13);
      testCase.verifyEqual(sOut.a0tilde, sOut_2.a0tilde, 'AbsTol', 1e-13);
      testCase.verifyEqual(sOut.logli, sOut_2.logli, 'AbsTol', 1e-4);
    end
    
    %% Timing
    function testTiming(testCase)
      ss = generateARmodel(10, 2, false);
      data = generateData(ss, 600);
      
      % Timing
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
      
      testCase.verifyLessThan(mexTime_filter, mTime_filter);
      testCase.verifyLessThan(mexTime_smooth, mTime_smooth);
    end
  end
end