%% Tests for StateSpace.gradient
%
% Run with
%{
  testOut = runtests('test/gradient_test.m');
%}

% David Kelley, 2016

classdef gradient_test < matlab.unittest.TestCase
  
  properties
    bbk
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      baseDir =  [subsref(strsplit(mfilename('fullpath'), 'MFSS'), ...
        struct('type', '{}', 'subs', {{1}})) 'MFSS'];
      addpath(baseDir);
      addpath(fullfile(baseDir, 'examples'));
      
      testCase.bbk = load(fullfile(baseDir, 'examples', 'data', 'bbk_data.mat'));
      y = testCase.bbk.data.indicators';
      y(:, any(isnan(y), 1)) = [];
      testCase.bbk.y = y;
    end
  end
  
  methods (Test)
    function testGeneratedLLM(testCase)
      p = 1; m = 0; timeDim = 500;
      ss = generateARmodel(p, m, false);
      ss.T = 1;
      y = generateData(ss, timeDim);
      
      tm = ThetaMap.ThetaMapAll(ss);
      
      tic;
      [~, analytic] = ss.gradient(y, tm);
      time_a = toc;
      tic;
      numeric = numericGradient(ss, tm, y, 1e-8);
      time_n = toc;
      fprintf(['\nModel: %d series, %d states, t = %d, nTheta = %d\n' ...
        'Analytic gradient took %3.2f%% of the time as the numeric version.\n'],...
        p, m, timeDim, tm.nTheta, 100*(time_a/time_n));
      
      testCase.verifyEqual(numeric, analytic, 'AbsTol', 3e-3, 'RelTol', 1e-4);
    end
    
    function testGeneratedLLMexplicitInitial(testCase)
      p = 1; m = 2; timeDim = 500;
      ss = generateARmodel(p, m, false);
      y = generateData(ss, timeDim);
      
      ss = ss.checkSample(y);
      ss = ss.setDefaultInitial();
      ss.a0 = [y(1); 0; 0];
      ss.usingDefaulta0 = false;
      ss.usingDefaultP0 = false;
      
      tm = ThetaMap.ThetaMapAll(ss);
      
      tic;
      [~, analytic] = ss.gradient(y, tm);
      time_a = toc;
      tic;
      numeric = numericGradient(ss, tm, y, 1e-8);
      time_n = toc;
      fprintf(['\nModel: %d series, %d states, t = %d, nTheta = %d\n' ...
        'Analytic gradient took %3.2f%% of the time as the numeric version.\n'],...
        p, m, timeDim, tm.nTheta, 100*(time_a/time_n));
      
      testCase.verifyEqual(numeric, analytic, 'AbsTol', 4e-5, 'RelTol', 1e-5);
    end
    
    function testGeneratedSmallDefaultInitial(testCase)
      p = 2; m = 1; timeDim = 500;
      ss = generateARmodel(p, m, false);
      y = generateData(ss, timeDim);
      
      ss = ss.checkSample(y);
      tm = ThetaMap.ThetaMapAll(ss);
      
      tic;
      [~, analytic] = ss.gradient(y, tm);
      time_a = toc;
      tic;
      numeric = numericGradient(ss, tm, y, 1e-8);
      time_n = toc;
      
      fprintf(['\nModel: %d series, %d states, t = %d, nTheta = %d\n' ...
        'Analytic gradient took %3.2f%% of the time as the numeric version.\n'],...
        p, m, timeDim, tm.nTheta, 100*(time_a/time_n));
      
      testCase.verifyEqual(analytic, numeric, 'AbsTol', 3e-3, 'RelTol', 1e-4);
    end
    
    function testGeneratedMedium(testCase)
      p = 4; m = 4; timeDim = 500;
      ss = generateARmodel(p, m, false);
      y = generateData(ss, timeDim);
      
      tm = ThetaMap.ThetaMapAll(ss);
      
      tic;
      [~, analytic] = ss.gradient(y, tm);
      time_a = toc;
      tic;
      numeric = numericGradient(ss, tm, y, 1e-8);
      time_n = toc;

      fprintf(['\nModel: %d series, %d states, t = %d, nTheta = %d\n' ...
        'Analytic gradient took %3.2f%% of the time as the numeric version.\n'],...
        p, m, timeDim, tm.nTheta, 100*(time_a/time_n));
      testCase.verifyEqual(analytic, numeric, 'AbsTol', 3e-3, 'RelTol', 1e-4);
    end
    
    function testGeneratedMediumExplicitIntial(testCase)
      p = 4; m = 4; timeDim = 500;
      ss = generateARmodel(p, m, false);
      y = generateData(ss, timeDim);
      
      ss = ss.checkSample(y);
      ss = ss.setDefaultInitial();
      ss.usingDefaulta0 = false;
      ss.usingDefaultP0 = false;
      
      tm = ThetaMap.ThetaMapAll(ss);
      
      tic;
      [~, analytic] = ss.gradient(y, tm);
      time_a = toc;
      tic;
      numeric = numericGradient(ss, tm, y, 1e-8);
      time_n = toc;

      fprintf(['\nModel: %d series, %d states, t = %d, nTheta = %d\n' ...
        'Analytic gradient took %3.2f%% of the time as the numeric version.\n'],...
        p, m, timeDim, tm.nTheta, 100*(time_a/time_n));
      testCase.verifyEqual(analytic, numeric, 'AbsTol', 3e-3, 'RelTol', 1e-4);
    end
    
    function testGeneratedMediumDiffuseInitial(testCase)
      p = 4; m = 0; timeDim = 500;
      ss = generateARmodel(p, m, true);
      ss.T(1) = 1;
      y = generateData(ss, timeDim);
      
      ss = ss.checkSample(y);
      
      tm = ThetaMap.ThetaMapAll(ss);
      
      tic;
      [~, analytic] = ss.gradient(y, tm);
      time_a = toc;
      tic;
      numeric = numericGradient(ss, tm, y, 1e-8);
      time_n = toc;

      fprintf(['\nModel: %d series, %d states, t = %d, nTheta = %d\n' ...
        'Analytic gradient took %3.2f%% of the time as the numeric version.\n'],...
        p, m, timeDim, tm.nTheta, 100*(time_a/time_n));
      testCase.verifyEqual(analytic, numeric, 'AbsTol', 3e-3, 'RelTol', 4e-4);
    end
    
    function testGeneratedLarge(testCase)
      p = 20; m = 2; timeDim = 500;
      ss = generateARmodel(p, m, false);
      y = generateData(ss, timeDim);
      
      tm = ThetaMap.ThetaMapAll(ss);
      
      tic;
      [~, analytic] = ss.gradient(y, tm);
      time_a = toc;
      tic;
      numeric = numericGradient(ss, tm, y, 1e-8);
      time_n = toc;
      
      fprintf(['\nModel: %d series, %d states, t = %d, nTheta = %d\n' ...
        'Analytic gradient took %3.2f%% of the time as the numeric version.\n'],...
        p, m, timeDim, tm.nTheta, 100*(time_a/time_n));
      testCase.verifyEqual(analytic, numeric, 'AbsTol', 3e-3, 'RelTol', 1e-4);
    end
    
    function testFactorModel(testCase)
      % Set up state
      rnfacs = 2;
      nSeries = testCase.bbk.dims.nSeries;
      nlags = testCase.bbk.dims.nlags;
      
      Z = [nan(nSeries, rnfacs) zeros(nSeries, rnfacs * (nlags-1))];
      d = zeros(nSeries, 1);
      H = eye(nSeries);
      H(H == 1) = nan;
      
      T = [nan(rnfacs, rnfacs * nlags);
        eye(rnfacs * (nlags-1)) zeros(rnfacs)];
      c = zeros(rnfacs * nlags, 1);
      R = [eye(rnfacs); zeros(rnfacs * (nlags-1), rnfacs)];
      Q = nan(rnfacs);
      
      ss = StateSpaceEstimation(Z, d, H, T, c, R, Q);
      tm = ThetaMap.ThetaMapEstimation(ss);
          
      % Initial values
      [Z(:, 1:rnfacs), f0] = pca(testCase.bbk.y', 'NumComponents', rnfacs);
      f0(any(isnan(f0), 2), :) = [];
      
      H = diag(var(testCase.bbk.y' - f0 * Z(:, 1:rnfacs)'));
      
      y_var = f0(nlags+1:end, :);
      assert(nlags == 2);
      x = [f0(2:end-1, :) f0(1:end-2, :)];
      
      yTx = y_var' * x;
      xTx = x' * x;
      yTy = y_var' * y_var;
      
      T(1:rnfacs, :) = yTx / xTx;
      Q = (yTy - yTx / xTx * yTx') ./ size(testCase.bbk.y, 1);

      % Set up initial values
      ss0 = StateSpace(Z, d, H, T, c, R, Q);
      
      % Test
      tic;
      [~, analytic] = ss0.gradient(testCase.bbk.y, tm);
      time_a = toc;
      tic;
      numeric = numericGradient(ss0, tm, testCase.bbk.y, 1e-8);
      time_n = toc;

      fprintf(['\nModel: %d series, %d states, t = %d (diagonal H), nTheta = %d\n' ...
        'Analytic gradient took %3.2f%% of the time as the numeric version.\n'],...
        nSeries, rnfacs, length(testCase.bbk.y), tm.nTheta, 100*(time_a/time_n));
      testCase.verifyEqual(analytic, numeric, 'AbsTol', 4e-3, 'RelTol', 1e-3);
    end
  end

end
