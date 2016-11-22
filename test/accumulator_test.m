% Test Harvey accumulators for filter/smoother

% David Kelley, 2016

classdef accumulator_test < matlab.unittest.TestCase
  
  properties
    bbk
    deai
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      % Factor model data
      testCase.bbk = struct;
      [testCase.bbk.data, testCase.bbk.opts, testCase.bbk.dims] = loadFactorModel();
      
      baseDir =  [subsref(strsplit(mfilename('fullpath'), 'StateSpace'), ...
        struct('type', '{}', 'subs', {{1}})) 'StateSpace'];
      testCase.deai = load(fullfile(baseDir, 'test', 'data', 'deai.mat'));

      addpath('C:\Users\g1dak02\Documents\MATLAB\StateSpace');
    end
  end
  
  methods (Test)
    
    function testNoAccumWithMissing(testCase)
      % Run smoother over a dataset with missing observations, check that
      % its close to a dataset without missing values. 
      p = 10; m = 2; timeDim = 500;
      ss = generateARmodel(p, m, false);
      Y = generateData(ss, timeDim);
      
      missingMask = logical(randi([0 1], p, timeDim));
      missingMask(:, sum(missingMask, 1) > 4) = 0;
      obsY = Y;
      obsY(missingMask) = nan;
    
      alpha = ss.smooth(Y);
      obsAlpha = ss.smooth(obsY);
      
      allowedDiffPeriods = sum(sum(missingMask) > 0);
      diffPeriods = sum((abs(alpha(1, :)' - obsAlpha(1,:)')) > 0.02);
      testCase.verifyLessThanOrEqual(diffPeriods, allowedDiffPeriods);
    end
    
    function testSumAccumulatorSmoother(testCase)
      p = 2; m = 2; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      latentY = generateData(ssGen, timeDim);
      
      timeGroups = sort(repmat((1:ceil(timeDim/3))', [3 1]));
      timeGroups(end, :) = [];
      
      aggSeries = logical([0 1]);
      aggY = grpstats(latentY(aggSeries, :)', timeGroups, 'mean')';
      aggY(:, end) = [];
      Y = latentY;
      Y(aggSeries, :) = nan;
      Y(aggSeries, 3:3:end) = aggY;
      
      accum = struct;
      accum.xi = aggSeries;
      accum.psi = repmat([1 2 3]', [(timeDim+1)/3, sum(aggSeries)])';
      accum.Horizon = repmat(3, [timeDim+1, sum(aggSeries)])';
      
      ss = StateSpace(ssGen.Z, ssGen.d, ssGen.H, ...
        ssGen.T, ssGen.c, ssGen.R, ssGen.Q, accum);
      
      alpha = ss.smooth(Y);
      latentAlpha = ssGen.smooth(latentY);
      
      testCase.verifyEqual(alpha(1,:), latentAlpha(1,:), 'AbsTol', 0.5);      
    end
    
    function testDetroit(testCase)
      import matlab.unittest.constraints.IsFinite;
      
      ss0 = StateSpace(testCase.deai.Z, testCase.deai.d, testCase.deai.H, ...
        testCase.deai.T, testCase.deai.c, testCase.deai.R, testCase.deai.Q, ...
        testCase.deai.Harvey);
      [~, ll] = ss0.filter(testCase.deai.Y);
      
      testCase.verifyThat(ll, IsFinite);
    end
  end
end