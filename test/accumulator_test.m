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
      baseDir =  [subsref(strsplit(mfilename('fullpath'), 'MFSS'), ...
        struct('type', '{}', 'subs', {{1}})) 'MFSS'];
      addpath(baseDir);
      addpath(fullfile(baseDir, 'examples'));

      testCase.bbk = load(fullfile(baseDir, 'examples', 'data', 'bbk_data.mat'));
      testCase.deai = load(fullfile(baseDir, 'examples', 'data', 'deai.mat'));
    end
  end
  
  methods (Test)
    function testNoAccumWithMissing(testCase)
      % Run smoother over a dataset with missing observations, check that
      % its close to a dataset without missing values. 
      p = 10; m = 1; timeDim = 500;
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
      p = 2; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      latentY = generateData(ssGen, timeDim);
      
      timeGroups = sort(repmat((1:ceil(timeDim/3))', [3 1]));
      timeGroups(end, :) = [];
      
      aggSeries = logical([0 1]);
      aggY = grpstats(latentY(aggSeries, :)', timeGroups, 'mean')' .* 3;
      aggY(:, end) = [];
      Y = latentY;
      Y(aggSeries, :) = nan;
      Y(aggSeries, 3:3:end) = aggY;
      
      accum = struct;
      accum.xi = aggSeries;
      accum.psi = repmat([1 2 3]', [(timeDim+1)/3, sum(aggSeries)])';
      accum.Horizon = repmat(3, [timeDim+1, sum(aggSeries)])';
      
      ss = StateSpace(ssGen.Z, ssGen.d, ssGen.H, ...
        ssGen.T, ssGen.c, ssGen.R, ssGen.Q);
      ssA = ss.addAccumulators(accum);
      
      alpha = ssA.smooth(Y);
      latentAlpha = ssGen.smooth(latentY);
      
      testCase.verifyGreaterThan(corr(alpha(1,:)', latentAlpha(1,:)'), 0.96);
      testCase.verifyEqual(alpha(1,:), latentAlpha(1,:), 'AbsTol', 0.75, 'RelTol', 0.5);      
    end
    
    function testDetroit(testCase)
      import matlab.unittest.constraints.IsFinite;
      detroit = testCase.deai;
      
      ss0 = StateSpace(detroit.Z, detroit.d, detroit.H, ...
                       detroit.T, detroit.c, detroit.R, detroit.Q);
      
      ss0A = ss0.addAccumulators(detroit.Harvey);
      
      [~, ll] = ss0.filter(detroit.Y);
      
      testCase.verifyThat(ll, IsFinite);
    end
  end
end