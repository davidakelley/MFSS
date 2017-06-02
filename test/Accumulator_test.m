% Test Harvey accumulators for filter/smoother

% David Kelley, 2017

classdef Accumulator_test < matlab.unittest.TestCase
  
  properties
    ssGen
    Y
    
    sumAccum
    sumAug
    avgAccum
    avgAug
    triAccum
    triAug
  end
  
  methods(TestClassSetup)
    function setupOnce(testCase)
      % Factor model data
      baseDir = fileparts(fileparts(mfilename('fullpath')));
      addpath(baseDir);
      addpath(fullfile(baseDir, 'examples'));
      
      % Basic state space set up
      p = 3; m = 1; timeDim = 599;
      testCase.ssGen = generateARmodel(p, m, false);
      testCase.ssGen.T(1,:) = [0.5 0.3];
      testCase.ssGen.c = [1 -0.5]';

      testCase.Y = generateData(testCase.ssGen, timeDim)';
      testCase.ssGen.n = timeDim;
      testCase.ssGen = testCase.ssGen.setInvariantTau();

      % Sum accum
      sumData = testCase.Y;
      sumData(:, 2) = Accumulator_test.aggregateY(sumData(:, 2), 3, 'sum');
      
      testCase.sumAccum = Accumulator.GenerateRegular(sumData, {'', 'sum', ''}, [1 3 1]);
      testCase.sumAug = testCase.sumAccum.computeAugSpecification(testCase.ssGen);
      
      % Avg accum
      avgData = testCase.Y;
      avgData(:, 2) = Accumulator_test.aggregateY(avgData(:, 2), 3, 'avg');
      
      testCase.avgAccum = Accumulator.GenerateRegular(avgData, {'', 'avg', ''}, [1 1 1]);
      testCase.avgAug = testCase.avgAccum.computeAugSpecification(testCase.ssGen);
      
      % Triangle accumulator
      testCase.triAccum = Accumulator.GenerateRegular(avgData, {'', 'avg', ''}, [1 3 1]);
      testCase.triAug = testCase.triAccum.computeAugSpecification(testCase.ssGen);
    end
  end
  
  methods (Test)
    %% Sum accumulator
    function testSumAugmentedT(testCase)
      accum = testCase.sumAccum;
      aug = testCase.sumAug;
      ss = testCase.ssGen;
      
      newT = accum.augmentParamT(ss.T, aug);
      
      % There should be 2 slices of T
      testCase.verifySize(newT, [3 3 2]);
      
      % The first 2 states shouldn't have changed
      testCase.verifyEqual(newT(1:2, 1:2, :), repmat(ss.T, [1 1 2]));
      testCase.verifyEqual(newT(1:2, 3, :), repmat([0; 0], [1 1 2]));
      
      % The last state should use the existing elements of T the same
      testCase.verifyEqual(newT(3, 1:2, :), repmat(ss.T(1, 1:2), [1 1 2]));
      
      % It should have a zero in the non-end-of-period versions
      testCase.verifyEqual(newT(3, 3, 1), 0);
      
      % It should have a one in the end-of-period versions
      testCase.verifyEqual(newT(3, 3, 2), 1);
    end
    
    function testSumAugmentedc(testCase)
      accum = testCase.sumAccum;
      aug = testCase.sumAug;
      ss = testCase.ssGen;
      
      newc = accum.augmentParamc(ss.c, aug);
      
      % There should be 2 slices of c
      testCase.verifySize(newc, [3 2]);
      
      % The first 2 states shouldn't have changed
      testCase.verifyEqual(newc(1:2, :), repmat(ss.c, [1 2]));
      
      % The last state should use the existing elements of c
      testCase.verifyEqual(newc(3, :), repmat(ss.c(1), [1 2]));
    end
    
    function testSumAugmentedR(testCase)
      accum = testCase.sumAccum;
      aug = testCase.sumAug;
      ss = testCase.ssGen;
      
      newR = accum.augmentParamR(ss.R, aug);
      
      % There should be 2 slices of R
      testCase.verifySize(newR, [3 1 2]);
      
      % The first 2 states shouldn't have changed
      testCase.verifyEqual(newR(1:2, :, :), repmat(ss.R, [1 1 2]));
      
      % The last state should use the existing elements of R
      testCase.verifyEqual(newR(3, :, :), repmat(ss.R(1, :), [1 1 2]));
    end
    
    function testSumAugmentedZ(testCase)
      accum = testCase.sumAccum;
      aug = testCase.sumAug;
      ss = testCase.ssGen;
      
      newZ = accum.augmentParamZ(ss.Z, aug);
      
      % Z should expand to cover the new state
      testCase.verifySize(newZ, [3 3]);
      
      % The states that weren't accumulated shouldn't change
      testCase.verifyEqual(newZ([1 3], 1:2), ss.Z([1 3], :));
      testCase.verifyEqual(newZ([1 3], 3), zeros(2, 1));
      
      % The accumulated observation Z elements should be moved
      testCase.verifyEqual(newZ(2, 1:2), zeros(1, 2));
      testCase.verifyEqual(newZ(2, 3), ss.Z(2, 1));      
    end
    
    %% Average accumulator
    function testAvgAugmentedT(testCase)
      accum = testCase.avgAccum;
      aug = testCase.avgAug;
      ss = testCase.ssGen;
      
      newT = accum.augmentParamT(ss.T, aug);
      
      % We've added a state and there should now be 3 slices 
      testCase.verifySize(newT, [3 3 3]);

      % The first two states shouldn't have changed
      testCase.verifyEqual(newT(1:2, 1:2, :), repmat(ss.T, [1 1 3]));
      testCase.verifyEqual(newT(1:2, 3, :), repmat([0; 0], [1 1 3]));
      
      % The existing states should load on the accumulator state with T./cal
      testCase.verifyEqual(newT(3, 1:2, :), ...
        repmat(ss.T(1, 1:2), [1 1 3]) ./ reshape(1:3, [1 1 3]));
      
      % The accumulator state loads on itself with (cal-1)./cal
      testCase.verifyEqual(squeeze(newT(3, 3, :)), ((0:2)./(1:3))');
    end
    
    function testAvgAugmentedc(testCase)
      accum = testCase.avgAccum;
      aug = testCase.avgAug;
      ss = testCase.ssGen;
      
      newc = accum.augmentParamc(ss.c, aug);
      
      % We've added a state and there should now be 3 slices
      testCase.verifySize(newc, [3 3]);

      % The first two states shouldn't have changed
      testCase.verifyEqual(newc(1:2, :), repmat(ss.c, [1 3]));
      
      % The existing states should load on the accumulator state with T./cal
      testCase.verifyEqual(newc(3, :), ...
        repmat(ss.c(1), [1 3]) ./ reshape(1:3, [1 3]));
    end
    
    function testAvgAugmentedR(testCase)
      accum = testCase.avgAccum;
      aug = testCase.avgAug;
      ss = testCase.ssGen;
      
      newR = accum.augmentParamR(ss.R, aug);
      
      % We've added a state and there should now be 3 slices
      testCase.verifySize(newR, [3 1 3]);
      
      % The first 2 states shouldn't have changed
      testCase.verifyEqual(newR(1:2, :, :), repmat(ss.R, [1 1 3]));
      
      % The last state should use the existing elements of R divided by cal
      testCase.verifyEqual(squeeze(newR(3, :, :)), repmat(ss.R(1, :), [3 1]) ./ (1:3)');
    end
    
    function testAvgAugmentedZ(testCase)
      accum = testCase.avgAccum;
      aug = testCase.avgAug;
      ss = testCase.ssGen;
      
      newZ = accum.augmentParamZ(ss.Z, aug);
      
      % Z should expand to cover the new state
      testCase.verifySize(newZ, [3 3]);
      
      % The states that weren't accumulated shouldn't change
      testCase.verifyEqual(newZ([1 3], 1:2), ss.Z([1 3], :));
      testCase.verifyEqual(newZ([1 3], 3), zeros(2, 1));
      
      % The accumulated observation Z elements should be moved
      testCase.verifyEqual(newZ(2, 1:2), zeros(1, 2));
      testCase.verifyEqual(newZ(2, 3), ss.Z(2, 1));
    end
    
    %% Triangle average
    % Z, c and R matricies are computed the same regardless of the horizon of 
    % the accumulator - see tests above. 
    
    function testTriAugmentedT(testCase)
      accum = testCase.triAccum;
      aug = testCase.triAug;
      ss = testCase.ssGen;
      
      newT = accum.augmentParamT(ss.T, aug);
      
      % We've added a state and there should now be 3 slices 
      testCase.verifySize(newT, [3 3 3]);

      % The first two states shouldn't have changed
      testCase.verifyEqual(newT(1:2, 1:2, :), repmat(ss.T, [1 1 3]));
      testCase.verifyEqual(newT(1:2, 3, :), repmat([0; 0], [1 1 3]));
      
      % The existing states should load on the accumulator state with T./cal
      testCase.verifyEqual(newT(3, 1:2, :), ...
        repmat(ss.T(1, 1:2), [1 1 3]) ./ reshape(1:3, [1 1 3]) + reshape(repmat(1./(1:3)', [1 2])', [1 2 3]));
      
      % The accumulator state loads on itself with (cal-1)./cal
      testCase.verifyEqual(squeeze(newT(3, 3, :)), ((0:2)./(1:3))');
    end
    
    %% Utilities
    function testGenerateRegular(testCase)
      data = testCase.Y;
      data(:, 2) = Accumulator_test.aggregateY(data(:, 2), 12, 'sum');
      data(:, 3) = Accumulator_test.aggregateY(data(:, 3), 3, 'avg');

      accum = Accumulator.GenerateRegular(data, {'', 'sum', 'avg'}, [1 12 1]);

      % There should be 2 accumulated series
      testCase.verifyEqual(accum.index, [2 3]);
      
      % First calendar: all zeros with ones every 12 places
      testCase.verifyEqual(accum.calendar(setdiff(1:599, 12:12:599), 1), zeros(550, 1));
      testCase.verifyEqual(accum.calendar(12:12:end, 1), ones(50, 1));
      
      % Second calendar: cycling 1:3
      testCase.verifyEqual(accum.calendar(1:3:end, 2), ones(200, 1));
      testCase.verifyEqual(accum.calendar(2:3:end, 2), 2 * ones(200, 1));
      testCase.verifyEqual(accum.calendar(3:3:end, 2), 3 * ones(200, 1));

      % Horizons: all 12s and 1s
      testCase.verifyEqual(accum.horizon(:, 1), 12 * ones(600, 1));
      testCase.verifyEqual(accum.horizon(:, 2), 1 * ones(600, 1));
    end
    
  end
  
  %% Utility functions
  methods (Static)
    function aggY = aggregateY(Y, pers, type)
      % Aggregate a series evenly every pers periods by sum and average types
      % Note: cannot generate triangle averaged data - horizon on avg type
      % accumulators should always be 1.
      
      timeDim = size(Y, 1);
      timeGroupsPer = sort(repmat((1:ceil(timeDim/pers))', [pers 1]));
      timeGroupsPer(timeDim+1:end, :) = [];
      
      % Generate aggregated data
      if strcmpi(type, 'sum')
        aggYLowF = grpstats(Y, timeGroupsPer, 'mean') .* pers;
      else
        aggYLowF = grpstats(Y, timeGroupsPer, 'mean');
      end
      
      % Trim aggregated data
      if sum(timeGroupsPer == max(timeGroupsPer)) ~= pers
        aggYLowF(end, :) = [];
      end
      
      % Assign to correct timing
      aggY = nan(size(Y));
      aggY(pers:pers:end, :) = aggYLowF;
    end
  end
  
end