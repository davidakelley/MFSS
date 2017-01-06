% Test Harvey accumulators for filter/smoother

% David Kelley, 2016

classdef accumulatorIntegration_test < matlab.unittest.TestCase
  
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
    %% StateSpace tests    
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
      
      Y = generateData(ssGen, timeDim)';
      Y(:, 2) = accumulator_test.aggregateY(Y(:, 2), 3, 'sum');

      accum = Accumulator.GenerateRegular(Y, {'', 'sum'}, [1 3]);
      ssA = accum.augmentStateSpace(ssGen);
       
      % Sizes
      testCase.verifySize(ssA.Z, [p m+1+1]);
      testCase.verifySize(ssA.T, [m+2 m+2 2]);
      testCase.verifySize(ssA.c, [m+2 2]);
      testCase.verifySize(ssA.R, [m+2 1 2]);
      
      % Make sure 0 and 1 elements in the right places
      testCase.verifyEqual(ssA.T(m+2, m+2, 1), 0);
      testCase.verifyEqual(ssA.T(m+2, m+2, 2), 1);
      
      % Make sure Z elements got moved
      testCase.verifyEqual(ssA.Z(2, m+2), ssGen.Z(2,1));
      testCase.verifyEqual(ssA.Z(2, 1), 0);
      
      % Make sure the smoother works
      [~, ll] = ssA.filter(Y');
      testCase.verifyThat(ll, matlab.unittest.constraints.IsFinite)

%       alpha = ssA.smooth(Y);
%       latentAlpha = ssGen.smooth(Y);
%       testCase.verifyGreaterThan(corr(alpha(1,:)', latentAlpha(1,:)'), 0.94);
%       testCase.verifyEqual(alpha(1,:), latentAlpha(1,:), 'AbsTol', 0.75, 'RelTol', 0.5);
    end
    
    function testSumAccumutlatorMultiple(testCase)
      p = 3; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      Y = generateData(ssGen, timeDim)';
      
      Y(:, 2) = accumulator_test.aggregateY(Y(:, 2), 3, 'sum');
      Y(:, 3) = accumulator_test.aggregateY(Y(:, 3), 12, 'sum');
      
      accum = Accumulator.GenerateRegular(Y, {'', 'sum', 'sum'}, [1 3 12]);
      
      ssA = accum.augmentStateSpace(ssGen);
      
      % Sizes
      testCase.verifySize(ssA.Z, [3 13]);
      testCase.verifySize(ssA.T, [13 13 3]);
      testCase.verifySize(ssA.c, [13 3]);
      testCase.verifySize(ssA.R, [13 1 3]);
      
      % Make sure 0 and 1 elements in the right places
      testCase.verifyEqual(ssA.T(12:13, 12:13, 1), zeros(2, 2));
      testCase.verifyEqual(ssA.T(12:13, 12:13, 2), [1 0; 0 0]);
      testCase.verifyEqual(ssA.T(12:13, 12:13, 3), eye(2));
      
      % Make sure Z elements got moved
      testCase.verifyEqual(ssA.Z(2, 12), ssGen.Z(2,1));
      testCase.verifyEqual(ssA.Z(3, 13), ssGen.Z(3,1));
      testCase.verifyEqual(ssA.Z(2, 1), 0);
      testCase.verifyEqual(ssA.Z(3, 1), 0);
      
      % Make sure the smoother works
      [~, ll] = ssA.filter(Y');
      testCase.verifyThat(ll, matlab.unittest.constraints.IsFinite)
    end
    
    function testAvgAccumulatorTwoAccum(testCase)
      p = 10; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      
      Y = generateData(ssGen, timeDim)';

      aggY = Y;
      aggY(:, 2:3) = accumulator_test.aggregateY(Y(:, 2:3), 3, 'avg');

      accum = Accumulator.GenerateRegular(aggY, {'', 'avg', 'avg'}, [1 3 3]);
      ssA = accum.augmentStateSpace(ssGen);
            
%       testCase.verifyEqual();
    end
    
    function testAvgAccumulatorAddLags(testCase)
      p = 10; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      
      Y = generateData(ssGen, timeDim)';

      aggY = Y;
      aggY(:, 2) = accumulator_test.aggregateY(Y(:, 2), 6, 'avg');

      % Not really the right horizon for this data, but we need a test for lags
      accum = Accumulator.GenerateRegular(aggY, {'', 'avg', ''}, [1 6 1]);
      ssA = accum.augmentStateSpace(ssGen);
            
      testCase.verifyEqual(ssA.T(2:5, 1:4, 1), eye(4));
    end
    
    function testAvgAccumulatorSmoother(testCase)
      p = 10; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      
      Y = generateData(ssGen, timeDim)';

      aggY = Y;
      aggY(:, 2:3) = accumulator_test.aggregateY(Y(:, 2:3), 3, 'avg');
      aggY(:, 4) = accumulator_test.aggregateY(Y(:, 4), 12, 'avg');

      accum = Accumulator.GenerateRegular(aggY, {'', 'avg', 'avg', 'avg'}, [1 3 3 12]);
      ssA = accum.augmentStateSpace(ssGen);
            
      % Sizes
      testCase.verifySize(ssA.Z, [p 13]);
      testCase.verifySize(ssA.T, [13 13 12]);
      testCase.verifySize(ssA.c, [13 12]);
      testCase.verifySize(ssA.R, [13 1 12]);
      
      % Make sure 0 and 1 elements in the right places
      testCase.verifyEqual(ssA.T(12, 12, 1:4), zeros(1, 1, 4));
      testCase.verifyEqual(ssA.T(1:1+m, 1:1+m, :), repmat(ssGen.T(1:1+m, 1:1+m), [1 1 12]));
      
      % Make sure Z elements got moved
      testCase.verifyEqual(ssA.Z(2, 12), ssGen.Z(2,1));
      testCase.verifyEqual(ssA.Z(3, 13), ssGen.Z(3,1));
      testCase.verifyEqual(ssA.Z(2:3, 1), zeros(2, 1));
      
      % Make sure the smoother works
      [~, ll] = ssA.filter(Y');
      testCase.verifyThat(ll, matlab.unittest.constraints.IsFinite)
      
%       latentAlpha = ssGen.smooth(Y);
%       alpha = ssA.smooth(aggY);
%       testCase.verifyGreaterThan(corr(alpha(1,:)', latentAlpha(1,:)'), 0.96);
%       testCase.verifyEqual(alpha(1,:), latentAlpha(1,:), 'AbsTol', 0.75, 'RelTol', 0.5);
    end
    
    function testMultipleStateProcess(testCase)
      p = 7; timeDim = 205;
      m1 = generateARmodel(p, 1, true);
      m1.T(1,:) = [0.85 0.05];
      m2 = generateARmodel(p, 2, true);
      m2.T(1,:) = [0.85 0.05 -0.2];
      m = m1.m + m2.m;
      ssGen = StateSpace([m1.Z m2.Z], zeros(p,1), m1.H, ...
        blkdiag(m1.T, m2.T), zeros(m, 1), blkdiag(m1.R, m2.R), blkdiag(m1.Q, m2.Q));
      
      [Y, trueAlpha] = generateData(ssGen, timeDim);

      aggY = Y';
      aggY(:, 3) = accumulator_test.aggregateY(Y(3, :)', 3, 'sum');
      aggY(:, 6) = accumulator_test.aggregateY(Y(4, :)', 3, 'avg');
      aggY(:, 7) = accumulator_test.aggregateY(Y(4, :)', 12, 'avg');

      accum = Accumulator.GenerateRegular(aggY, ...
        {'', '', 'sum', '', '', 'avg', 'avg'}, [1 1 3 1 1 1 1]);
      ssA = accum.augmentStateSpace(ssGen);
        
      % Sizes
      testCase.verifySize(ssA.Z, [p m+6]);
      testCase.verifySize(ssA.T, [m+6 m+6 12]);
      testCase.verifySize(ssA.c, [m+6 12]);
      testCase.verifySize(ssA.R, [m+6 2 12]);
      
      % Make sure 0 and 1 elements in the right places - not sure about this
%       testCase.verifyEqual(ssA.T(12, 12, 1), zeros(1, 1, 4));
%       testCase.verifyEqual(ssA.T(1:1+m, 1:1+m, :), repmat(ssGen.T(1:1+m, 1:1+m), [1 1 12]));
      
      % Make sure Z elements got moved
      testCase.verifyEqual(ssA.Z([3 6 7], [1 3]), zeros(3, 2));
      testCase.verifyEqual(ssA.Z(3, [6 7]), ssGen.Z(3, [1 3]));
      testCase.verifyEqual(ssA.Z(6, [8 9]), ssGen.Z(6, [1 3]));
      testCase.verifyEqual(ssA.Z(7, [10 11]), ssGen.Z(7, [1 3]));
      
      % Make sure the smoother works
      [aFilt, ll] = ssA.filter(Y);
      testCase.verifyThat(ll, matlab.unittest.constraints.IsFinite)
      
    end
    
    function testAccumExistingTVP(testCase)
      % Check to make sure we can add an accumulator to a StateSpace that's got
      % slices of T, c or R. Easiest way to test this is to add 2 accumulators
      % to a StateSpace separately.
    end
    
    function testDetroit(testCase)
      import matlab.unittest.constraints.IsFinite;
      detroit = testCase.deai;
      
      ss0 = StateSpace(detroit.Z, detroit.d, detroit.H, ...
        detroit.T, detroit.c, detroit.R, detroit.Q);
      
      deaiAccum = Accumulator(detroit.Harvey.xi, detroit.Harvey.psi', detroit.Harvey.Horizon');
      ss0A = deaiAccum.augmentStateSpace(ss0);
      
      [~, ll] = ss0A.filter(detroit.Y);
      
      testCase.verifyThat(ll, IsFinite);
    end
    
    %% ThetaMap tests
    function testThetaMapAR(testCase)
      p = 2; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      
      Y = generateData(ssGen, timeDim)';
      Y(:, 2) = accumulator_test.aggregateY(Y(:, 2), 3, 'sum');

      accum = Accumulator.GenerateRegular(Y, {'', 'sum'}, [1 3]);

      ssA = accum.augmentStateSpace(ssGen);
      
      TssE = ssGen.T;
      TssE(1,:) = [nan nan];
      ssE = StateSpaceEstimation(ssGen.Z, ssGen.d, ssGen.H, ...
        TssE, ssGen.c, ssGen.R, ssGen.Q);
      tm = ssE.ThetaMapping;
      
      tm2 = accum.augmentThetaMap(tm);
      
      % Test system -> theta
      testCase.verifyEqual(tm2.nTheta, tm.nTheta);
      thetaOrig = tm.system2theta(ssGen);
      thetaAug = tm2.system2theta(ssA);
      testCase.verifyEqual(thetaOrig, thetaAug);
      
      % Test theta -> system
      thetaTest = [12, -7.4]';
      ssTestAug = tm2.theta2system(thetaTest);
      thetaTestAug = tm2.system2theta(ssTestAug);
      testCase.verifyEqual(thetaTest, thetaTestAug);
      
      ssNew = tm.theta2system(thetaTest);
      ssNewAug = accum.augmentStateSpace(ssNew);
      testCase.verifyEqual(ssTestAug, ssNewAug);      
    end
   
    function testThetaMapARAllAvg(testCase)
      p = 2; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      
      Y = generateData(ssGen, timeDim)';
      Y(:, 2) = accumulator_test.aggregateY(Y(:, 2), 3, 'avg');

      accum = Accumulator.GenerateRegular(Y, {'', 'avg'}, [1 3]);

      ssA = accum.augmentStateSpace(ssGen);
      
      tm = ThetaMap.ThetaMapAll(ssGen);
      tm2 = accum.augmentThetaMap(tm);

      % Test system -> theta
      testCase.verifyEqual(tm2.nTheta, tm.nTheta);
      thetaOrig = tm.system2theta(ssGen);
      thetaAug = tm2.system2theta(ssA);
      testCase.verifyEqual(thetaOrig, thetaAug);
      
      % Test theta -> system
      thetaTest = [12, -7.4]';
      ssTestAug = tm2.theta2system(thetaTest);
      thetaTestAug = tm2.system2theta(ssTestAug);
      testCase.verifyEqual(thetaTest, thetaTestAug);
      
      ssNew = tm.theta2system(thetaTest);
      ssNewAug = accum.augmentStateSpace(ssNew);
      testCase.verifyEqual(ssTestAug, ssNewAug);      
    end
    
  end
end