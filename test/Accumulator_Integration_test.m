% Test accumulators for filter/smoother

% David Kelley, 2016-2018

classdef Accumulator_Integration_test < matlab.unittest.TestCase
  
  methods (Test)
    %% StateSpace parameter tests
    function testSumParameters(testCase)
      p = 2; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      
      Y = generateData(ssGen, timeDim)';
      Y(:, 2) = Accumulator_test.aggregateY(Y(:, 2), 3, 'sum');
      
      accum = Accumulator.GenerateRegular(Y, {'', 'sum'}, [1 3]);
      ssA = accum.augmentStateSpace(ssGen);
      
      % Sizes
      testCase.verifySize(ssA.Z, [p m+1+1]);
      testCase.verifySize(ssA.T, [m+2 m+2 2]);
      testCase.verifySize(ssA.c, [m+2 1]);
      testCase.verifySize(ssA.R, [m+2 1]);
      
      % Make sure Z elements got moved
      testCase.verifyEqual(ssA.Z(2, m+2), ssGen.Z(2,1));
      testCase.verifyEqual(ssA.Z(2, 1), 0);
      
      % T Matrix - Make sure 0 and 1 elements in the right places
      testCase.verifyEqual(ssA.T(m+2, m+2, 1), 0);
      testCase.verifyEqual(ssA.T(m+2, m+2, 2), 1);

      % c vector 
      testCase.verifyEqual(ssA.c, zeros(m+2,1));            
      % R matrix
      testCase.verifyEqual(ssA.R, [1; 0; 1]);
    end
    
    function testSumMultipleParameters(testCase)
      p = 3; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      Y = generateData(ssGen, timeDim)';
      
      Y(:, 2) = Accumulator_test.aggregateY(Y(:, 2), 3, 'sum');
      Y(:, 3) = Accumulator_test.aggregateY(Y(:, 3), 12, 'sum');
      
      accum = Accumulator.GenerateRegular(Y, {'', 'sum', 'sum'}, [1 3 12]);
      
      ssA = accum.augmentStateSpace(ssGen);
      
      % Sizes
      testCase.verifySize(ssA.Z, [3 4]);
      testCase.verifySize(ssA.T, [4 4 3]);
      testCase.verifySize(ssA.c, [4 1]);
      testCase.verifySize(ssA.R, [4 1]);
      
      % Make sure Z elements got moved
      testCase.verifyEqual(ssA.Z(2, 3), ssGen.Z(2,1));
      testCase.verifyEqual(ssA.Z(3, 4), ssGen.Z(3,1));
      testCase.verifyEqual(ssA.Z(2, 1), 0);
      testCase.verifyEqual(ssA.Z(3, 1), 0);

      % T matrix
      testCase.verifyEqual(ssA.T(3, 1:2, 1), ssGen.T(1,1:2));
      testCase.verifyEqual(ssA.T(4, 1:2, 1), ssGen.T(1,1:2));
      testCase.verifyEqual(ssA.T(3, 1:2, 2), ssGen.T(1,1:2));
      testCase.verifyEqual(ssA.T(4, 1:2, 2), ssGen.T(1,1:2));
      testCase.verifyEqual(ssA.T(3, 1:2, 3), ssGen.T(1,1:2));
      testCase.verifyEqual(ssA.T(4, 1:2, 3), ssGen.T(1,1:2));
      % Make sure 0 and 1 elements in the right places
      testCase.verifyEqual(ssA.T(3:4, 3:4, 1), zeros(2, 2));
      testCase.verifyEqual(ssA.T(3:4, 3:4, 2), [0 0; 0 1]);
      testCase.verifyEqual(ssA.T(3:4, 3:4, 3), eye(2));
      
      % c vector 
      testCase.verifyEqual(ssA.c, zeros(4,1));

      % R matrix
      testCase.verifyEqual(ssA.R, [1; 0; 1; 1]);
      
    end
    
    function testSumCommonParameters(testCase)      
      p = 3; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      Y = generateData(ssGen, timeDim)';
      
      Y(:, 2) = Accumulator_test.aggregateY(Y(:, 2), 3, 'sum');
      Y(:, 3) = Accumulator_test.aggregateY(Y(:, 3), 3, 'sum');
      
      accum = Accumulator.GenerateRegular(Y, {'', 'sum', 'sum'}, [1 3 3]);
      
      ssA = accum.augmentStateSpace(ssGen);
      
      % Sizes
      testCase.verifySize(ssA.Z, [3 3]);
      testCase.verifySize(ssA.T, [3 3 2]);
      testCase.verifySize(ssA.c, [3 1]);
      testCase.verifySize(ssA.R, [3 1]);
      
      % Make sure Z elements got moved
      testCase.verifyEqual(ssA.Z(2:3, 3), ssGen.Z(2:3,1));
      testCase.verifyEqual(ssA.Z(2:3, 1), [0; 0]);
      
      % T matrix
      testCase.verifyEqual(ssA.T(3, 1:2, 1:2), repmat(ssGen.T(1,1:2), [1 1 2]));
      testCase.verifyEqual(ssA.T(3, 3, 1:2), cat(3, 0, 1));
      
    end
    
    function testAvgParameters(testCase)
      p = 10; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      
      Y = generateData(ssGen, timeDim)';
      
      aggY = Y;
      aggY(:, 2) = Accumulator_test.aggregateY(Y(:, 2), 3, 'avg');
      
      accum = Accumulator.GenerateRegular(aggY, {'', 'avg'}, [1 3]);
      ssA = accum.augmentStateSpace(ssGen);
      
      % Sizes
      testCase.verifySize(ssA.Z, [p m+1+1]);
      testCase.verifySize(ssA.T, [m+2 m+2 3]);
      testCase.verifySize(ssA.c, [m+2 3]);
      testCase.verifySize(ssA.R, [m+2 1 3]);
      
      % Make sure Z elements got moved
      newZ = zeros(size(ssGen.Z,1), 3);
      newZ([1 3:10],1) = ssGen.Z([1 3:10],1);
      newZ(2,3) = ssGen.Z(2,1);
      testCase.verifyEqual(ssA.Z, newZ);
      
      % T Matrix - Make sure 0 and 1 elements in the right places
      testCase.verifyEqual(ssA.T(1:2, 1:2, :), repmat(ssGen.T, [1 1 3]));
      testCase.verifyEqual(ssA.T(3, 1:2, 1), ssGen.T(1,1:2) + [1 1]);
      testCase.verifyEqual(ssA.T(3, 1:2, 2), (ssGen.T(1,1:2) + [1 1]) ./ 2);
      testCase.verifyEqual(ssA.T(3, 1:2, 3), (ssGen.T(1,1:2) + [1 1]) ./ 3, 'AbsTol', 1e-15);
      testCase.verifyEqual(ssA.T(3, 3, :), reshape([0 1/2 2/3], [1 1 3]));

      % c vector 
      testCase.verifyEqual(ssA.c, zeros(m+2,3));            
      % R matrix
      testCase.verifyEqual(ssA.R(:,:,1), [1; 0; 1]);
      testCase.verifyEqual(ssA.R(:,:,2), [1; 0; 1/2]);
      testCase.verifyEqual(ssA.R(:,:,3), [1; 0; 1/3]);      
    end
    
    function testAvgMultipleParameters(testCase)
      p = 10; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      
      Y = generateData(ssGen, timeDim)';
      
      aggY = Y;
      aggY(:, 2:3) = Accumulator_test.aggregateY(Y(:, 2:3), 3, 'avg');
      
      accum = Accumulator.GenerateRegular(aggY, {'', 'avg', 'avg'}, [1 3 3]);
      ssA = accum.augmentStateSpace(ssGen);

      % Sizes
      testCase.verifySize(ssA.Z, [p m+1+1]);
      testCase.verifySize(ssA.T, [m+2 m+2 3]);
      testCase.verifySize(ssA.c, [m+2 3]);
      testCase.verifySize(ssA.R, [m+2 1 3]);
      
      % Make sure Z elements got moved
      newZ = zeros(size(ssGen.Z,1), 3);
      newZ([1 4:10],1) = ssGen.Z([1 4:10],1);
      newZ(2:3,3) = ssGen.Z(2:3,1);
      testCase.verifyEqual(ssA.Z, newZ);
      
      % T Matrix - Make sure 0 and 1 elements in the right places
      testCase.verifyEqual(ssA.T(1:2, 1:2, :), repmat(ssGen.T, [1 1 3]));
      testCase.verifyEqual(ssA.T(3, 1:2, 1), ssGen.T(1,1:2) + [1 1]);
      testCase.verifyEqual(ssA.T(3, 1:2, 2), (ssGen.T(1,1:2) + [1 1]) ./ 2);
      testCase.verifyEqual(ssA.T(3, 1:2, 3), (ssGen.T(1,1:2) + [1 1]) ./ 3, 'AbsTol', 1e-15);
      testCase.verifyEqual(ssA.T(3, 3, :), reshape([0 1/2 2/3], [1 1 3]));

      % c vector 
      testCase.verifyEqual(ssA.c, zeros(m+2,3));            
      % R matrix
      testCase.verifyEqual(ssA.R(:,:,1), [1; 0; 1]);
      testCase.verifyEqual(ssA.R(:,:,2), [1; 0; 1/2]);
      testCase.verifyEqual(ssA.R(:,:,3), [1; 0; 1/3]);      
    end
    
    function testAvgMultipleParametersSeparated(testCase)
      p = 10; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      
      Y = generateData(ssGen, timeDim)';
      
      aggY = Y;
      aggY(:, [2 5]) = Accumulator_test.aggregateY(Y(:, [2 5]), 3, 'avg');
      
      accum = Accumulator.GenerateRegular(aggY, {'', 'avg', '', '', 'avg'}, [1 3 1 1 3]);
      ssA = accum.augmentStateSpace(ssGen);

      % Sizes
      testCase.verifySize(ssA.Z, [p m+1+1]);
      testCase.verifySize(ssA.T, [m+2 m+2 3]);
      testCase.verifySize(ssA.c, [m+2 3]);
      testCase.verifySize(ssA.R, [m+2 1 3]);
      
      % Make sure Z elements got moved
      newZ = zeros(size(ssGen.Z,1), 3);
      newZ([1 3:4 6:10],1) = ssGen.Z([1 3:4 6:10],1);
      newZ([2 5],3) = ssGen.Z([2 5],1);
      testCase.verifyEqual(ssA.Z, newZ);
      
      % T Matrix - Make sure 0 and 1 elements in the right places
      testCase.verifyEqual(ssA.T(1:2, 1:2, :), repmat(ssGen.T, [1 1 3]));
      testCase.verifyEqual(ssA.T(3, 1:2, 1), ssGen.T(1,1:2) + [1 1]);
      testCase.verifyEqual(ssA.T(3, 1:2, 2), (ssGen.T(1,1:2) + [1 1]) ./ 2);
      testCase.verifyEqual(ssA.T(3, 1:2, 3), (ssGen.T(1,1:2) + [1 1]) ./ 3, 'AbsTol', 1e-15);
      testCase.verifyEqual(ssA.T(3, 3, :), reshape([0 1/2 2/3], [1 1 3]));

      % c vector 
      testCase.verifyEqual(ssA.c, zeros(m+2,3));            
      % R matrix
      testCase.verifyEqual(ssA.R(:,:,1), [1; 0; 1]);
      testCase.verifyEqual(ssA.R(:,:,2), [1; 0; 1/2]);
      testCase.verifyEqual(ssA.R(:,:,3), [1; 0; 1/3]);
    end
    
    function testAvgAddLagsParameters(testCase)
      p = 10; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      
      Y = generateData(ssGen, timeDim)';
      
      aggY = Y;
      aggY(:, 2) = Accumulator_test.aggregateY(Y(:, 2), 6, 'avg');
      
      % Not really the right horizon for this data, but we need a test for lags
      accum = Accumulator.GenerateRegular(aggY, {'', 'avg', ''}, [1 6 1]);
      ssA = accum.augmentStateSpace(ssGen);
            
      % Sizes
      testCase.verifySize(ssA.Z, [p 6]);
      testCase.verifySize(ssA.T, [6 6 6]);
      testCase.verifySize(ssA.c, [6 6]);
      testCase.verifySize(ssA.R, [6 1 6]);
      
      % Make sure Z elements got moved
      newZ = zeros(size(ssGen.Z,1), 6);
      newZ([1 3:10],1) = ssGen.Z([1 3:10],1);
      newZ(2,6) = ssGen.Z(2,1);
      testCase.verifyEqual(ssA.Z, newZ);
      
      % T Matrix - Make sure 0 and 1 elements in the right places
      testCase.verifyEqual(ssA.T(1:2, 1:2, :), repmat(ssGen.T, [1 1 6]));
      testCase.verifyEqual(ssA.T(2:5, 1:4, :), repmat(eye(4), [1 1 6]));
      testCase.verifyEqual(ssA.T(6, 1:5, :), ...
        repmat([ssGen.T(1,1:2) zeros(1,3)] + ones(1,5), [1 1 6]) ...
        ./ reshape(1:6, [1 1 6]), 'AbsTol', 1e-15);
      testCase.verifyEqual(ssA.T(6, 6, :), reshape((0:5) ./ (1:6), [1 1 6]));

      % c vector 
      testCase.verifyEqual(ssA.c, zeros(6,6));
      % R matrix
      testCase.verifyEqual(ssA.R(1,1,:), ones(1,1,6));
      testCase.verifyEqual(ssA.R(6,1,:), reshape(1 ./ (1:6), [1 1 6]));
    end
    
    function testAvgAddLagsMultipleParameters(testCase)
      % Test adding lags for a multiple observation, multiple state process
      timeDim = 599;
      p = 4;
      Z = [1 0 1 0; 1 0 1 0; 1 0 0 0; 0 0 1 0];
      H = 0.2 * eye(4);
      T = [0.69 0.29 0 0; 1 0 0 0; 0 0 .35 .64; 0 0 1 0];
      Q = eye(2);
      R = [1 0; 0 0; 0 1; 0 0];
      ssGen = StateSpace(Z, H, T, Q, 'R', R);
      
      Y = generateData(ssGen, timeDim)';
      
      aggY = Y;
      aggY(:, 3) = Accumulator_test.aggregateY(Y(:, 3), 6, 'avg');
      aggY(:, 4) = Accumulator_test.aggregateY(Y(:, 4), 6, 'avg');

      % Not really the right horizon for this data, but we need a test for lags
      accum = Accumulator.GenerateRegular(aggY, {'', '', 'avg', 'avg'}, [1 1 6 6]);
      ssA = accum.augmentStateSpace(ssGen);
            
      % Sizes
      testCase.verifyEqual(ssA.m, 12);
      
      % Make sure Z elements got moved
      testCase.verifyEqual(ssA.Z(3:4,11:12), eye(2));
      
      % T Matrix - Make sure the lag elements got put in the right places
      % State 2 -> state 5, state 5 -> state 6 & state 6 -> state 7 for lags of state 1
      testCase.verifyEqual(ssA.T(5,2), 1); 
      testCase.verifyEqual(ssA.T(6,5), 1); 
      testCase.verifyEqual(ssA.T(7,6), 1); 
      % State 4 -> state 8, state 8 -> state 9 & state 9 -> state 10 for lags of state 2
      testCase.verifyEqual(ssA.T(8,4), 1); 
      testCase.verifyEqual(ssA.T(9,8), 1); 
      testCase.verifyEqual(ssA.T(10,9), 1);
    end
    
    function testAccumExistingTVP(testCase)
      % Check to make sure we can add an accumulator to a StateSpace that's got
      % slices of T, c or R. Easiest way to test this is to add 2 accumulators
      % to a StateSpace separately.
      
      assumeFail(testCase); % Write full test
    end
        
    function testSameStateSumParameters(testCase)
      % Test that accumlators for the first and third observations work. This is really
      % testing the ordering of the accumulator variables. We should order them by
      % observation then by state, in the order that they are needed based on the first
      % observation they're used for. 
      
      Z = reshape((1:8)', [4 2]);
      H = eye(4);      
      T = diag([.99 -.5]);
      Q = eye(2);
      
      ss = StateSpace(Z, H, T, Q);
      % Generate data for y and aggregate the first and 3rd series
      y = generateData(ss, 300);
      y(1,:) = reshape([nan(100,2) mean(reshape(y(1,:), [100 3]), 2)]', [], 1)';
      y(3,:) = reshape([nan(50,5) sum(reshape(y(3,:), [50 6]), 2)]', [], 1)';
      y(4,:) = reshape([nan(100,2) mean(reshape(y(4,:), [100 3]), 2)]', [], 1)';
      
      accum = Accumulator.GenerateRegular(y', {'sum', '', 'sum', 'sum'}, [3 1 6 3]);
      ssA = accum.augmentStateSpace(ss);
      
      % Sizes
      testCase.verifySize(ssA.Z, [4 6]);
      testCase.verifySize(ssA.T, [6 6 3]);
      testCase.verifySize(ssA.c, [6 1]);
      testCase.verifySize(ssA.R, [6 2]);
      
      % Make sure Z elements got moved and the states are ordered correctly.
      testZ = [...
        0 0 1 5 0 0
        2 6 0 0 0 0
        0 0 0 0 3 7
        0 0 4 8 0 0];
      testCase.verifyEqual(ssA.Z, testZ);
      
      % T Matrix - Make sure 0 and 1 elements in the right places
      testCase.verifyEqual(ssA.T(:, 1:2, :), repmat(ss.T, [3 1 3]));
      testCase.verifyEqual(ssA.T(3:4, 3:4, 1:2), zeros(2,2,2));
      testCase.verifyEqual(ssA.T(3:4, 3:4, 3), eye(2));
      testCase.verifyEqual(ssA.T(5:6, 5:6, 1), zeros(2, 2));
      testCase.verifyEqual(ssA.T(5:6, 5:6, 2:3), repmat(eye(2), [1 1 2]));
      
      % c vector 
      testCase.verifyEqual(ssA.c, zeros(6,1));
      
      % R matrix
      testCase.verifyEqual(ssA.R, repmat(ss.R, [3 1]));      
    end
    
    function testSameStateSumParametersB(testCase)
      % Test that accumlators in the first and third observations work. Basically the same
      % test as above but a different ordering to make sure we did it right. 
            
      Z = reshape((1:8)', [4 2]);
      H = eye(4);
      
      T = diag([.99 -.5]);
      Q = eye(2);
      
      ss = StateSpace(Z, H, T, Q);
      % Generate data for y and aggregate the first and 3rd series
      y = generateData(ss, 300);
      y(1,:) = reshape([nan(50,5) sum(reshape(y(1,:), [50 6]), 2)]', [], 1)';
      y(3,:) = reshape([nan(50,5) sum(reshape(y(3,:), [50 6]), 2)]', [], 1)';
      y(4,:) = reshape([nan(100,2) sum(reshape(y(4,:), [100 3]), 2)]', [], 1)';
      
      accum = Accumulator.GenerateRegular(y', {'sum', '', 'sum', 'sum'}, [6 1 6 3]);
      ssA = accum.augmentStateSpace(ss);
      
      % Sizes
      testCase.verifySize(ssA.Z, [4 6]);
      testCase.verifySize(ssA.T, [6 6 3]);
      testCase.verifySize(ssA.c, [6 1]);
      testCase.verifySize(ssA.R, [6 2]);
      
      % Make sure Z elements got moved and the states are ordered correctly.
      testZ = [...
        0 0 1 5 0 0
        2 6 0 0 0 0
        0 0 3 7 0 0
        0 0 0 0 4 8 ];
      testCase.verifyEqual(ssA.Z, testZ);
      
      % T Matrix - Make sure 0 and 1 elements in the right places
      testCase.verifyEqual(ssA.T(:, 1:2, :), repmat(ss.T, [3 1 3]));
      testCase.verifyEqual(ssA.T(3:4, 3:4, 1), zeros(2,2));
      testCase.verifyEqual(ssA.T(3:4, 3:4, 2:3), repmat(eye(2), [1 1 2]));
      testCase.verifyEqual(ssA.T(5:6, 5:6, 1:2), zeros(2, 2, 2));
      testCase.verifyEqual(ssA.T(5:6, 5:6, 3), eye(2));
      
      % c vector 
      testCase.verifyEqual(ssA.c, zeros(6,1));
      
      % R matrix
      testCase.verifyEqual(ssA.R, repmat(ss.R, [3 1]));
    end
    
    function testSameStateAvgAccumulators(testCase)
      % Test that accumlators in the first and third observations work
      Z = [1; .5; 1];
      H = diag(ones(3,1));
      
      T = diag(.99);
      Q = 1;
      
      ss = StateSpace(Z, H, T, Q);
      % Generate data for y and aggregate the first and 3rd series
      y = generateData(ss, 300);
      y(1,:) = reshape([nan(100,2) mean(reshape(y(1,:), [100 3]), 2)]', [], 1)';
      y(3,:) = reshape([nan(100,2) mean(reshape(y(3,:), [100 3]), 2)]', [], 1)';
      
      accum = Accumulator.GenerateRegular(y', {'avg', '', 'avg'}, [3 1 3]);
      ssA = accum.augmentStateSpace(ss);
      
      % Sizes
      testCase.verifySize(ssA.Z, [3 3]);
      testCase.verifySize(ssA.T, [3 3 3]);
      testCase.verifySize(ssA.c, [3 3]);
      testCase.verifySize(ssA.R, [3 1 3]);
      
      % Make sure Z elements got moved
      testCase.verifyEqual(ssA.Z, [0 0 1; .5 0 0; 0 0 1]);
      
      % T Matrix - Make sure 0 and 1 elements in the right places
      testCase.verifyEqual(ssA.T(1, 1, :), repmat(ss.T, [1 1 3]));
      testCase.verifyEqual(ssA.T(2, 1, :), ones(1,1,3));
      testCase.verifyEqual(ssA.T(3, 1:2, 1), [ss.T+1 1]);
      testCase.verifyEqual(ssA.T(3, 1:2, 2), [ss.T+1 1] ./ 2);
      testCase.verifyEqual(ssA.T(3, 1:2, 3), [ss.T+1 1] ./ 3, 'AbsTol', 1e-15);
      testCase.verifyEqual(ssA.T(3, 3, :), reshape([0 1/2 2/3], [1 1 3]));

      % c vector 
      testCase.verifyEqual(ssA.c, zeros(3,3));            
      % R matrix
      testCase.verifyEqual(ssA.R(:,:,1), [1; 0; 1]);
      testCase.verifyEqual(ssA.R(:,:,2), [1; 0; 1/2]);
      testCase.verifyEqual(ssA.R(:,:,3), [1; 0; 1/3]);
    end
    
    function testMultipleObsAccum(testCase)
      % Test that accumlators in the first and third observations work      
      Z = [1 2; 3 4; 5 6];
      H = diag(ones(3,1));
      
      T = diag([.99 -.1]);
      Q = diag([1 1]);
      
      ss = StateSpace(Z, H, T, Q);
      % Generate data for y and aggregate the first and 3rd series
      y = generateData(ss, 300);
      y(1,:) = reshape([nan(100,2) sum(reshape(y(1,:), [3 100]))']', [], 1)';
      y(3,:) = reshape([nan(50,5) sum(reshape(y(3,:)', [6 50]))']', [], 1)';
      
      accum = Accumulator.GenerateRegular(y', {'sum', '', 'sum'}, [3 1 6]);
      ssA = accum.augmentStateSpace(ss);
      
      % Sizes
      testCase.verifySize(ssA.Z, [3 6]);
      testCase.verifySize(ssA.T, [6 6 3]);
      testCase.verifySize(ssA.c, [6 1]);
      testCase.verifySize(ssA.R, [6 2]);
      
      % Make sure Z elements got moved
      testCase.verifyEqual(ssA.Z(1, 3:4), ss.Z(1,:));
      testCase.verifyEqual(ssA.Z(2, 1:2), ss.Z(2,:));
      testCase.verifyEqual(ssA.Z(3, 5:6), ss.Z(3,:));
      
      % T matrix
      testCase.verifyEqual(ssA.T(1:2, 1:2, :), repmat(ss.T, [1 1 3]));
      testCase.verifyEqual(ssA.T(3:4, 1:2, :), repmat(ss.T, [1 1 3]));
      testCase.verifyEqual(ssA.T(5:6, 1:2, :), repmat(ss.T, [1 1 3]));
      % Make sure s_t is correct
      testCase.verifyEqual(ssA.T(3:4, 3:4, 1:2), zeros(2,2,2));
      testCase.verifyEqual(ssA.T(3:4, 3:4, 3), eye(2));
      testCase.verifyEqual(ssA.T(5:6, 5:6, 1), zeros(2));
      testCase.verifyEqual(ssA.T(5:6, 5:6, 2:3), repmat(eye(2), [1 1 2]));
      
      % c vector 
      testCase.verifyEqual(ssA.c, zeros(6,1));

      % R matrix
      testCase.verifyEqual(ssA.R, repmat(ss.R, [3 1]));
    end
    
    function testSeparatedAccumulators(testCase)
      % Test that accumlators in the first and third observations work      
      Z = [0 1; .5 1; 1 1];
      H = diag(ones(3,1));
      
      T = diag([.99 -.1]);
      Q = diag([1 1]);
      
      ssGen = StateSpace(Z, H, T, Q);
      % Generate data for y and aggregate the first and 3rd series
      y = generateData(ssGen, 300);
      y(1,:) = reshape([nan(100,2) mean(reshape(y(1,:), [100 3]), 2)]', [], 1)';
      y(3,:) = reshape([nan(100,2) mean(reshape(y(3,:), [100 3]), 2)]', [], 1)';
      
      accum = Accumulator.GenerateRegular(y', {'avg', '', 'avg'}, [1 1 3]);
      ssA = accum.augmentStateSpace(ssGen);
      
      % Sizes
      testCase.verifySize(ssA.Z, [3 7]);
      testCase.verifySize(ssA.T, [7 7 3]);
      testCase.verifySize(ssA.c, [7 3]);
      testCase.verifySize(ssA.R, [7 2 3]);
      
      % Make sure Z elements got moved
      newZ = zeros(size(ssGen.Z,1), 7);
      newZ(1,5) = ssGen.Z(1,2);
      newZ(2,1:2) = ssGen.Z(2,1:2);
      newZ(3,6:7) = ssGen.Z(3,1:2);
      testCase.verifyEqual(ssA.Z, newZ);
      
      % T Matrix - Make sure 0 and 1 elements in the right places
      testCase.verifyEqual(ssA.T(1:2, 1:2, :), repmat(ssGen.T, [1 1 3]));
      testCase.verifyEqual(ssA.T(3:4, 1:2, :), repmat(eye(2), [1 1 3]));
      testCase.verifyEqual(ssA.T(5, 1:2, :), ...
        repmat(ssGen.T(2,1:2), [1 1 3]) ./ reshape(1:3, [1 1 3]));
      testCase.verifyEqual(ssA.T(5, 3:4, :), zeros(1, 2, 3))
      testCase.verifyEqual(ssA.T(5, 5, :), reshape((0:2) ./ (1:3), [1 1 3]));
      
      testCase.verifyEqual(ssA.T(6:7, 1:2, :), ...
        (repmat(ssGen.T, [1 1 3]) + eye(2)) ./ reshape(1:3, [1 1 3]));
      testCase.verifyEqual(ssA.T(6:7, 3:4, :), eye(2) ./ reshape(1:3, [1 1 3]));
      testCase.verifyEqual(ssA.T(6, 6, :), reshape((0:2) ./ (1:3), [1 1 3]));
      testCase.verifyEqual(ssA.T(7, 7, :), reshape((0:2) ./ (1:3), [1 1 3]));

      % c vector 
      testCase.verifyEqual(ssA.c, zeros(7,3));
      % R matrix
      testCase.verifyEqual(ssA.R(1:2,1:2,:), repmat(ssGen.R, [1 1 3]));
      testCase.verifyEqual(ssA.R(5,1:2,:), repmat(ssGen.R(2,:), [1 1 3]) ./ reshape(1:3, [1 1 3]))
      testCase.verifyEqual(ssA.R(6:7,1:2,:), repmat(ssGen.R, [1 1 3]) ./ reshape(1:3, [1 1 3]))
    end
    
    function testAddLagSumAndAvg(testCase)
      nSeries = 3;
      nLags = 2;
      
      % One series that's a annual sum, one that's a quarterly average
      y = [randn(264, 1), reshape([randn(1,22); nan(11,22)], [], 1) ...
        reshape([randn(1,88); nan(2,88)], [], 1)] ;
      accum = Accumulator.GenerateRegular(y, {'', 'avg', 'avg'}, [0 12 3]);
      
      % Factor model with AR(1) errors 
      Z = [ones(nSeries,1) zeros(nSeries,1) eye(nSeries)];
      H = eye(nSeries);
      
      T = blkdiag([.6 ./ nLags * ones(1, nLags); 1 0], .4 * eye(nSeries));
      R = blkdiag([1; zeros(2-1, 1)], eye(nSeries));
      Q = blkdiag(1, .3 * eye(nSeries));
      
      ss0 = StateSpace(Z, H, T, Q, 'R', R);
      ss0A = accum.augmentStateSpace(ss0);
      
      [~, ll] = ss0A.filter(y);
      testCase.verifyThat(ll, matlab.unittest.constraints.IsReal);
    end
    
    
    %% Tests that accumlated states equal the data
    function testSumSmoother(testCase)
      % Test that when a low-frequency series has no measurement error that the
      % smoothed accumulator variable for that observation matches the observation. 
      p = 2; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      ssGen.Z(2,1) = 1;
      ssGen.H = blkdiag(1, 0);
      
      Y = generateData(ssGen, timeDim)';
      Y(:, 2) = Accumulator_test.aggregateY(Y(:, 2), 3, 'sum');
      
      accum = Accumulator.GenerateRegular(Y, {'', 'sum'}, [1 3]);
      ssA = accum.augmentStateSpace(ssGen);
      
      % Make sure we can run the filter
      [~, ll] = ssA.filter(Y');
      testCase.verifyThat(ll, matlab.unittest.constraints.IsFinite)
      
      % Test smoother
      alpha = ssA.smooth(Y);
      testCase.verifyEqual(alpha(3:3:end,3), Y(3:3:end,2), 'AbsTol', 1e-10);
    end
    
    function testAvgSmoother(testCase)
      % Test that when a low-frequency series has no measurement error that the
      % smoothed accumulator variable for that observation matches the observation. 
      p = 2; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.Z(2,1) = 1;
      ssGen.H = diag([1 0]);
      ssGen.T(1,:) = [0.5 0.3];
      
      Y = generateData(ssGen, timeDim)';
      
      aggY = Y;
      aggY(:, 2) = Accumulator_test.aggregateY(Y(:, 2), 3, 'avg');
      
      accum = Accumulator.GenerateRegular(aggY, {'', 'avg'}, [1 3]);
      ssA = accum.augmentStateSpace(ssGen);
      
      % Make sure we can run the filter
      [~, ll] = ssA.filter(Y');
      testCase.verifyThat(ll, matlab.unittest.constraints.IsFinite)
      
      % Test smoother
      alpha = ssA.smooth(aggY);
      testCase.verifyEqual(alpha(3:3:end,end), aggY(3:3:end,2), 'AbsTol', 1e-10);
      % We can only test the sample where we have full information to average over
      triangleAvg = filter([1 2 3 2 1], 1, alpha(:,1)) ./ 3;
      testCase.verifyEqual(alpha(6:3:end,3), triangleAvg(6:3:end), 'AbsTol', 1e-10);
    end
    
    function testCommonStateProcess(testCase)
      p = 7; timeDim = 205;
      m1 = generateARmodel(p, 1, true);
      m1.T(1,:) = [0.85 0.05];
      m2 = generateARmodel(p, 2, true);
      m2.T(1,:) = [0.85 0.05 -0.2];
      m = m1.m + m2.m;
      ssGen = StateSpace([m1.Z m2.Z], m1.H, ...
        blkdiag(m1.T, m2.T), blkdiag(m1.Q, m2.Q), 'R',  blkdiag(m1.R, m2.R));
      ssGen.Z(6,[1 3]) = [1 1];
      ssGen.H(6,6) = 0;
      
      Y = generateData(ssGen, timeDim);
      
      aggY = Y';
      aggY(:, 3) = Accumulator_test.aggregateY(Y(3, :)', 3, 'sum');
      aggY(:, 6) = Accumulator_test.aggregateY(Y(4, :)', 3, 'avg');
      
      accum = Accumulator.GenerateRegular(aggY, ...
        {'', '', 'sum', '', '', 'avg', ''}, [1 1 3 1 1 3 1]);
      ssA = accum.augmentStateSpace(ssGen);
      
      % Make sure we can run the filter
      [~, ll] = ssA.filter(Y');
      testCase.verifyThat(ll, matlab.unittest.constraints.IsFinite)
      
      % Test smoother
      alpha = ssA.smooth(aggY);
      testCase.verifyEqual(sum(alpha(3:3:end, end-1:end), 2), ...
        aggY(3:3:end,6), 'AbsTol', 1e-10);
    end
    
    % Not sure what this is doing here:
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
    
    function testNotOneStart(testCase)
      p = 2; m = 0; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = 0.5; 
      ssGen.Z(:,1) = [1; 1];
      ssGen.H = diag([1 0]);

      Y = generateData(ssGen, timeDim)';
      aggY = Accumulator_test.aggregateY(Y(:, 2), 3, 'sum');
      Y(:, 2) = nan;
      Y(4:3:end, 2) = aggY(3:3:end-1);
      
      accum = Accumulator.GenerateRegular(Y, {'', 'avg'}, [1 3]);
      ssA = accum.augmentStateSpace(ssGen);
      
      % Make sure we can run the filter
      [~, ll] = ssA.filter(Y');
      testCase.verifyThat(ll, matlab.unittest.constraints.IsFinite)
      
      % Test smoother
      alpha = ssA.smooth(Y);
      testCase.verifyEqual(alpha(4:3:end,end), Y(4:3:end,2), 'AbsTol', 1e-10);
      
      % We can only test starting when we have enough high-frequency obs to aggregate
      triangleAvg = filter([1 2 3 2 1], 1, alpha(:,1)) ./ 3;
      testCase.verifyEqual(alpha(7:3:end,3), triangleAvg(7:3:end), 'AbsTol', 1e-10);
    end
    

    %% ThetaMap tests
    function testThetaMapAR(testCase)
      p = 2; m = 1; timeDim = 599;
      ssGen = generateARmodel(p, m, false);
      ssGen.T(1,:) = [0.5 0.3];
      
      Y = generateData(ssGen, timeDim)';
      Y(:, 2) = Accumulator_test.aggregateY(Y(:, 2), 3, 'sum');
      
      accum = Accumulator.GenerateRegular(Y, {'', 'sum'}, [1 3]);
      
      ssA = accum.augmentStateSpace(ssGen);
      
      TssE = ssGen.T;
      TssE(1,:) = [nan nan];
      ssE = StateSpaceEstimation(ssGen.Z, ssGen.H, TssE, ssGen.Q, 'R', ssGen.R);
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
      Y(:, 2) = Accumulator_test.aggregateY(Y(:, 2), 3, 'avg');
      accum = Accumulator.GenerateRegular(Y, {'', 'avg'}, [1 3]);
      
      % Create augmented StateSpace
      ssA = accum.augmentStateSpace(ssGen);
      
      % Generate ThetaMap
      tm = ThetaMap.ThetaMapAll(ssGen);
      tm.index.Z(:, 2) = 0;
      tm.index.T(2, :) = 0;
      tm.fixed.T(2, 1) = 1;
      tm.index.c(2) = 0;
      tm.index.R(2, 1) = 0;
      tm = tm.validateThetaMap();
      
      tm2 = accum.augmentThetaMap(tm);
      
      testCase.verifyEqual(tm2.nTheta, tm.nTheta);
      
      % Test theta -> system
      thetaGen = tm.system2theta(ssGen);
      
      ssTestAug = tm2.theta2system(thetaGen);
      testCase.verifyEqual(ssTestAug.Z, ssA.Z, 'AbsTol', 1e-14);
      testCase.verifyEqual(ssTestAug.d, ssA.d, 'AbsTol', 1e-14);
      testCase.verifyEqual(ssTestAug.H, ssA.H, 'AbsTol', 1e-14);
      testCase.verifyEqual(ssTestAug.T, ssA.T, 'AbsTol', 1e-14);
      testCase.verifyEqual(ssTestAug.c, ssA.c, 'AbsTol', 1e-14);
      testCase.verifyEqual(ssTestAug.R, ssA.R, 'AbsTol', 1e-14);
      testCase.verifyEqual(ssTestAug.Q, ssA.Q, 'AbsTol', 1e-14);
      
      % Test system -> theta
      thetaTestAug = tm2.system2theta(ssTestAug);
      testCase.verifyEqual(thetaGen, thetaTestAug);
      thetaAug = tm2.system2theta(ssA);
      testCase.verifyEqual(thetaGen, thetaAug);
    end
    
    function testThetaMapSym(testCase)
      syms lambda rho sigmaKappa sigmaZeta a
      Z = [1, 0, a, 0];
      H = 0;
      T = blkdiag([1 1; 0 1], rho .* [cos(lambda), sin(lambda); -sin(lambda) cos(lambda)]);
      R = [zeros(1, 3); eye(3)];
      Q = diag([sigmaZeta; sigmaKappa; sigmaKappa]);
      ssE = StateSpaceEstimation(Z, H, T, Q, 'R', R);
      
      accum = Accumulator(1, [repmat((1:3)', [100 1]); 1], repmat(3, [301 1]));
      ssEA = accum.augmentStateSpaceEstimation(ssE);

      tm1 = ssE.ThetaMapping;
      tm2 = ssEA.ThetaMapping; 
      testCase.verifyEqual(tm2.nTheta, tm1.nTheta);

      % Test theta -> system
      rng(0);
      thetaTest = rand(tm1.nTheta, 1);
      ssTestAug = tm2.theta2system(thetaTest);
      thetaTestAug = tm2.system2theta(ssTestAug);
      testCase.verifyEqual(thetaTest, thetaTestAug, 'AbsTol', 1e-6);
      
      % Test augmenting system after the fact
      ss1 = tm1.theta2system(thetaTest);
      ssNewAug = accum.augmentStateSpace(ss1);
      testCase.verifyEqual(ssTestAug, ssNewAug);
    end
    
    function testOutOfOrder(testCase)
      % Set up a model with sum and average accumulator out of order, make sure it gets
      % set up in the reverse order. 
      
      ssGen = generateARmodel(3, 2, false);
      
      % Series 3 should have an average accumulator. Series 2 should have a sum.
      accum = Accumulator([3 2], [repmat((1:3)', [40 1]) repmat([0; 1; 1], [40 1])], ...
        repmat(3, [120 2]));
      % Because they get ordered according to how the observations are ordered, state 4
      % should be the sum accumulator, state 5 should be the average accumulator. 
      ssAug = accum.augmentStateSpace(ssGen);
      
      testCase.verifyEqual(reshape(ssAug.T(4,1:3,:), [3 3])', repmat(ssGen.T(1,1:3), [3 1]));
      testCase.verifyEqual(squeeze(ssAug.T(4,4,:)), [0; 1; 1]);
      
      testCase.verifyEqual(reshape(ssAug.T(5,1:3,:), [3 3])', ...
        (ssGen.T(1,1:3) + [1 1 0]) ./ repmat([1; 2; 3], [1 3]), 'AbsTol', 1e-14);
      testCase.verifyEqual(squeeze(ssAug.T(5,5,:)), [0; 0.5; 2/3]);      
    end
    
  end
end