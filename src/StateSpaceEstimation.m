classdef StateSpaceEstimation < AbstractStateSpace
  % Maximum likelihood estimation of parameters 
  %
  % Estimate parameter values of a state space system. Indicate values to
  % be estimated by setting them to nan or with symbolic variables. 
  
  % David Kelley, 2016-2017
  
  properties
    % Mapping from theta vector to parameters
    ThetaMapping      
    
    % Screen output during ML estimation
    verbose = false;
    % Show window during estimation
    diagnosticPlot = true;
    
    % Solver options
    solver = {'fmincon', 'fminsearch'};
    solveIterMax = 20;
    % Tolerance for each solver 
    stepTol = 1e-8;     
    % Tolerance for improvement between solvers
    solveTol = 1e-10;   
    
    % Compute gradient internally (or else let fminunc/fmincon do so)
    useInternalNumericGrad = true;
    
    % Function handle to constraints on theta
    constraints
    fminsearchMaxIter = 500;
    
    % Allowable flags in estimation
    flagsAllowed = -1:5;
    
    % Number of attempts at random initialization
    initializeAttempts = 20;
    
    % Range to attempt initialization over
    initializeRange = [-2 2];
  end
  
  properties (Dependent)
    a0 
    P0
  end
  
  methods
    %% Setter/getter methods for initial values
    function a0 = get.a0(obj)
      a0 = obj.a0Private;
    end
    
    function obj = set.a0(obj, newa0)
      obj.a0Private = newa0;
      obj.ThetaMapping = obj.ThetaMapping.updateInitial(newa0, obj.P0Private);
    end
    
    function P0 = get.P0(obj)
      P0 = obj.P0Private;
    end
    
    function obj = set.P0(obj, newP0)
      obj.P0Private = newP0;
      obj.ThetaMapping = obj.ThetaMapping.updateInitial(obj.a0Private, newP0);      
    end
    
  end
  
  methods
    %% Constructor
    function obj = StateSpaceEstimation(Z, H, T, Q, varargin)
      % Constructor 
      % 
      % Arguments: 
      %   Z, H, T Q (double): state space parameters
      % Optional arguments (name-value pairs): 
      %   d, beta, c, gamma, R (double) state space parameters
      %   a0, P0 (double): initial state values
      %   LowerBound, UpperBound (StateSpace): bounds on parameters
      %   ThetaMap (ThetaMap): mapping from parameter vector to state space parameters
      % Output: 
      %   obj (StateSpaceEstimation): estimation object
      
      inP = inputParser;
      inP.addParameter('d', []);
      inP.addParameter('beta', []);
      inP.addParameter('c', []);
      inP.addParameter('gamma', []);
      inP.addParameter('R', []);
      inP.addParameter('a0', [], @isnumeric);
      inP.addParameter('P0', [], @isnumeric);
      inP.addParameter('LowerBound', [], @(x) isa(x, 'StateSpace'));
      inP.addParameter('UpperBound', [], @(x) isa(x, 'StateSpace'));
      inP.addParameter('ThetaMap', [], @(x) isa(x, 'ThetaMap'));
      inP.parse(varargin{:});
      inOpts = inP.Results;

      obj = obj@AbstractStateSpace(Z, inOpts.d, inOpts.beta, H, ...
        T, inOpts.c, inOpts.gamma, inOpts.R, Q);

      % Initial ThetaMap generation - will be augmented to include restrictions
      if ~isempty(inOpts.ThetaMap)
        obj.ThetaMapping = inOpts.ThetaMap;
      else
        obj.ThetaMapping = ThetaMap.ThetaMapEstimation(obj);
      end
      
      % Initial values - If initial values are not passed they will be set as 
      % the default values at each iteration of the estimation, effectively 
      % making them a function of other parameters. 
      if ~isempty(inOpts.a0) && ~isempty(inOpts.P0)
        obj.a0 = inOpts.a0;
      end
      if ~isempty(inOpts.P0)
        obj.P0 = inOpts.P0;
      end
      
      % Add bounds
      obj.ThetaMapping = obj.ThetaMapping.addRestrictions(inOpts.LowerBound, inOpts.UpperBound);
    end
    
    %% Estimation methods
    function [ss, diagnostic, thetaHat, gradient] = estimate(obj, y, ss0, x, w)
      % Estimate missing parameter values via maximum likelihood.
      %
      % Arguments: 
      %     y (double): observe data
      % Optional arguments (additional inputs): 
      %     ss0 (StateSpace or double): StateSpace or theta vector of initial values
      %     x (double): exogenous measurement equation data
      %     w (double): exogenous state equation data
      % Outputs: 
      %     ss (StateSpace): estimated StateSpace
      %     diagnostic (structure): structure containing diagnostics on estimation
      %     thetaHat (double): estimated parameter vector
      %     gradient (double): gradient of the likelihood at estimated theta
      
      if nargin < 4
        x = [];
      end
      if nargin < 5
        w = [];
      end
      
      [obj, y, x, w] = obj.checkSample(y, x, w);
        
      firstSwarm = isequal(obj.solver, 'swarm') || ...
        (iscell(obj.solver) &&  isequal(obj.solver{1}, 'swarm'));
      
      if nargin < 3 || isempty(ss0) && ~firstSwarm
        [theta0U, theta0, ss0] = obj.initializeRandom(y, x, w);
      elseif firstSwarm
        theta0U = [];
      else
        % Initialization
        if isa(ss0, 'StateSpace')
          obj.checkConformingSystem(ss0);
          theta0 = obj.ThetaMapping.system2theta(ss0);
        else
          theta0 = ss0;
          validateattributes(theta0, {'numeric'}, ...
            {'vector', 'numel', obj.ThetaMapping.nTheta}, 'estimate', 'theta0');
          ss0 = obj.ThetaMapping.theta2system(theta0);
        end
        assert(all(isfinite(theta0)), 'Non-finite values in starting point.');
        
        theta0U = obj.ThetaMapping.unrestrictTheta(theta0);
      end
        
      progress = EstimationProgress(theta0, obj.diagnosticPlot, obj.m, ss0);
      outputFcn = @(thetaU, oVals, st) ...
          progress.update(obj.ThetaMapping.restrictTheta(thetaU), oVals);
        
      function [stop, optOpts, deltaOpt] = outputFcnSimanneal(optOpts, oVals, ~)
        if nargin >=1
          stop = progress.update(obj.ThetaMapping.restrictTheta(oVals.x), oVals);
        else
          stop = false;
        end        
        deltaOpt = false;
      end
      function [stop, optOpts, deltaOpt] = outputFcnSwarm(optOpts, ~)
        if nargin >=1
          oVals = struct('fval', optOpts.bestfval);
          stop = progress.update(obj.ThetaMapping.restrictTheta(optOpts.bestx), oVals);
        else
          stop = false;
        end        
        deltaOpt = false;
      end
                        
      assert(isnumeric(y), 'y must be numeric.');
      assert(isnumeric(x), 'x must be numeric.');
      assert(isnumeric(w), 'w must be numeric.');
      assert(isa(ss0, 'StateSpace') || isnumeric(ss0));
      assert(obj.ThetaMapping.nTheta > 0, 'All parameters known. Unable to estimate.');
      
      % Run fminunc/fmincon
      nonlconFn = @obj.nlConstraintFun;
      solverFun = obj.solver;

      if obj.verbose
        displayType = 'iter-detailed';
      else
        displayType = 'none';
      end
      
      % Optimizer options
      optFMinCon = optimoptions(@fmincon, ...
        'Algorithm', 'interior-point', ...
        'SpecifyObjectiveGradient', obj.useInternalNumericGrad, ...
        'UseParallel', obj.useParallel && ~obj.useInternalNumericGrad, ...
        'Display', displayType, ...
        'MaxFunctionEvaluations', 50000, ...
        'MaxIterations', 1000, ...
        'FunctionTolerance', obj.stepTol, ...
        'OptimalityTolerance', obj.stepTol, ...
        'StepTolerance', obj.stepTol, ...
        'TolCon', 0, ...
        'OutputFcn', outputFcn);
      
      optFMinUnc = optimoptions(@fminunc, ...
        'Algorithm', 'quasi-newton', ...
        'SpecifyObjectiveGradient', obj.useInternalNumericGrad, ...
        'UseParallel', obj.useParallel && ~obj.useInternalNumericGrad, ...
        'Display', displayType, ...
        'MaxFunctionEvaluations', 50000, ...
        'MaxIterations', 1000, ...
        'FunctionTolerance', obj.stepTol, ...
        'OptimalityTolerance', obj.stepTol, ...
        'StepTolerance', obj.stepTol, ...
        'OutputFcn', outputFcn);
      
      optSimulanneal = optimoptions(@simulannealbnd, ...
        'OutputFcn', @outputFcnSimanneal);
      
      optSwarm = optimoptions(@particleswarm, ...
        'OutputFcn', @outputFcnSwarm, ...
        'InitialSwarmSpan', 4, ...
        'UseParallel', obj.useParallel);
      
      if all(strcmpi(obj.solver, 'fminsearch')) && ...
          (ischar(obj.fminsearchMaxIter) && ...
          strcmpi(obj.fminsearchMaxIter, 'default'))
        searchMaxIter = 200 * obj.ThetaMapping.nTheta;
      else
        searchMaxIter = obj.fminsearchMaxIter;
      end      
      optFMinSearch = optimset('Display', displayType, ...
        'MaxFunEvals', 5000 * obj.ThetaMapping.nTheta, ...
        'MaxIter', searchMaxIter, ...
        'OutputFcn', outputFcn);
      
      % Loop over optimizers 
      loopContinue = true;
      iter = 0; logli = []; 
      warning off MATLAB:nearlySingularMatrix;
      while loopContinue
        iter = iter + 1;
        logli0 = logli;
        if iscell(obj.solver)
          solverFun = obj.solver{mod(iter+1, length(obj.solver))+1};
        end
        
        switch solverFun
          case 'fminunc'
            minfunc = @(thetaU) obj.minimizeFun(thetaU, y, x, w, progress, obj.useInternalNumericGrad);
            [thetaUHat, logli, outflag, ~, gradient] = fminunc(...
              minfunc, theta0U, optFMinUnc);
          case 'fmincon'
            minfunc = @(thetaU) obj.minimizeFun(thetaU, y, x, w, progress, obj.useInternalNumericGrad);
            try
              [thetaUHat, logli, outflag, ~, ~, gradient] = fmincon(...
                minfunc, theta0U, [], [], [], [], [], [], nonlconFn, optFMinCon);
            catch ex
              switch ex.identifier
                case 'optim:barrier:GradUndefAtX0'
                  if iter > 1
                    warning('StateSpaceEstimation:estimate:badInitialGrad', ...
                      ['Gradient contains Inf, NaN, or complex values. ' ...
                      'Returning previous solver output']);
                  else
                    rethrow(ex);
                  end
                case 'optim:barrier:DerivUndefAtX0'
                  if iter == 1
                    rethrow(ex);
                  else
                    warning('Unable to evaluate derivative. Optimization cannot continue.')
                  end
                otherwise
                  rethrow(ex);
              end
            end
          case 'fminsearch'
            minfunc = @(thetaU) obj.minimizeFun(thetaU, y, x, w, progress, false);
            
            [thetaUHat, logli, outflag] = fminsearch(...
              minfunc, theta0U, optFMinSearch);
            
            gradient = [];
          case 'sa'
            minfunc = @(thetaU) obj.minimizeFun(thetaU, y, x, w, progress, false);
            
            [thetaUHat, logli, outflag] = simulannealbnd(...
              minfunc, theta0U, [],  [], optSimulanneal);
            
            gradient = [];
            
          case 'swarm'
            minfunc = @(thetaU) obj.minimizeFun(thetaU, y, x, w, progress, false);
            
            [thetaUHat, logli, outflag] = particleswarm(...
              minfunc, obj.ThetaMapping.nTheta, [],  [], optSwarm);
            
            gradient = [];
          otherwise
            error('Unknown solver.');
        end
        
        progress.nextSolver();
        stopestim = strcmpi(progress.stopStatus, 'stopestim');
        
        assert(~isempty(intersect(obj.flagsAllowed, outflag)), ...
          'Optimizer returned disallowed flag %d.', outflag);
        
        loopContinue = ~stopestim && iter < obj.solveIterMax && ...
          (iter <= 2 || abs(logli0 - logli) > obj.solveTol);
        
        if logli0 < logli - 1e-12
          warning('StateSpaceEstimation:estimate:solverDecrease', ...
            ['Solver decreased likelihood by %g. \n' ...
            'Returning higher likelihood solution.'], logli - logli0);
          diagnostic = progress.diagnostics();

          thetaUHat = obj.ThetaMapping.unrestrictTheta(diagnostic.thetaHist(...
            find(-diagnostic.likelihoodHist == logli0, 1, 'last'), :)');
          assert(~isempty(thetaUHat));
          logli = logli0;
          gradient = [];
          loopContinue = false;
        end
        
        assert(~any(isnan(thetaUHat)), 'Estimation Error');
        
        theta0U = thetaUHat;
      end
      warning on MATLAB:nearlySingularMatrix;
  
      % Save estimated system to current object
      thetaHat = obj.ThetaMapping.restrictTheta(thetaUHat);
      ss = obj.ThetaMapping.theta2system(thetaHat);
      
      % Run smoother, plot smoothed state
      if obj.diagnosticPlot
        progress.alpha = ss.smooth(y, x, w);
        if progress.visible && isvalid(progress.figHandle) && ...
            strcmpi(progress.updateStatus, 'active')     
          progress.updateFigure();
        end
      end
      
      % Get diagnostic info on estimation progress
      diagnostic = progress.diagnostics();
    end
  end
  
  methods (Hidden = true)
    %% Maximum likelihood estimation helper methods
    function [negLogli, gradient] = minimizeFun(obj, thetaU, y, x, w, progress, calcGrad)
      % Get the likelihood of thetaU
      
      theta = obj.ThetaMapping.restrictTheta(thetaU);
      ss1 = obj.ThetaMapping.theta2system(theta);
      ss1 = ss1.setDefaultInitial();
      
      if any(diag(ss1.H) < 0)
        error('Negative variance');
      end
      
      % Really enforce constraints
      if any(obj.nlConstraintFun(thetaU) > 0)
        % warning('Constraint violated in solver.');
        negLogli = 1e30 * max(obj.nlConstraintFun(thetaU));
        gradient = nan(size(theta));
        return
      end
      
      % Avoid bad parameters
      ss1Vectorized = obj.ThetaMapping.vectorizeStateSpace(ss1, ...
          ~obj.ThetaMapping.usingDefaulta0, ~obj.ThetaMapping.usingDefaultP0);
        
      if any(~isfinite(ss1Vectorized))
        negLogli = nan;
        gradient = nan;
        return
      end
      
      if calcGrad && nargout == 1
        warning('Calculating unused gradient!');
      end
      
      try
        if calcGrad && nargout > 1
          progress.totalEvaluations = progress.totalEvaluations + ...
            1 + (2 * obj.numericGradPrec * obj.ThetaMapping.nTheta * obj.useInternalNumericGrad);

           % Calculate likelihood and gradient
          [rawLogli, thetaGradient, fOut] = ss1.gradient(y, x, w, obj.ThetaMapping, theta);

          GthetaUtheta = obj.ThetaMapping.thetaUthetaGrad(thetaU);
          rawGradient = GthetaUtheta * thetaGradient;
        else
          progress.totalEvaluations = progress.totalEvaluations + 1;

          [~, rawLogli, fOut] = ss1.filter(y, x, w);
          rawGradient = [];          
        end
        
        % Don't plot the diffuse parts of the state because they look odd
        a = fOut.a;
        a(:,1:fOut.dt) = nan;
        
        if ~isnan(rawLogli) && imag(rawLogli) ~= 0
          % Imaginary logliklihood, throw to catch condition
          error('StateSpaceEstimation:imaginaryLL', 'Imaginary likelihood evaluation.');
        end
        
        % Put filtered state in figure for plotting
        progress.a = a;  
        progress.ss = ss1;
        
      catch 
        % Set output to nan to indicate the likelihood evaluation failed
        rawLogli = nan;
        rawGradient = nan(obj.ThetaMapping.nTheta, 1);
        
        progress.nanIterations = progress.nanIterations + 1;
      end
      
      negLogli = -rawLogli;
      gradient = -rawGradient;
      
      % Allow callbacks to process in EstimationProgress:
      drawnow;
    end
    
    function [cx, ceqx] = nlConstraintFun(obj, thetaU)
      % Constraints of the form c(x) <= 0 and ceq(x) = 0.
      scale = 1e6;
      theta = obj.ThetaMapping.restrictTheta(thetaU);

      % User constraints
      cx = [];
      if ~isempty(obj.constraints)
        if ~iscell(obj.constraints)
          obj.constraints = {obj.constraints};
        end
        
        for iC = 1:length(obj.constraints)
          cx = [cx; obj.constraints{iC}(theta)]; %#ok<AGROW>
        end
      end
      
      % Return the negative determinants in cx
      ss1 = obj.ThetaMapping.theta2system(theta);
      if ~obj.ThetaMapping.usingDefaultP0
        cx = [cx; scale * -det(ss1.Q0)];
      end         
      ceqx = 0;
      
      % Constrain a submatrix of T to be stationary: 
      % c(x) should be abs(max(eig(T))) - 1
      if ~isempty(obj.stationaryStates)
        eigs = eig(ss1.T(obj.stationaryStates,obj.stationaryStates,obj.tau.T(1)));
        stationaryCx = scale * (max(abs(eigs)) - 1);
        cx = [cx; stationaryCx];
      end
      
      if any(isnan(cx))
        warning('Nan constraint.');
      end
    end
    
    function [theta0U, theta0, ss0] = initializeRandom(obj, y, x, w)
      % Generate default initialization
      
      % The default initialization
      iAttempt = 0; 
      iLogli = nan(obj.initializeAttempts, 1); 
      iTheta0U = nan(obj.ThetaMapping.nTheta, obj.initializeAttempts);
      
      for iAttempt = 1:obj.initializeAttempts
        iTheta0U(:,iAttempt) = obj.initializeRange(1) + rand(obj.ThetaMapping.nTheta, 1) * ...
          (obj.initializeRange(2) - obj.initializeRange(1));
        
        try
          theta0 = obj.ThetaMapping.restrictTheta(iTheta0U(:,iAttempt));
          ss0 = obj.ThetaMapping.theta2system(theta0);
          [~, iLogli(iAttempt)] = ss0.filter(y, x, w);
        catch
        end
      end
      
      % Find the best starting point with a valid gradient
      iLogli(~isfinite(iLogli)) = -Inf;
      [~, thetaOrder] = sort(iLogli, 'descend');
      for iGradAtt = 1:sum(~isnan(iLogli))
        theta0U = iTheta0U(:, thetaOrder(iGradAtt));
        theta0 = obj.ThetaMapping.restrictTheta(iTheta0U(:,thetaOrder(iGradAtt)));
        ss0 = obj.ThetaMapping.theta2system(theta0);
        try
          [ll0, grad0] = ss0.gradient(y, x, w, obj.ThetaMapping, theta0);
        catch ex
          if strcmpi('MATLAB:UndefinedFunction', ex.identifier)
            rethrow(ex)
          else
            continue
          end          
        end
        
        if isfinite(ll0) && all(isfinite(grad0))
          break
        end
      end
      
      % Check that we got a good starting point
      if (~isfinite(ll0) || any(~isfinite(grad0))) && iAttempt == obj.initializeAttempts
        error('Could not find initialization. Please specify valid starting point.');
      end
      
    end
  end
  
end
