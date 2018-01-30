classdef StateSpaceEstimation < AbstractStateSpace
  % Maximum likelihood estimation of parameters 
  % -------------------------------------------
  %   ss_estimated = ss.estimate(y, ss0)
  %
  % Estimate parameter values of a state space system. Indicate values to
  % be estimated by setting them to nan. An initial set of parameters must
  % be provided in ss0.
  %
  % Pseudo maximum likelihood estimation
  % ------------------------------------
  %   ss_estimated = ss.em_estimate(y, ss0)
  %
  % Initial work on implementing a general EM algorithm.
  
  % David Kelley, 2016-2017
  
  properties
    % Screen output during ML estimation
    verbose = false;
    diagnosticPlot = true;
    
    solver = {'fmincon', 'fminsearch'};
    solveIterMax = 20;
    
    % ML-estimation tolerances
    tol = 1e-10;      % Final estimation tolerance
    stepTol = 1e-8;  % Step-size tolerance for ML theta elements
    solveTol = 1e-11;   % EM tolerance
    
    % ML Estimation parameters
    ThetaMapping      % Mapping from theta vector to parameters
    
    % Function handle to constraints on theta
    constraints
    
    fminsearchMaxIter = 500;
    
    % Indicator for use of analytic gradient
    useAnalyticGrad = false;

    % Indicator to use more accurate, slower gradient
    useInternalNumericGrad = true;
    
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
      obj.ThetaMapping = obj.ThetaMapping.updateInitial(newa0, []);
    end
    
    function P0 = get.P0(obj)
      P0 = obj.P0Private;
    end
    
    function obj = set.P0(obj, newP0)
      obj.P0Private = newP0;
      obj.ThetaMapping = obj.ThetaMapping.updateInitial([], newP0);      
    end
    
    function obj = set.useAnalyticGrad(obj, newGrad)
      obj.useAnalyticGrad = newGrad;
      if ~isempty(obj.ThetaMapping) %#ok<MCSUP>
        obj.ThetaMapping.useAnalyticGrad = newGrad; %#ok<MCSUP>
      end
    end
  end
  
  methods
    %% Constructor
    function obj = StateSpaceEstimation(Z, d, H, T, c, R, Q, varargin)
      obj = obj@AbstractStateSpace(Z, d, H, T, c, R, Q);
      
      inP = inputParser;
      inP.addParameter('a0', [], @isnumeric);
      inP.addParameter('P0', [], @isnumeric);
      inP.addParameter('LowerBound', [], @(x) isa(x, 'StateSpace'));
      inP.addParameter('UpperBound', [], @(x) isa(x, 'StateSpace'));
      inP.addParameter('ThetaMap', [], @(x) isa(x, 'ThetaMap'));
      inP.parse(varargin{:});
      inOpts = inP.Results;

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
    function [ss_out, diagnostic, gradient] = estimate(obj, y, ss0)
      % Estimate missing parameter values via maximum likelihood.
      %
      % ss = ss.estimate(y, ss0) estimates any missing parameters in ss via
      % maximum likelihood on the data y using ss0 as an initialization.
      %
      % ss.estimate(y, ss0, a0, P0) uses the initial values a0 and P0
      %
      % [ss, flag] = ss.estimate(...) also returns the fmincon flag
      
      if size(y, 1) ~= obj.p && size(y,2) == obj.p
        y = y';
      end
        
      if nargin < 3 || isempty(ss0)
        [theta0U, theta0, ss0] = obj.initializeRandom(y);
      else
        % Initialization
        if isa(ss0, 'StateSpace')
          obj.checkConformingSystem(ss0);
          theta0 = obj.ThetaMapping.system2theta(ss0);
        else
          theta0 = ss0;
          validateattributes(theta0, {'numeric'}, ...
            {'vector', 'numel', obj.ThetaMapping.nTheta}, 'estimate', 'theta0');
        end
        assert(all(isfinite(theta0)), 'Non-finite values in starting point.');
        
        theta0U = obj.ThetaMapping.unrestrictTheta(theta0);
      end
        
      progress = EstimationProgress(theta0, obj.diagnosticPlot, obj.m, ss0);
      outputFcn = @(thetaU, oVals, st) ...
          progress.update(obj.ThetaMapping.restrictTheta(thetaU), oVals);
        
      assert(isnumeric(y), 'y must be numeric.');
      assert(isa(ss0, 'StateSpace') || isnumeric(ss0));
      assert(obj.ThetaMapping.nTheta > 0, ...
        'All parameters known. Unable to estimate.');
      
      
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
        'SpecifyObjectiveGradient', obj.useAnalyticGrad || obj.useInternalNumericGrad, ...
        'UseParallel', obj.useParallel && ~obj.useInternalNumericGrad, ...
        'Display', displayType, ...
        'MaxFunctionEvaluations', 50000, ...
        'MaxIterations', 1000, ...
        'FunctionTolerance', obj.tol, ...
        'OptimalityTolerance', obj.tol, ...
        'StepTolerance', obj.stepTol, ...
        'TolCon', 0, ...
        'OutputFcn', outputFcn);
      
      optFMinUnc = optimoptions(@fminunc, ...
        'Algorithm', 'quasi-newton', ...
        'SpecifyObjectiveGradient', obj.useAnalyticGrad || obj.useInternalNumericGrad, ...
        'UseParallel', obj.useParallel && ~obj.useInternalNumericGrad, ...
        'Display', displayType, ...
        'MaxFunctionEvaluations', 50000, ...
        'MaxIterations', 1000, ...
        'FunctionTolerance', obj.tol, ...
        'OptimalityTolerance', obj.tol, ...
        'StepTolerance', obj.stepTol, ...
        'OutputFcn', outputFcn);
      
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
            minfunc = @(thetaU) obj.minimizeFun(thetaU, y, progress, true);
            [thetaUHat, logli, outflag, ~, gradient] = fminunc(...
              minfunc, theta0U, optFMinUnc);
          case 'fmincon'
            minfunc = @(thetaU) obj.minimizeFun(thetaU, y, progress, true);
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
                otherwise
                  rethrow(ex);
              end 
            end
          case 'fminsearch'
            tempGrad = obj.useAnalyticGrad;
            obj.useAnalyticGrad = false;
            minfunc = @(thetaU) obj.minimizeFun(thetaU, y, progress, false);

            [thetaUHat, logli, outflag] = fminsearch(...
              minfunc, theta0U, optFMinSearch);
            obj.useAnalyticGrad = tempGrad;
            
            gradient = [];
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
      ss_out = obj.ThetaMapping.theta2system(thetaHat);
      
      % Run smoother, plot smoothed state
      if obj.diagnosticPlot
        progress.alpha = ss_out.smooth(y);
        progress.updateFigure();
      end
      
      % Get diagnostic info on estimation progress
      diagnostic = progress.diagnostics();
    end
  end
  
  methods (Hidden = true)
    %% Maximum likelihood estimation helper methods
    function [negLogli, gradient] = minimizeFun(obj, thetaU, y, progress, calcGrad)
      % Get the likelihood of
      theta = obj.ThetaMapping.restrictTheta(thetaU);
      ss1 = obj.ThetaMapping.theta2system(theta);
      ss1 = ss1.setDefaultInitial();
      
      if any(diag(ss1.H) < 0)
        keybaord;
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
          [rawLogli, thetaGradient, fOut] = ss1.gradient(y, obj.ThetaMapping, theta);

          GthetaUtheta = obj.ThetaMapping.thetaUthetaGrad(thetaU);
          rawGradient = GthetaUtheta * thetaGradient;
        else
          progress.totalEvaluations = progress.totalEvaluations + 1;

          [~, rawLogli, fOut] = ss1.filter(y);
          rawGradient = [];          
        end
        
        % Don't plot the diffuse parts of the state because they look odd
        a = fOut.a;
        a(:,1:fOut.dt) = nan;
        
        if ~isnan(rawLogli) && imag(rawLogli) ~= 0
          error();
          keyboard;
        end
        
        % Put filtered state in figure for plotting
        progress.a = a;  
        progress.ss = ss1;
      catch ex
        rawLogli = nan;
        rawGradient = nan(obj.ThetaMapping.nTheta, 1);
        
        progress.nanIterations = progress.nanIterations + 1;
      end
      
      if rawLogli > 0
        keyboard;
      end
      
      negLogli = -rawLogli;
      gradient = -rawGradient;
    end
    
    function [cx, ceqx] = nlConstraintFun(obj, thetaU)
      % Constraints of the form c(x) <= 0 and ceq(x) = 0.
      scale = 1e6;
      % vec = @(M) reshape(M, [], 1);

      theta = obj.ThetaMapping.restrictTheta(thetaU);

      % User constraints
      % TODO/FIXME: Is it a problem that deltaCX doesn't match this now? Yes.
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
      % cx = scale * -[det(ss1.H) det(ss1.Q)]';
      % cx = []; % Removed H, Q dets on 1/25
      if ~obj.ThetaMapping.usingDefaultP0
        cx = [cx; scale * -det(ss1.Q0)];
      end         
      ceqx = 0;
      
      % And the gradients 
      warning off MATLAB:nearlySingularMatrix
      warning off MATLAB:singularMatrix
      % G = obj.ThetaMapping.parameterGradients(theta);
      % deltaCX = scale * -[det(ss1.H) * G.H * vec(inv(ss1.H)), ...
      %   det(ss1.Q) * G.Q * vec(inv(ss1.Q))];
      if ~obj.ThetaMapping.usingDefaultP0
        % [~, G.P0] = obj.ThetaMapping.initialValuesGradients(ss1, G, theta);
        % This is suspect. Need to rewrite G.P0 first.
        % deltaCX = [deltaCX det(ss1.Q0) * G.P0 * vec(ss1.R0 / ss1.Q0 * ss1.R0')];
      end
      warning on MATLAB:nearlySingularMatrix
      warning on MATLAB:singularMatrix
      
      % deltaCeqX = sparse(0);
      
      % Constrain a submatrix of T to be stationary: 
      % c(x) should be abs(max(eig(T))) - 1
      if ~isempty(obj.stationaryStates)
        eigs = eig(ss1.T(obj.stationaryStates,obj.stationaryStates,obj.tau.T(1)));
        stationaryCx = scale * (max(abs(eigs)) - 1);
        cx = [cx; stationaryCx];
        % deltaCX = [];
      end
      
      if any(isnan(cx))
        warning('Nan constraint.');
      end
    end
    
    function [theta0U, theta0, ss0] = initializeRandom(obj, y)
      % Generate default initialization
      
      % The default initialization
      iAttempt = 0; 
      iLogli = nan(obj.initializeAttempts, 1); 
      iTheta0U = nan(obj.ThetaMapping.nTheta, obj.initializeAttempts);
      
      %while (~isfinite(ll0) || any(~isfinite(grad0))) && iAttempt < obj.initializeAttempts
      for iAttempt = 1:obj.initializeAttempts
        iTheta0U(:,iAttempt) = obj.initializeRange(1) + rand(obj.ThetaMapping.nTheta, 1) * ...
          (obj.initializeRange(2) - obj.initializeRange(1));
        
        try
          theta0 = obj.ThetaMapping.restrictTheta(iTheta0U(:,iAttempt));
          ss0 = obj.ThetaMapping.theta2system(theta0);
          [~, iLogli(iAttempt)] = ss0.filter(y);
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
        [ll0, grad0] = ss0.gradient(y, obj.ThetaMapping);
        
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
