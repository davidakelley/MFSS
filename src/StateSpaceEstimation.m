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
    
    fminsearchMaxIter = 500;
    
    % Indicator for use of analytic gradient
    useAnalyticGrad = true;

    % Indicator to use more accurate, slower gradient
    useInternalNumericGrad = true;
    
    % Allowable flags in estimation
    flagsAllowed = -1:5;
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
      
      obj.ThetaMapping = obj.ThetaMapping.addRestrictions(inOpts.LowerBound, inOpts.UpperBound);
      
      % Estimation restrictions - estimation will be bounded by bounds on
      % parameter matricies passed and non-negative restrictions on variances. 
      % obj = obj.generateRestrictions(inOpts.LowerBound, inOpts.UpperBound);
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
                 
      assert(isnumeric(y), 'y must be numeric.');
      assert(isa(ss0, 'StateSpace'));
      assert(obj.ThetaMapping.nTheta > 0, ...
        'All parameters known. Unable to estimate.');
      
      % Initialize
      obj.checkConformingSystem(ss0);

      theta0 = obj.ThetaMapping.system2theta(ss0);
      assert(all(isfinite(theta0)), 'Non-finite values in starting point.');
      
      progress = EstimationProgress(theta0, obj.diagnosticPlot);
      outputFcn = @(x, oVals, st) progress.update(x, oVals);
      
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
        'SpecifyObjectiveGradient', obj.useAnalyticGrad | obj.useInternalNumericGrad, ...
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
            minfunc = @(theta) obj.minimizeFun(theta, y, progress, true);
            [thetaHat, logli, outflag, ~, gradient] = fminunc(...
              minfunc, theta0, optFMinUnc);
          case 'fmincon'
            minfunc = @(theta) obj.minimizeFun(theta, y, progress, true);
            [thetaHat, logli, outflag, ~, ~, gradient] = fmincon(... 
              minfunc, theta0, [], [], [], [], [], [], nonlconFn, optFMinCon);
          case 'fminsearch'
            tempGrad = obj.useAnalyticGrad;
            obj.useAnalyticGrad = false;
            minfunc = @(theta) obj.minimizeFun(theta, y, progress, false);

            [thetaHat, logli, outflag] = fminsearch(...
              minfunc, theta0, optFMinSearch);
            obj.useAnalyticGrad = tempGrad;
            
            gradient = [];
        end
        
        progress.nextSolver();
        stopestim = strcmpi(progress.stopStatus, 'stopestim');
        
        assert(~isempty(intersect(obj.flagsAllowed, outflag)), ...
          'Optimizer returned disallowed flag %d.', outflag);
        
        loopContinue = ~stopestim && iter < obj.solveIterMax && ...
          (iter <= 2 || abs(logli0 - logli) > obj.solveTol);
        
        if logli0 < logli
          warning('StateSpaceEstimation:estimate:solverDecrease', ...
            ['Solver decreased likelihood by %g. \n' ...
            'Returning higher likelihood solution.'], logli - logli0);
          diagnostic = progress.diagnostics();

          thetaHat = diagnostic.thetaHist(find(-diagnostic.likelihoodHist == logli0, 1, 'last'), :)';
          assert(~isempty(thetaHat));
          logli = logli0;
          gradient = [];
          loopContinue = false;
        end
        
        assert(~any(isnan(thetaHat)), 'Estimation Error');
        
        theta0 = thetaHat;
      end
      warning on MATLAB:nearlySingularMatrix;
  
      % Save estimated system to current object
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
    function [negLogli, gradient] = minimizeFun(obj, theta, y, progress, calcGrad)
      % Get the likelihood of
      ss1 = obj.ThetaMapping.theta2system(theta);
      
      if any(diag(ss1.H) < 0)
        keybaord;
      end
      
      % Really enforce constraints
      if any(obj.nlConstraintFun(theta) > 0)
        % warning('Constraint violated in solver.');
        negLogli = 1e30 * max(obj.nlConstraintFun(theta));
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
      
      if max(abs(ss1Vectorized)) > 1e15
        % keyboard;
      end
      
      try
        if calcGrad && obj.useInternalNumericGrad
          % Calculate likelihood and gradient
          [rawLogli, rawGradient, fOut] = ss1.gradient(y, obj.ThetaMapping, theta);
        else
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
      catch ex
        rawLogli = nan;
        rawGradient = nan(obj.ThetaMapping.nTheta, 1);
        
        progress.nanIterations = progress.nanIterations + 1;
      end
      
      negLogli = -rawLogli;
      gradient = -rawGradient;
    end
    
    function [cx, ceqx, deltaCX, deltaCeqX] = nlConstraintFun(obj, theta)
      % Constraints of the form c(x) <= 0 and ceq(x) = 0.
      scale = 1e6;
      vec = @(M) reshape(M, [], 1);
      
      % Return the negative determinants in cx
      ss1 = obj.ThetaMapping.theta2system(theta);
      % cx = scale * -[det(ss1.H) det(ss1.Q)]';
      cx = []; % Removed H, Q dets on 1/25
      if ~obj.ThetaMapping.usingDefaultP0
        cx = [cx; scale * -det(ss1.Q0)];
      end         
      ceqx = 0;
      
      % And the gradients 
      warning off MATLAB:nearlySingularMatrix
      warning off MATLAB:singularMatrix
      G = obj.ThetaMapping.parameterGradients(theta);
    	deltaCX = scale * -[det(ss1.H) * G.H * vec(inv(ss1.H)), ...
                 det(ss1.Q) * G.Q * vec(inv(ss1.Q))];
      if ~obj.ThetaMapping.usingDefaultP0
        [~, G.P0] = obj.ThetaMapping.initialValuesGradients(ss1, G, theta);
        % This is suspect. Need to rewrite G.P0 first.
        deltaCX = [deltaCX det(ss1.Q0) * G.P0 * vec(ss1.R0 / ss1.Q0 * ss1.R0')];
      end
      warning on MATLAB:nearlySingularMatrix
      warning on MATLAB:singularMatrix
      
      deltaCeqX = sparse(0);
      
      % Constrain a submatrix of T to be stationary: 
      % c(x) should be abs(max(eig(T))) - 1
      if ~isempty(obj.stationaryStates)
        eigs = eig(ss1.T(obj.stationaryStates,obj.stationaryStates,obj.tau.T(1)));
        stationaryCx = scale * (max(abs(eigs)) - 1);
        cx = [cx; stationaryCx];
        deltaCX = [];
      end
      
      if any(isnan(cx))
        warning('Nan constraint.');
      end
    end
  end
  
end
