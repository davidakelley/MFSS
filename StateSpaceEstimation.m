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
  
  % David Kelley, 2016
  %
  % TODO (12/1/16)
  % ---------------
  %   - Restore gradient for P0 positive definite-ness.
  %   - TVP/accumulators
  %   - EM algorithm
  
  properties
    % Screen output during ML estimation
    verbose = false;
    
    % ML-estimation tolerances
    tol = 1e-10;      % Final estimation tolerance
    stepTol = 1e-12;  % Step-size tolerance for ML theta elements
    iterTol = 1e-6;   % EM tolerance
    
    % ML Estimation parameters
    ThetaMapping      % Mapping from theta vector to parameters
    useGrad = false;   % Indicator for use of analytic gradient
  end
  
  methods
    %% Constructor
    function obj = StateSpaceEstimation(Z, d, H, T, c, R, Q, accumulator, varargin)
      obj = obj@AbstractStateSpace(Z, d, H, T, c, R, Q, accumulator);
      
      inP = inputParser;
      inP.addParameter('a0', [], @isnumeric);
      inP.addParameter('P0', [], @isnumeric);
      inP.addParameter('LowerBound', [], @(x) isa(x, 'StateSpace'));
      inP.addParameter('UpperBound', [], @(x) isa(x, 'StateSpace'));
      inP.addParameter('ThetaMap', [], @(x) isa(x, 'ThetaMap'));
      inP.parse(varargin{:});
      inOpts = inP.Results;
      
      % Initial values - If initial values are not passed they will be set as 
      % the default values at each iteration of the estimation, effectively 
      % making them a function of other parameters. 
      if ~isempty(inOpts.a0) && ~isempty(inOpts.P0)
        obj = obj.setInitial(inOpts.a0, inOpts.P0);
      elseif ~isempty(inOpts.a0) 
        obj = obj.setInitial(inOpts.a0);
      end
      
      % Initial ThetaMap generation - will be augmented to include restrictions
      if ~isempty(inOpts.ThetaMap)
        obj.ThetaMapping = inOpts.ThetaMap;
      else
        obj.ThetaMapping = ThetaMap.ThetaMapEstimation(obj);
      end
      
      obj.ThetaMapping.addRestrictions(inOpts.LowerBound, inOpts.UpperBound);      
      
      % Estimation restrictions - estimation will be bounded by bounds on
      % parameter matricies passed and non-negative restrictions on variances. 
%       obj = obj.generateRestrictions(inOpts.LowerBound, inOpts.UpperBound);
    end
    
    %% Estimation methods
    function [obj, flag, gradient] = estimate(obj, y, ss0)
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
      
      % Initialize
      obj.checkConformingSystem(ss0);

      theta0 = obj.ThetaMapping.system2theta(ss0);
      assert(all(isfinite(theta0)), 'Non-finite values in starting point.');
      
      % Run fminunc/fmincon
      minfunc = @(theta) obj.minimizeFun(theta, y);
      nonlconFn = @obj.nlConstraintFun;
      
      plotFcns = {@optimplotfval, @optimplotfirstorderopt, ...
        @optimplotstepsize, @optimplotconstrviolation};
      if obj.verbose
        displayType = 'iter-detailed';
      else
        displayType = 'none';
      end
      options = optimoptions(@fmincon, ...
        'Algorithm', 'interior-point', ...
        'SpecifyObjectiveGradient', obj.useGrad, ...
        'Display', displayType, ...
        'MaxFunctionEvaluations', 50000, ...
        'MaxIterations', 10000, ...
        'FunctionTolerance', obj.tol, 'OptimalityTolerance', obj.tol, ...
        'StepTolerance', obj.stepTol, ...
        'PlotFcns', plotFcns, ...
        'TolCon', 0);
      
      warning off MATLAB:nearlySingularMatrix;
      
%       obj.iterDisplay([]);
      iter = 0; lolgli = []; logli0 = []; % lineLen = [];
      while iter < 2 || logli0 - lolgli > obj.iterTol
        iter = iter + 1;
        logli0 = lolgli;
        
        [thetaHat, lolgli, flag, ~, ~, gradient] = fmincon(minfunc, ...
          theta0, [], [], [], [], [], [], nonlconFn, options);
        
        % lineLen = obj.iterDisplay(iter, lolgli, logli0, lineLen);
        theta0 = thetaHat;
      end
      
      warning on MATLAB:nearlySingularMatrix;
%       obj.iterDisplay(-1);
  
      % Save estimated system to current object
      obj = obj.ThetaMapping.theta2system(thetaHat);
    end
    
    function obj = em_estimate(obj, y, ss0, varargin)
      % Estimate parameters through pseudo-maximum likelihood EM algoritm
      assert(obj.timeInvariant, ...
        'EM Algorithm only developed for time-invariant cases.');
      
      [obj, ss0] = obj.checkConformingSystem(y, ss0, varargin{:});
      iter = 0; logli0 = nan; gap = nan;
      
      % Generate F and G matricies for state and observation equations
      % Does it work to run the restricted OLS for the betas we know while
      % letting the variances be free in the regression but only keeping
      % the ones we're estimating afterward? I think so.
      
      
      % Iterate over EM algorithm
      while iter < 2 || gap > obj.tol
        iter = iter + 1;
        
        % E step: Estiamte complete log-likelihood
        [alpha, sOut] = ss0.smooth(y);
        
        % M step: Get parameters that maximize the likelihood
        % Measurement equation
        [ss0.Z, ss0.d, ss0.H] = obj.restrictedOLS(y', alpha', V, J, Fobs, Gobs);
        % State equation
        [ss0.T, ss0.c, RQR] = obj.restrictedOLS(...
          alpha(:, 2:end)', alpha(:, 1:end-1)', V, J, Fstate, Gstate);
        
        % Report
        if obj.verbose
          gap = abs((sOut.logli - logli0)/mean([sOut.logli; logli0]));
        end
        logli0 = sOut.logli;
      end
      
      obj = ss0;
    end
  end
  
  methods (Hidden = true)
    %% Maximum likelihood estimation helper methods
    function [negLogli, gradient] = minimizeFun(obj, theta, y)
      % Get the likelihood of
      ss1 = obj.ThetaMapping.theta2system(theta);
      
      try
        if obj.useGrad
          [rawLogli, rawGradient] = ss1.gradient(y, obj.ThetaMapping);
        else
          [~, rawLogli] = ss1.filter(y);
          rawGradient = [];
        end
      catch
        rawLogli = nan;
        rawGradient = nan(obj.ThetaMapping.nTheta, 1);
      end
      
      negLogli = -rawLogli;
      gradient = -rawGradient;
    end
    
    function [cx, ceqx, deltaCX, deltaCeqX] = nlConstraintFun(obj, theta)
      % Constraints of the form c(x) <= 0 and ceq(x) = 0.
      scale = 1e6;
      
      % Return the negative determinants in cx
      ss1 = obj.ThetaMapping.theta2system(theta);
      cx = scale * -[det(ss1.H) det(ss1.Q)]';
      if ~obj.usingDefaultP0
        cx = [cx; det(ss0.P0)];
      end      
      ceqx = 0;
      
%       G = ss1.generateParameterGradients();
%     	
%       vec = @(M) reshape(M, [], 1);
% 
%       % Should also give gradients
%       deltaCX = scale * -[det(ss1.H) * G.H * vec(inv(ss1.H)), ...
%                  det(ss1.Q) * G.Q * vec(inv(ss1.Q))];
%       if ~obj.usingDefaultP0
%         deltaCX = [deltaCX; det(P0_theta) * G.P0 * vec(inv(P0_theta))];
%       end
      deltaCX = [];
      deltaCeqX = sparse(0);
    end
    
    %% EM Algorithm Helper functions
    function [beta, sigma] = restrictedOLS(y, X, V, J, F, G)
      % Restricted OLS regression
      % See Econometric Analysis (Greene) for details.
      T_dim = size(y, 1);
      
      % Simple OLS
      xxT = sum(V, 3) + X * X';
      yxT = sum(J, 3) + y * X';
      yyT = sum(V, 3) + y * y';
      
      betaOLS = yxT/xxT;
      sigmaOLS = (yyT-OLS*yxT');
      
      % Restricted OLS estimator
      beta = betaOLS - (betaOLS * F - G) / (F' / xxT * F) * (F' / xxT);
      sigma = (sigmaOLS + (betaOLS * F - G) / ...
        (F' / xxT * F) * (betaOLS * F - G)') / T_dim;
    end
    
  end
end
