classdef StateSpace < matlab.mixin.CustomDisplay
  % Mixed-frequency state space model
  %
  % Includes filtering/smoothing algorithms and maximum likelihood
  % estimation of parameters with restrictions.
  %
  % Object construction
  % -------------------
  %   ss = StateSpace(Z, d, H, T, c, R, Q)
  %   ss = StateSpace(Z, d, H, T, C, R, Q, accumulator)
  %
  % The d & c parameters may be entered as empty for convenience.
  %
  % Time varrying parameters may be passed as structures with a field for
  % the parameter values and a vector indicating the timing. Using Z as an
  % example the field names would be Z.Zt and Z.tauZ.
  %
  % Accumulators may be defined for each observable series in the accumulator
  % structure. Three fields need to be defined:
  %   index     - linear indexes of series needing accumulation
  %   calendar  - calendar of observations for accumulated series
  %   horizon   - periods covered by each observation
  % These fields may also be named xi, psi, and Horizon. For more
  % information, see the readme.
  %
  % Filtering/smoothing
  % -------------------
  %   [a, logl] = ss.filter(y)
  %   [a, logl] = ss.filter(y, a0, P0)
  %   [a, logl, filterValues] = ss.filter(...)
  %   alpha = ss.smooth(y)
  %   alpha = ss.smooth(y, a0, P0)
  %   [alpha, smootherValues] = ss.smooth(...)
  %
  % Additional estimates from the filter (P, v, F, M, K, L) and
  % smoother (eta, r, N, a0tilde) are returned in the filterValues and
  % smootherValues structures. The multivariate filter also returns w and
  % Finv (the inverse of F). The multivariate smoother also returns V and J.
  %
  % When the initial value parameters are not passed or are empty, default
  % values will be generated as either the stationary solution (if it exists)
  % or the approximate diffuse case with large kappa. The value of kappa
  % can be set with ss.kappa before the use of the filter/smoother.
  %
  % The univariate filter/smoother will be used if H is diagonal. Set
  % ss.filterUni to false to force the use of the multivaritate versions.
  %
  % Mex versions will be used unless ss.useMex is set to false or the mex
  % files cannot be found.
  %
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
  % TODO (10/27/16)
  % ---------------
  %   - TVP for gradient (G.x matricies)
  %   - Accumulators for gradient - repeated/fixed entries
  %   - TVP/accumulators in estimation/thetaMap
  %   - mex version of the gradient function
  %   - EM algorithm
  %   - Create utiltiy methods for standard accumulator creation
  %   - Can we make Y be (n x p)?
  
  properties
    Z, d, H           % Observation equation parameters
    T, c, R, Q        % State equation parameters
    accumulator       % Structure defining accumulated series
    tau               % Structure of time-varrying parameter indexes
    
    a0, P0            % Initial value parameters
    kappa = 1e6;      % Diffuse initialization constant
    
    useMex = true;    % Use mex versions. Set to false if no mex compiled.
    filterUni         % Use univarite filter if appropriate (H is diagonal)
    
    verbose = true;   % Screen output during ML estimation
    tol = 1e-10;      % ML-estimation likelihood tolerance
    stepTol = 1e-12;
    iterTol = 1e-6;
  end
  
  properties(Hidden=true)
    % Dimensions
    p                 % Number of observed series
    m                 % Number of states
    g                 % Number of shocks
    n                 % Time periods
    
    % General options
    systemParam = {'Z', 'd', 'H', 'T', 'c', 'R', 'Q', 'a0', 'P0'};
    symmetricParams = {'H', 'Q', 'P0', 'P'};
    timeInvariant     % Indicator for TVP models
    
    % ML Estimation parameters
    thetaMap          % Structure of mappings from theta vector to parameters
    useGrad = true;   % Indicator for use of analytic gradient
  end
  
  methods
    function obj = StateSpace(Z, d, H, T, c, R, Q, accumulator)
      % StateSpace constructor
      % Pass state parameters to construct new object (or pass a structure
      % containing the neccessary parameters)
      
      if nargin == 1
        % Structure of parameter values was passed, contains all parameters
        accumulator = Z.accumulator;
        parameters = rmfield(Z, 'accumulator');
      elseif nargin == 7
        parameters = obj.compileStruct(Z, d, H, T, c, R, Q);
        accumulator = [];
      elseif nargin >= 8
        parameters = obj.compileStruct(Z, d, H, T, c, R, Q);
      else
        error('Input error.');
      end
      
      obj = obj.systemParameters(parameters);
      obj = obj.addAccumulators(accumulator);
      
      % Check mex files exist
      mexMissing = any([isempty(which('ss_mex.kfilter_uni'));
                        isempty(which('ss_mex.kfilter_multi'));
                        isempty(which('ss_mex.kfilter_uni'));
                        isempty(which('ss_mex.kfilter_multi'))]);         
      if obj.useMex && mexMissing 
        obj.useMex = false;
        warning('MEX files not found. See .\mex\make.m');
      end
    end
    
    function [a, logli, filterOut] = filter(obj, y, a0, P0)
      obj = obj.checkSample(y);
      if nargin > 3
        obj = obj.setInitial(a0, P0);
      elseif nargin > 2
        obj = obj.setInitial(a0);
      end
      obj = setDefaultInitial(obj);

      if obj.filterUni
        if obj.useMex
          [a, logli, filterOut] = obj.filter_uni_mex(y);
        else
          [a, logli, filterOut] = obj.filter_uni_m(y);
        end
      else
        if obj.useMex
          [a, logli, filterOut] = obj.filter_multi_mex(y);
        else
          [a, logli, filterOut] = obj.filter_multi_m(y);
        end
      end
    end
    
    function [alpha, smootherOut] = smooth(obj, y, a0, P0)
      obj = obj.checkSample(y);
      if nargin > 3
        obj = obj.setInitial(a0, P0);
      elseif nargin > 2
        obj = obj.setInitial(a0);
      end
      obj = setDefaultInitial(obj);
      
      [~, logli, filterOut] = obj.filter(y);
      
      if obj.filterUni
        if obj.useMex
          [alpha, smootherOut] = obj.smoother_uni_mex(y, filterOut);
        else
          [alpha, smootherOut] = obj.smoother_uni_m(y, filterOut);
        end
      else
        if obj.useMex
          [alpha, smootherOut] = obj.smoother_multi_mex(y, filterOut);
        else
          [alpha, smootherOut] = obj.smoother_multi_m(y, filterOut);
        end
      end
      smootherOut.logli = logli;
    end
    
    function [obj, flag] = estimate(obj, y, ss0, varargin)
      % Estimate missing parameter values via maximum likelihood.
      %
      % ss = ss.estimate(y, ss0) estimates any missing parameters in ss via
      % maximum likelihood on the data y using ss0 as an initialization.
      %
      % ss.estimate(y, ss0, a0, P0) uses the initial values a0 and P0
      %
      % [ss, flag] = ss.estimate(...) also returns the fmincon flag
      
      % Set initial values
      [obj, ss0] = obj.checkConformingSystem(y, ss0, varargin{:});
      
      %
      obj = obj.generateThetaMap();
      paramVec = obj.getParamVec();
      
      % Restrict the diagonal of H and Q to be positive
      estimInd = find(obj.thetaMap.estimated);
      varPos = [obj.thetaMap.elem.H(1:obj.thetaMap.shape.H(1)+1:end), ...
        obj.thetaMap.elem.Q(1:obj.thetaMap.shape.Q(1)+1:end)];
      paramVarPos = intersect(estimInd, varPos);
      [~, thetaVarPos] = intersect(estimInd, paramVarPos);
      
      thetaPositiveLB = -Inf * ones(sum(obj.thetaMap.estimated), 1);
      thetaPositiveLB(thetaVarPos, :) = 0;
      
      % Initialize
      ss0.thetaMap = obj.thetaMap;
      paramVec0 = ss0.getParamVec();
      assert(all(paramVec0(~obj.thetaMap.estimated) == ...
        paramVec(~obj.thetaMap.estimated)), ...
        'Starting values must match constrained parameters.');
      theta0 = paramVec0(obj.thetaMap.estimated);
      assert(all(isfinite(theta0)), 'Non-finite values in starting point.');
      
      % Run fminunc/fmincon
      minfunc = @(theta) obj.minimizeFun(theta, y);
      plotFcns = {@optimplotfval, @optimplotfirstorderopt, ...
        @optimplotstepsize, @optimplotconstrviolation};
%       if obj.verbose
%         displayType = 'iter-detailed';
%       else
        displayType = 'none';
%       end
      options = optimoptions(@fmincon, ...
        'Algorithm', 'interior-point', ...
        'SpecifyObjectiveGradient', obj.useGrad, ...
        'Display', displayType, ...
        'MaxFunctionEvaluations', 50000, ...
        'MaxIterations', 10000, ...
        'FunctionTolerance', obj.tol, 'OptimalityTolerance', obj.tol, ...
        'StepTolerance', obj.stepTol, ...
        'PlotFcns', plotFcns);
      
      warning off MATLAB:nearlySingularMatrix;
      
      obj.iterDisplay([]);
      iter = 0; lolgli = []; logli0 = []; lineLen = [];
      while iter < 2 || logli0 - lolgli > obj.iterTol
        iter = iter + 1;
        logli0 = lolgli;
        
        [thetaHat, lolgli, flag] = fmincon(minfunc, theta0, [], [], [], [], ...
          thetaPositiveLB, [], [], options);
        
        lineLen = obj.iterDisplay(iter, lolgli, logli0, lineLen);
        theta0 = thetaHat;
      end
      
      warning on MATLAB:nearlySingularMatrix;
      obj.iterDisplay(-1);
  
      % Save estimated system to current object
      obj = obj.theta2system(thetaHat);
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
  
  methods(Hidden = true)
    %% Filter/smoother methods
    function [a, logli, filterOut] = filter_uni_m(obj, y)
      % Univariate filter
      
      a       = zeros(obj.m, obj.n+1);
      P       = zeros(obj.m, obj.m, obj.n+1);
      LogL    = zeros(obj.p, obj.n);
      v       = zeros(obj.p, obj.n);
      F       = zeros(obj.p, obj.n);
      M       = zeros(obj.m, obj.p, obj.n);
      K       = zeros(obj.m, obj.p, obj.n);
      L       = zeros(obj.m, obj.m, obj.n);
      
      a(:,1) = obj.T(:,:,obj.tau.T(1)) * obj.a0 + obj.c(:,obj.tau.c(1));
      P(:,:,1) = obj.T(:,:,obj.tau.T(1)) * obj.P0 * obj.T(:,:,obj.tau.T(1))' ...
        + obj.R(:,:,obj.tau.R(1)) * obj.Q(:,:,obj.tau.Q(1)) * obj.R(:,:,obj.tau.R(1))';
      
      for ii = 1:obj.n
        ind = find( ~isnan(y(:,ii)) );
        ati    = a(:,ii);
        Pti    = P(:,:,ii);
        for jj = 1:length(ind)
          Zjj = obj.Z(ind(jj),:,obj.tau.Z(ii));
          
          v(ind(jj),ii) = y(ind(jj),ii) - ...
            Zjj * ati - obj.d(ind(jj),obj.tau.d(ii));
          F(ind(jj),ii) = Zjj * Pti * Zjj' + ...
            obj.H(ind(jj),ind(jj),obj.tau.H(ii));
          M(:,ind(jj),ii) = Pti * Zjj';
          
          LogL(ind(jj),ii) = (log(F(ind(jj),ii)) + (v(ind(jj),ii)^2) / F(ind(jj),ii));
          
          ati = ati + M(:,ind(jj),ii) / F(ind(jj),ii) * v(ind(jj),ii);
          Pti = Pti - M(:,ind(jj),ii) / F(ind(jj),ii) * M(:,ind(jj),ii)';
        end
        K(:,ind,ii) = obj.T(:,:,obj.tau.T(ii+1)) * M(:,ind,ii);
        L(:,:,ii) = obj.T(:,:,obj.tau.T(ii+1)) - K(:,ind,ii) * obj.Z(ind,:,obj.tau.Z(ii));
        
        a(:,ii+1) = obj.T(:,:,obj.tau.T(ii+1)) * ati + obj.c(:,obj.tau.c(ii+1));
        P(:,:,ii+1) = obj.T(:,:,obj.tau.T(ii+1)) * Pti * obj.T(:,:,obj.tau.T(ii+1))' + ...
          obj.R(:,:,obj.tau.R(ii+1)) * obj.Q(:,:,obj.tau.Q(ii+1)) * obj.R(:,:,obj.tau.R(ii+1))';
      end
      
      logli = -(0.5 * sum(sum(isfinite(y)))) * log(2 * pi) - 0.5 * sum(sum(LogL));
      filterOut = obj.compileStruct(a, P, v, F, M, K, L);
    end
    
    function [a, logli, filterOut] = filter_uni_mex(obj, y)
      % Call mex function kfilter_uni
      [LogL, a, P, v, F, M, K, L] = ss_mex.kfilter_uni(y, ...
        obj.Z, obj.tau.Z, obj.d, obj.tau.d, obj.H, obj.tau.H, ...
        obj.T, obj.tau.T, obj.c, obj.tau.c, obj.R, obj.tau.R, obj.Q, obj.tau.Q, ...
        obj.a0, obj.P0);
      
      logli = -(0.5 * sum(sum(isfinite(y)))) * log(2 * pi) - 0.5 * sum(sum(LogL));
      
      filterOut = obj.compileStruct(a, P, v, F, M, K, L);
    end
    
    function [a, logli, filterOut] = filter_multi_m(obj, y)
      % Multivariate filter
      
      a       = zeros(obj.m, obj.n+1);
      P       = zeros(obj.m, obj.m, obj.n+1);
      LogL    = zeros(obj.n, 1);
      v       = zeros(obj.p, obj.n);
      w       = zeros(obj.p, obj.n);
      K       = zeros(obj.m, obj.p, obj.n);
      L       = zeros(obj.m, obj.m, obj.n);
      F       = zeros(obj.p, obj.p, obj.n);
      M       = zeros(obj.m, obj.p, obj.n);
      Finv    = zeros(obj.p, obj.p, obj.n);
      
      a(:,1) = obj.T(:,:,obj.tau.T(1)) * obj.a0 + obj.c(:,obj.tau.c(1));
      P(:,:,1) = obj.T(:,:,obj.tau.T(1)) * obj.P0 * obj.T(:,:,obj.tau.T(1))'...
        + obj.R(:,:,obj.tau.R(1)) * obj.Q(:,:,obj.tau.Q(1)) * obj.R(:,:,obj.tau.R(1))';
      
      for ii = 1:obj.n
        % Create W matrix for possible missing values
        W = eye(obj.p);
        ind = ~isnan(y(:,ii));
        W = W((ind==1),:);
        
        Zii = W * obj.Z(:,:,obj.tau.Z(ii));
        v(ind,ii) = y(ind,ii) - Zii * a(:,ii) - W * obj.d(:,obj.tau.d(ii));
        F(ind,ind,ii) = Zii * P(:,:,ii) * Zii' + ...
          W * obj.H(:,:,obj.tau.H(ii)) * W';
        [Finv(ind, ind, ii), logDetF] = obj.pseudoinv(F(ind, ind, ii), 1e-12);
        
        LogL(ii) = logDetF + v(ind,ii)' * Finv(ind,ind,ii) * v(ind,ii);
        
        M(:,ind,ii) = P(:,:,ii) * Zii' * Finv(ind,ind,ii);
        K(:,ind,ii) = obj.T(:,:,obj.tau.T(ii+1)) * M(:,ind,ii);
        L(:,:,ii) = obj.T(:,:,obj.tau.T(ii+1)) - K(:,ind,ii) * Zii;
        w(ind,ii) = Finv(ind,ind,ii) * v(ind,ii);
        
        a(:,ii+1) = obj.T(:,:,obj.tau.T(ii+1)) * a(:,ii)+...
          obj.c(:,obj.tau.c(ii+1)) + K(:,ind,ii) * v(ind,ii);
        P(:,:,ii+1) = obj.T(:,:,obj.tau.T(ii+1)) * P(:,:,ii) * L(:,:,ii)'+ ...
          obj.R(:,:,obj.tau.R(ii+1)) * obj.Q(:,:,obj.tau.Q(ii+1)) * ...
          obj.R(:,:,obj.tau.R(ii+1))';
      end
      
      logli = -(0.5 * sum(sum(isfinite(y)))) * log(2 * pi) - 0.5 * sum(LogL);
      filterOut = obj.compileStruct(a, P, v, F, M, K, L, w, Finv);
    end
    
    function [a, logli, filterOut] = filter_multi_mex(obj, y)
      % Call mex function kfilter_uni
      [LogL, a, P, v, F, M, K, L, w, Finv] = ss_mex.kfilter_multi(y, ...
        obj.Z, obj.tau.Z, obj.d, obj.tau.d, obj.H, obj.tau.H, ...
        obj.T, obj.tau.T, obj.c, obj.tau.c, obj.R, obj.tau.R, obj.Q, obj.tau.Q, ...
        obj.a0, obj.P0);
      
      logli = -(0.5 * sum(sum(isfinite(y)))) * log(2 * pi) - 0.5 * sum(sum(LogL));
      
      filterOut = obj.compileStruct(a, P, v, F, M, K, L, w, Finv);
    end
    
    function [alpha, smootherOut] = smoother_uni_m(obj, y, fOut)
      % Univariate smoother
      
      alpha = zeros(obj.m, obj.n);
      eta   = zeros(obj.g, obj.n);
      r     = zeros(obj.m, obj.n);
      N     = zeros(obj.m, obj.m, obj.n+1);
      
      rti = zeros(obj.m,1);
      Nti = zeros(obj.m,obj.m);
      for ii = obj.n:-1:1
        ind = find( ~isnan(y(:,ii)) );
        for jj = length(ind):-1:1
          Lti = eye(obj.m) - fOut.M(:,ind(jj),ii) * ...
            obj.Z(ind(jj),:,obj.tau.Z(ii)) / fOut.F(ind(jj),ii);
          rti = obj.Z(ind(jj),:,obj.tau.Z(ii))' / ...
            fOut.F(ind(jj),ii) * fOut.v(ind(jj),ii) + Lti' * rti;
          Nti = obj.Z(ind(jj),:,obj.tau.Z(ii))' / ...
            fOut.F(ind(jj),ii) * obj.Z(ind(jj),:,obj.tau.Z(ii)) ...
            + Lti' * Nti * Lti;
        end
        r(:,ii) = rti;
        N(:,:,ii) = Nti;
        
        alpha(:,ii) = fOut.a(:,ii) + fOut.P(:,:,ii) * r(:,ii);
        eta(:,ii) = obj.Q(:,:,obj.tau.Q(ii)) * obj.R(:,:,obj.tau.R(ii))' * r(:,ii);
        
        rti = obj.T(:,:,obj.tau.T(ii))' * rti;
        Nti = obj.T(:,:,obj.tau.T(ii))' * Nti * obj.T(:,:,obj.tau.T(ii));
      end
      a0tilde = obj.a0 + obj.P0 * rti;
      
      smootherOut = obj.compileStruct(alpha, eta, r, N, a0tilde);
    end
    
    function [alpha, smootherOut] = smoother_uni_mex(obj, y, fOut)
      % Call mex function kfilter_uni
      [alpha, eta, r, N, a0tilde] = ss_mex.ksmoother_uni(y, ...
        obj.Z, obj.tau.Z, obj.H, obj.tau.H, ...
        obj.T, obj.tau.T, obj.R, obj.tau.R, obj.Q, obj.tau.Q, ...
        fOut.a, fOut.P, fOut.v, fOut.F, fOut.M, fOut.L, ...
        obj.a0, obj.P0);
      
      smootherOut = obj.compileStruct(alpha, eta, r, N, a0tilde);
    end
    
    function [alpha, smootherOut] = smoother_multi_m(obj, y, fOut)
      % Multivariate smoother
      
      alpha   = zeros(obj.m, obj.n);
      eta     = zeros(obj.g, obj.n);
      epsilon = zeros(obj.p, obj.n);
      r       = zeros(obj.m, obj.n+1);
      N       = zeros(obj.m, obj.m, obj.n+1);
      V       = zeros(obj.m, obj.m, obj.n);
      J       = zeros(obj.m, obj.m, obj.n);
      
      for ii = obj.n:-1:1
        % Create W matrix for possible missing values
        W = eye(obj.p);
        ind = ~isnan(y(:,ii));
        W = W((ind==1),:);
        
        r(:,ii) = (W * obj.Z(:,:,obj.tau.Z(ii)))' * ...
          fOut.Finv(ind,ind,ii) * fOut.v(ind,ii) + fOut.L(:,:,ii)' * r(:,ii+1);
        N(:,:,ii) = (W * obj.Z(:,:,obj.tau.Z(ii)))' * fOut.Finv(ind,ind,ii) * ...
          (W * obj.Z(:,:,obj.tau.Z(ii))) ...
          + fOut.L(:,:,ii)' * N(:,:,ii+1) * fOut.L(:,:,ii);
        u(ind,ii) = fOut.Finv(ind,ind,ii) * fOut.v(ind,ii) - ...
          fOut.K(:,ind,ii)'*r(:,ii);
        
        alpha(:,ii) = fOut.a(:,ii) + fOut.P(:,:,ii)*r(:,ii);
        eta(:,ii)   = obj.Q(:,:,obj.tau.Q(ii)) * obj.R(:,:,obj.tau.R(ii))'*r(:,ii);
        epsilon(ind,ii) = (W * obj.H(:,:,obj.tau.H(ii)) * W') * u(ind,ii);
        
        V(:,:,ii) = fOut.P(:,:,ii) - ...
          fOut.P(:,:,ii) * N(:,:,ii) * fOut.P(:,:,ii);
        J(:,:,ii) = fOut.P(:,:,ii) * fOut.L(:,:,ii)' * ...
          eye(obj.m) * (eye(obj.m) - N(:,:,ii+1) * fOut.P(:,:,ii+1));
      end
      
      a0tilde = obj.a0 + obj.P0 * obj.T(:,:,obj.tau.T(1))'*r(:,1);
      smootherOut = obj.compileStruct(alpha, eta, epsilon, r, N, a0tilde, V, J);
    end
    
    function [alpha, smootherOut] = smoother_multi_mex(obj, y, fOut)
      % Call mex function kfilter_uni
      [alpha, eta, epsilon, r, N, a0tilde, V, J] = ss_mex.ksmoother_multi(y, ...
        obj.Z, obj.tau.Z, obj.H, obj.tau.H, ...
        obj.T, obj.tau.T, obj.R, obj.tau.R, obj.Q, obj.tau.Q, ...
        fOut.a, fOut.P, fOut.v, fOut.M, fOut.K, fOut.L, fOut.Finv, ...
        obj.a0, obj.P0);
      
      smootherOut = obj.compileStruct(alpha, eta, epsilon, r, N, a0tilde, V, J);
    end
    
    function [logli, gradient] = gradient_multi_filter(obj, y)
      % Gradient algorithm from Nagakura (SSRN # 1634552).
      obj.filterUni = false;
      [~, logli, fOut] = obj.filter(y);
      
      % Generate vectorized gradients of parameters
      paramNames = [obj.systemParam(1:7), {'a', 'P'}];
      nThetaElem = obj.thetaElements();
      nParamElem = cellfun(@numel, obj.parameters);
      G = struct;
      for iP = 1:length(paramNames)
        iParam = paramNames{iP};
        prevThetaElem = sum(nThetaElem(1:iP-1));
        paramGrad = zeros(sum(nThetaElem), nParamElem(iP));
        
        thetaInds = prevThetaElem+1:prevThetaElem+nThetaElem(iP);
        if ~strcmp(iParam, obj.symmetricParams)
          paramGrad(thetaInds, :) = eye(nThetaElem(iP));
        else
          paramGrad(thetaInds, :) = obj.gradThetaSym(sqrt(nParamElem(iP)));
        end
        if iP <= 7
          G.(iParam) = sparse(paramGrad(obj.thetaMap.estimated, :));
        else
          G.(iParam) = paramGrad(obj.thetaMap.estimated, :);
        end
      end
      
      commutation = obj.genCommutation(obj.m);
      Nm = (eye(obj.m^2) + commutation);
      vec = @(M) reshape(M, [], 1);
      
      % Compute partial results that have less time-variation (even with TVP)
      kronRR = zeros(obj.g*obj.g, obj.m*obj.m, max(obj.tau.R));
      for iR = 1:max(obj.tau.R)
        kronRR(:, :, iR) = kron(obj.R(:,:,iR)', obj.R(:,:,iR)');
      end
      
      [tauQRrows, ~, tauQR] = unique([obj.tau.R' obj.tau.Q'], 'rows');
      kronQRI = zeros(obj.g * obj.m, obj.m * obj.m, max(tauQR));
      for iQR = 1:max(tauQR)
        kronQRI(:, :, iQR) = kron(obj.Q(:,:,tauQRrows(iQR, 2)) * obj.R(:,:,tauQRrows(iQR, 1))', ...
          eye(obj.m));
      end
      
      % Initial period: G.a and G.P capture effects of a0, T
      G.a = G.a * obj.T(:,:,obj.tau.T(1))' + ...
        G.c + ...
        G.T * kron(obj.a0, eye(obj.m));
      G.P = G.P * kron(obj.T(:,:,obj.tau.T(1))', obj.T(:,:,obj.tau.T(1))') + ...
        G.Q * kron(obj.R(:,:,obj.tau.R(1))', obj.R(:,:,obj.tau.R(1))') + ...
        (G.T * kron(obj.P0 * obj.T(:,:,obj.tau.T(1))', eye(obj.m)) + ...
        G.R * kron(obj.Q(:,:,obj.tau.Q(1)) * ...
        obj.R(:,:,obj.tau.R(1))', eye(obj.m))) * ...
        (eye(obj.m^2) + commutation);
      
      % Recursion through time periods
      grad = zeros(sum(sum(obj.thetaMap.estimated)), obj.n);
      for ii = 1:obj.n
        W = sparse(eye(obj.p));
        ind = ~isnan(y(:,ii));
        W = W((ind==1),:);
        kronWW = kron(W', W');
        
        Zii = W * obj.Z(:, :, obj.tau.Z(ii));
        
        ww = fOut.w(ind,ii) * fOut.w(ind,ii)';
        Mv = fOut.M(:,:,ii) * fOut.v(:, ii);
         
        grad(:, ii) = G.a * Zii' * fOut.w(ind,ii) + ...
          0.5 * G.P * vec(Zii' * ww * Zii - Zii' * fOut.Finv(ind,ind,ii) * Zii) + ...
          G.d * W' * fOut.w(ind,ii) + ...
          G.Z * vec(W' * (fOut.w(ind,ii) * fOut.a(:,ii)' + ...
            fOut.w(ind,ii) * Mv' - fOut.M(:,ind,ii)')) + ...
          0.5 * G.H * kronWW * vec(ww - fOut.Finv(ind,ind,ii));
        
        % Set t+1 values
        PL = fOut.P(:,:,ii) * fOut.L(:,:,ii)';
        
        G.a = G.a * fOut.L(:,:,ii)' + ...
          G.P * kron(Zii' * fOut.w(ind,ii), fOut.L(:,:,ii)') + ...
          G.c - ...
          G.d * fOut.K(:,:,ii)' + ...
          G.Z * (kron(PL, fOut.w(:,ii)) - ...
            kron(fOut.a(:,ii) + Mv, fOut.K(:,:,ii)')) - ...
          G.H * kron(fOut.w(:,ii), fOut.K(:,:,ii)') + ...
          G.T * kron(fOut.a(:,ii) + Mv, eye(obj.m));
        
        G.P = G.P * kron(fOut.L(:,:,ii)', fOut.L(:,:,ii)') + ...
          G.H * kron(fOut.K(:,:,ii)', fOut.K(:,:,ii)') + ...
          G.Q * kronRR(:,:, obj.tau.R(ii+1)) + ...
          (G.T * kron(PL, eye(obj.m)) - ...
            G.Z * kron(PL, fOut.K(:,:,ii)') + ...
            G.R * kronQRI(:, :, tauQR(ii))) * ...
            Nm;
      end
      
      gradient = sum(grad, 2);
    end
    
    %% Accumulators
    function obj = addAccumulators(obj, accumulator)
      % Augment system parameters with Harvey accumulators
      %
      % Augment Standard State Space with Harvey Accumulator
      % $\xi$  - $\xi_{h}$ indice vector of each series accumulated
      % $A$    - selection matrix
      % $\psi$ - full history of accumulator indicator (for each series)
      % $\tau$ - number of unique accumulator types
      %        - Two Cases:'flow'    = 1 Standard Flow variable;
      %                    'average' = 0 Time-averaged stock variable
      
      % Based on work by Andrew Butters, 2013
      
      if isempty(accumulator)
        return
      else
        obj.accumulator = accumulator;
      end
      
      % Validate input
      if all(ismember(fieldnames(accumulator), {'xi', 'psi', 'Horizon'}))
        accum.index = accumulator.xi;
        accum.calendar = accumulator.psi;
        accum.horizon = accumulator.Horizon;
        accumulator = accum;
      end
      if islogical(accumulator.index)
        accumulator.index = find(accumulator.index);
      end
      assert(all(ismember(fieldnames(accumulator), {'index', 'calendar', 'horizon'})), ...
        'accumulator must be a structure with fields index, calendar, and horizon');
      assert(length(accumulator.index) == size(accumulator.calendar, 1));
      assert(length(accumulator.index) == size(accumulator.horizon, 1));
      
      accumIndex = accumulator.index;
      calendar = accumulator.calendar;
      horizon = accumulator.horizon;
      
      % Set tau
      if isempty(obj.tau)
        obj.n = size(calendar, 2)-1;
        obj = obj.setInvariantTau();
      end
      
      accumTypes = any((calendar==0),2);
      [accumObs, accumStates] = find(any(obj.Z(accumIndex,:,:) ~= 0, 3));
      
      [states, AddLags, ColAdd, RowPos, mNew] = obj.accumAugmentedStateDims(horizon, accumObs, accumStates);
      
      if sum(AddLags)==0
        newZ0 = obj.Z;
        newT0 = obj.T;
        newc0 = obj.c;
        newR0 = obj.R;
      else
        newZ0 = zeros(obj.p, obj.m + sum(AddLags), obj.n);
        newZ0(:, 1:obj.m,:) = obj.Z;
        newT0 = zeros(obj.m + sum(AddLags), obj.m + sum(AddLags), obj.n);
        newT0(1:obj.m, 1:obj.m,:) = Tt;
        newT0(sub2ind(size(newT0), (obj.m + 1:mNew)', reshape(ColAdd(ColAdd>0),[],1))) = 1;
        newc0 = zeros(obj.m + sum(AddLags), obj.n);
        newc0(1:obj.m,:) = ct;
        newR0 = zeros(obj.m + sum(AddLags), obj.g, obj.n);
        newR0(1:obj.m, :, :) = Rt;
        obj.m = mNew;
      end
      
      % Indentify unique elements of the state that need accumulation
      nonZeroZpos = zeros(length(accumIndex), obj.m);
      [APsi, ~, nonZeroZpos(sub2ind([length(accumIndex) obj.m], accumObs, accumStates))] ...
        = unique([accumStates accumTypes(accumObs) horizon(accumObs, :) calendar(accumObs, :)], 'rows');
      
      nTau  = size(APsi, 1);
      st = APsi(:, 1);
      accumTypes = APsi(:, 2);
      Hor = APsi(:, 3:size(horizon,2) + 2);
      Psi = APsi(:, size(horizon,2) + 3:end);
      
      mOld = obj.m;
      obj.m = obj.m + nTau;
      
      % Construct new T matrix
      Ttypes   = [obj.tau.T' Psi' Hor'];
      [uniqueTs,~, newtauT] = unique(Ttypes,'rows');
      NumT = size(uniqueTs, 1);
      newT = zeros(obj.m, obj.m, NumT);
      for jj = 1:NumT
        newT(1:mOld, :, jj) = [newT0(:,:,uniqueTs(jj,1)) zeros(mOld, nTau)];
        for h = 1:nTau
          if accumTypes(h) == 1
            newT(mOld + h, [1:mOld mOld + h],jj) = ...
              [newT0(APsi(h,1),:,uniqueTs(jj,1)) ...
              uniqueTs(jj,1+h)];
          else
            newT(mOld+h, [1:mOld mOld+h],jj) =...
              [(1/uniqueTs(jj,1+h)) * newT0(APsi(h,1),:,uniqueTs(jj,1))...
              (uniqueTs(jj,1+h)-1)/uniqueTs(jj,1+h)];
            if uniqueTs(jj, nTau+1+h) > 1
              cols = RowPos(st(h)==states, 1:uniqueTs(jj,nTau+1+h)-1);
              newT(mOld+h, cols, jj) = newT(mOld+h,cols,jj) + (1/uniqueTs(jj,1+h));
            end
          end
        end
      end
      obj.T = newT;
      obj.tau.T = newtauT;
      
      % Construct new c vector
      ctypes   = [obj.tau.c' Psi'];
      [uniquecs, ~, newtauc] = unique(ctypes,'rows');
      Numc = size(uniquecs, 1);
      newc = zeros(obj.m, Numc);
      for jj = 1:Numc
        newc(1:mOld, jj) = newc0(:, uniquecs(jj,1));
        for h = 1:nTau
          if accumTypes(h) == 1
            newc(mOld+h, jj) = newc0(APsi(h,1), uniquecs(jj,1));
          else
            newc(mOld+h, jj) = (1/uniquecs(jj,1+h))*...
              newc0(APsi(h,1), uniquecs(jj,1));
          end
        end
      end
      obj.c = newc;
      obj.tau.c = newtauc;
      
      % Construct new R matrix
      Rtypes   = [obj.tau.R' Psi'];
      [uniqueRs, ~, newtauR] = unique(Rtypes,'rows');
      NumR = size(uniqueRs, 1);
      newR = zeros(obj.m, obj.g, NumR);
      for jj = 1:NumR
        newR(1:mOld,1:obj.g,jj) = newR0(:, :, uniqueRs(jj,1));
        for h = 1:nTau
          if accumTypes(h) == 1
            newR(mOld+h,:,jj) = newR0(APsi(h,1), :, uniqueRs(jj,1));
          else
            newR(mOld+h,:,jj) = (1/uniqueRs(jj, 1+h))*...
              newR0(APsi(h,1), :, uniqueRs(jj,1));
          end
        end
      end
      obj.R = newR;
      obj.tau.R = newtauR;
      
      % Construct new Z matrix
      newZ = zeros(obj.p, obj.m, size(obj.Z,3));
      for jj = 1:size(obj.Z, 3)
        newZ(:,1:mOld,jj) = newZ0(:,:,jj);
        newZ(accumIndex,:,jj) = zeros(length(accumIndex), obj.m);
        for h = 1:length(accumIndex)
          cols = find(newZ0(accumIndex(h),:,jj));
          newZ(accumIndex(h), mOld + nonZeroZpos(h,cols),jj) = newZ0(accumIndex(h),cols,jj);
        end
      end
      obj.Z = newZ;
    end
    
    function [states, AddLags, ColAdd, RowPos, mNew] = ...
        accumAugmentedStateDims(obj, horizon, accumObs, accumStates)
      % Checking we have the correct number of lags of the states that need
      % accunulation to be compatible with Horizon
      states = unique(accumStates);
      maxHor = max(horizon, [], 2);
      AddLags = zeros(length(states), 1);
      ColAdd  = zeros(length(states), max(maxHor));
      RowPos  = zeros(length(states), max(maxHor));
      mNew = obj.m;
      for jj=1:length(states)
        mHor = max(maxHor(accumObs(states(jj) == accumStates)));
        [numlagsjj,lagposjj] = obj.LagsInState(states(jj));
        
        AddLags(jj) = max(mHor-1-numlagsjj-1,0);
        RowPos(jj,1:numlagsjj+1) = [states(jj) lagposjj'];
        if AddLags(jj) > 0
          ColAdd(jj,numlagsjj+2) = lagposjj(end);
          if AddLags(jj)>1
            ColAdd(jj,numlagsjj+3:numlagsjj+1+AddLags(jj)) = mNew:mNew+AddLags(jj)-1;
          end
        end
        RowPos(jj,numlagsjj+2:numlagsjj+1+AddLags(jj)) = mNew+1:mNew+AddLags(jj);
        mNew = mNew+AddLags(jj);
      end
    end
    
    function [nLags, positions] = LagsInState(obj, varPos)
      % Finds the lags of a variable in the state space
      %
      % Input:
      %   varPos    - linear index of vector position of variable of interest
      %
      % Output:
      %   nLags     - number of lags in the state of variable of interest
      %   positions - linear index position of the lags
      
      % Criteria for determining if a candidate variable is a "lag variable":
      %   - The variable of interest loads on to the candidate with a 1.
      %   - All other variables have a loading on the candidate of 0.
      %   - No shocks are selected for the candidate in R.
      criteriaFn = @(interest) all([(obj.T(:, interest, obj.tau.T(1)) == 1) ...
        all(obj.T(:,((1:size(obj.T,1)) ~= interest), obj.tau.T(1)) == 0,2) ...
        all(obj.R(:, :, obj.tau.R(1)) == 0, 2)], 2);
      
      nLags = 0;
      positions = [];
      while any(criteriaFn(varPos))
        nLags = nLags + 1;
        varPos = find(criteriaFn(varPos));
        positions = [positions; varPos]; %#ok<AGROW>
      end
    end
    
    %% Maximum likelihood estimation functions
    function [negLogli, gradient] = minimizeFun(obj, theta, y)
      % Get the likelihood of
      [ss1, a0_theta, P0_theta] = obj.theta2system(theta);
      
      try
        if obj.useGrad
          [rawLogli, rawGradient] = ss1.gradient(y, a0_theta, P0_theta);
          gradient = -rawGradient;
        else
          [~, rawLogli] = ss1.filter(y, a0_theta, P0_theta);
          gradient = [];
        end
      catch
        rawLogli = nan;
        gradient = nan(sum(obj.thetaMap.estimated), 1);
      end
      
      negLogli = -rawLogli;
    end
    
    function [newObj, a0, P0] = theta2system(obj, theta)
      % Generates a StateSpace object from a vector of parameters being
      % estimated, theta.
      
      if all(size(theta, 1) == sum(obj.thetaMap.estimated))
        % Theta is a vector of elements being estimated, get full parameter
        % vector first, replace estimated elements with theta
        paramVec = obj.thetaMap.constrained;
        paramVec(obj.thetaMap.estimated) = theta;
      else
        % Full vector of parameters being passed
        paramVec = theta;
      end
      
      % Generate system parameters from theta vector
      params = struct;
      for iP = 1:length(obj.systemParam)
        iParam = obj.systemParam{iP};
        if ~strcmp(iParam, obj.symmetricParams)
          params.(iParam) = reshape(paramVec(obj.thetaMap.elem.(iParam)), ...
            obj.thetaMap.shape.(iParam));
        else
          selectHalf = tril(true(obj.thetaMap.shape.(iParam)));
          params.(iParam) = zeros(obj.thetaMap.shape.(iParam));
          params.(iParam)(selectHalf) = paramVec(obj.thetaMap.elem.(iParam));
          halfMatTrans = params.(iParam)';
          params.(iParam)(selectHalf') = halfMatTrans(selectHalf');
        end
      end
      
      params.accumulator = obj.accumulator;
      a0 = params.a0;
      P0 = params.P0;
      
      % Create new StateSpace object from parameters
      newObj = StateSpace(params);
      newObj = newObj.generateThetaMap;
      newObj.thetaMap.estimated = obj.thetaMap.estimated;
    end
    
    function paramVec = getParamVec(obj)
      % TODO: Handle restrictions on the system parameters passed as non-nan values
      lowerH = obj.H(tril(true(size(obj.H))));
      lowerQ = obj.Q(tril(true(size(obj.Q))));
      lowerP0 = obj.P0(tril(true(size(obj.P0))));
      
      vec = @(M) reshape(M, [], 1);
      paramVec = [vec(obj.Z); vec(obj.d); lowerH; ...
        vec(obj.T); vec(obj.c); vec(obj.R); lowerQ; ...
        vec(obj.a0); lowerP0];
    end
    
    function obj = generateThetaMap(obj)
      % Generate map from parameters to theta
      
      obj.thetaMap = struct;
      elems = obj.thetaElements();
      
      elemMap = arrayfun(@(x) sum(elems(1:x-1))+1:sum(elems(1:x)), ...
        1:length(elems), 'Uniform', false);
      obj.thetaMap.elem = cell2struct(elemMap', obj.systemParam');
      
      shapes = cellfun(@size, obj.parameters, 'Uniform', false);
      obj.thetaMap.shape = cell2struct(shapes', obj.systemParam');
      
      paramVec = obj.getParamVec();
      obj.thetaMap.estimated = isnan(paramVec);
      obj.thetaMap.constrained = paramVec;
    end
    
    function elems = thetaElements(obj)
      elems = cellfun(@numel, obj.parameters);
      % Correct for symmetric matrixes (H, Q, P0):
      elems(3) = sum(sum(tril(ones(size(obj.H)))));
      elems(7) = sum(sum(tril(ones(size(obj.Q)))));
      elems(9) = sum(sum(tril(ones(size(obj.P0)))));
    end
    
    function Gtheta = gradThetaSym(obj, dim) %#ok<INUSL>
      % Generate the G_{\theta}(A) matrix for symmetric matricies A.
      % Only neccessary input is the size of A (must be square).
      % The theta vector is ordered as the vectorized lower triangular
      % portion of A.
      
      nFreeElem = sum(1:dim);
      nElem = dim.^2;
      Gtheta = zeros(nFreeElem, nElem);
      
      % Set the diagonal elements and the lower non-diagonal elements
      vec = @(M) reshape(M, [], 1);
      freeParam = vec(tril(true(dim)));
      Gtheta(:, freeParam) = eye(nFreeElem);
      
      % Set the upper non-diagonal elements
      % Find the lower off-diagonal elements we've already set. The
      % corresponding element will be (n-1)*(j-i) columns to the right.
      offDiag = vec(tril(true(dim), -1))';
      covarThetaRows = any(bsxfun(@eq, offDiag, Gtheta) & Gtheta ~= 0, 2);
      
      [~,lowerOffDiagCol] = find(Gtheta(covarThetaRows, :));
      iMinusj = vec(cumsum(tril(true(dim-1))));
      iMinusj(iMinusj == 0) = [];
      upperOffDiagCol = lowerOffDiagCol + (dim-1) * iMinusj;
      
      Gtheta(covarThetaRows, upperOffDiagCol) = eye(nElem - nFreeElem);
    end
    
    function [logli, gradient] = gradient(obj, y, a0, P0)
      % Returns the likelihood and the change in the likelihood given the
      % change in any system parameters that are currently set to nans.
      
      % TODO: First think about G.x carefully.
      assert(obj.timeInvariant, 'TVP gradient not developed yet.'); 
      
      obj = obj.checkSample(y);
      if nargin > 2
        obj = obj.setInitial(a0, P0);
      end
      obj = setDefaultInitial(obj);
      
      [logli, gradient] = obj.gradient_multi_filter(y);
    end
    
    function [obj, ss0] = checkConformingSystem(obj, y, ss0, a0, P0)
      % Make sure ss0 is a similar StateSpace system to obj
      assert(isa(ss0, 'StateSpace'));
      assert(obj.p == ss0.p, 'ss0 observation dimension mismatch');
      assert(obj.m == ss0.m, 'ss0 state dimension mismatch.');
      assert(obj.g == ss0.g, 'ss0 shock dimension mismatch.');
      assert(obj.timeInvariant == ss0.timeInvariant, ...
        'Mismatch in time varrying parameters.');
      if ~obj.timeInvariant
        assert(obj.n == obj.n, 'ss0 time dimension mismatch.');
      end
      
      obj = obj.checkSample(y);
      ss0 = ss0.checkSample(y);
      if nargin > 4
        ss0 = ss0.setInitial(a0, P0);
      elseif nargin > 3
        ss0 = ss0.setInitial(a0);
      end
      ss0 = ss0.setDefaultInitial();
      obj.a0 = ss0.a0;
      obj.P0 = ss0.P0;
    end
    
    function param = parameters(obj, index)
      % Getter for cell array of parameters
      param = {obj.Z, obj.d, obj.H, obj.T, obj.c, obj.R, obj.Q, ...
        obj.a0, obj.P0}; % Do a0, P0 need to be ss0? I don't think so.
      if nargin > 1
        param = param{index};
      end
    end
    
    %% Initialization
    function obj = setInitial(obj, a0, P0)
      % Setter for a0 and P0 (or kappa)
      if ~isempty(a0)
        assert(size(a0, 1) == obj.m, 'a0 should be a m X 1 vector');
        obj.a0 = a0;
      end
      
      if nargin > 2 && ~isempty(P0)
        if size(P0, 1) == 1 % (This is ok if m == 1 too)
          % Scalar value passed for kappa
          obj.P0 = eye(obj.m) * P0;
        elseif size(P0, 2) == 1
          % Vector value passed for kappa
          assert(size(P0, 1) == obj.m, 'kappa vector must be length m');
          obj.P0 = diag(P0);
        else
          % Regular P0 value passed
          assert(all(size(P0) == [obj.m obj.m]), 'P0 should be a m X m matrix');
          obj.P0 = P0;
        end
      end
    end
    
    function obj = setDefaultInitial(obj)
      % Set default a0 and P0.
      % Run before filter/smoother after a0 & P0 inputs have been processed
      if ~isempty(obj.a0) && ~isempty(obj.P0)
        % User provided a0 and P0.
        return
      end
      
      tempT = obj.T(:, :, obj.tau.T(1));
      if all(abs(eig(tempT)) < 1)
        % System is stationary. Compute unconditional estimate of state
        if isempty(obj.a0)
          obj.a0 = (eye(obj.m) - tempT) \ obj.c(:, obj.tau.c(1));
        end
        if isempty(obj.P0)
          tempR = obj.R(:, :, obj.tau.R(1));
          tempQ = obj.Q(:, :, obj.tau.Q(1));
          obj.P0 = reshape((eye(obj.m^2) - kron(tempT, tempT)) \ ...
            reshape(tempR * tempQ * tempR', [], 1), ...
            obj.m, obj.m);
        end
      else
        % Nonstationary case: use large kappa diffuse initialization
        if isempty(obj.a0)
          obj.a0 = zeros(obj.m, 1);
        end
        if isempty(obj.P0)
          obj.P0 = obj.kappa * eye(obj.m);
        end
      end
    end
    
    %% Utility Methods
    function obj = systemParameters(obj, parameters)
      % Obtain system matrices from structure
      % By passing both the different types of system matrices and the calendar
      % vector corresponding to the particular type used at any particular time
      % we allow for an arbitrary number of structural "breaks" in the state
      % space model. Furthermore, by allowing each system matrix/vector to have
      % its own calendar, no redundant matrices are saved in the workspace.
      
      % Assume system is time invariant, correct when we get a TVP matrix
      obj.timeInvariant = true;
      
      % Z system matrix
      if isstruct(parameters.Z)
        obj = obj.setTimeVarrying(length(parameters.Z.tauZ));
        obj.tau.Z = parameters.Z.tauZ;
        obj.Z = parameters.Z.Zt;
      else
        obj.Z = parameters.Z;
      end
      
      % d system matrix
      if isstruct(parameters.d)
        obj = obj.setTimeVarrying(length(parameters.d.taud));
        obj.tau.d = parameters.d.taud;
        obj.d = parameters.d.dt;
      elseif size(parameters.d, 2) > 1
        obj = obj.setTimeVarrying(size(parameters.d, 2));
        obj.tau.d = 1:obj.n;
        obj.d = parameters.d;
      else
        obj.d = parameters.d;
      end
      
      % H system matrix
      if isstruct(parameters.H)
        obj = obj.setTimeVarrying(length(parameters.H.tauH));
        obj.tau.H = parameters.H.tauH;
        obj.H = parameters.H.Ht;
      else
        obj.H = parameters.H;
      end
      
      % T system matrix
      if isstruct(obj.T)
        obj = obj.setTimeVarrying(length(parameters.T.tauT) - 1);
        obj.tau.T = parameters.T.tauT;
        obj.T = parameters.T.Tt;
      else
        obj.T = parameters.T;
      end
      
      % c system matrix
      if isstruct(parameters.c)
        obj = obj.setTimeVarrying(length(parameters.c.tauc) - 1);
        obj.tau.c = parameters.c.tauc;
        obj.c = parameters.c.ct;
      elseif size(parameters.c, 2) > 1
        obj = obj.setTimeVarrying(size(parameters.c, 2) - 1);
        obj.tau.c = [(1:obj.n) obj.n];
        obj.c = parameters.c;
      else
        obj.c = parameters.c;
      end
      
      % R system matrix
      if isstruct(parameters.R)
        obj = obj.setTimeVarrying(length(parameters.R.tauR) - 1);
        obj.tau.R = parameters.R.tauR;
        obj.R = parameters.R.Rt;
      else
        obj.R = parameters.R;
      end
      
      % Q system matrix
      if isstruct(parameters.Q)
        obj = obj.setTimeVarrying(length(parameters.Q.tauQ) - 1);
        obj.tau.Q = parameters.Q.tauQ;
        obj.Q = parameters.Q.Qt;
      else
        obj.Q = parameters.Q;
      end
      
      if ~obj.timeInvariant
        tauDims = [length(obj.tau.Z) length(obj.tau.d) length(obj.tau.H) ...
          length(obj.tau.T)-1 length(obj.tau.c)-1 ...
          length(obj.tau.R)-1 length(obj.tau.Q)-1];
        assert(all(tauDims == obj.n));
      end
      
      % Set dimensions:
      %   p - number of observables
      %   m - number of state variables
      %   g - number of shocks
      obj.p = size(obj.Z(:,:,end), 1);
      [obj.m, obj.g] = size(obj.R(:,:,end));
      
      validateKFilter(obj);
      
      % Check if we can use the univariate filter
      diagH = true;
      for iH = 1:size(obj.H, 3)
        if ~isdiag(obj.H(:,:,iH))
          diagH = false;
        end
      end
      obj.filterUni = diagH;
    end
    
    function obj = setTimeVarrying(obj, n)
      if obj.timeInvariant
        obj.timeInvariant = false;
        obj.n = n;
      else
        assert(obj.n == n, 'TVP calendar length mismatch.');
      end
    end
    
    function obj = setInvariantTau(obj)
      % Set the tau structure for TVP.
      % FIXME: obj.tau shouldn't be set if it's already defined
      taus = [repmat({ones([1 obj.n])}, [3 1]); ...
        repmat({ones([1 obj.n+1])}, [4 1])];
      obj.tau = cell2struct(taus, obj.systemParam(1:7));
    end
    
    function obj = checkSample(obj, y)
      assert(size(y, 1) == obj.p, ...
        'Number of series does not match observation equation.');
      % TODO: check accumulated series?
      
      if ~obj.timeInvariant
        % System with TVP, make sure length of taus matches data
        assert(size(y, 2) == obj.n);
      else
        %
        obj.n = size(y, 2);
        obj = obj.setInvariantTau();
      end
    end
    
    function validateKFilter(obj)
      % Check dimensions of inputs to Kalman filter.
      if obj.timeInvariant
        maxTaus = ones([7 1]);
      else
        maxTaus = structfun(@max, obj.tau);  % Untested?
      end
      
      validate = @(x, sz, name) validateattributes(x, {'numeric'}, ...
        {'size', sz}, 'StateSpace', name);
      
      % Measurement equation
      validate(obj.Z, [obj.p obj.m maxTaus(1)], 'Z');
      validate(obj.d, [obj.p maxTaus(2)], 'd');
      validate(obj.H, [obj.p obj.p maxTaus(3)], 'H');
      
      % State equation
      validate(obj.T, [obj.m obj.m maxTaus(4)], 'T');
      validate(obj.c, [obj.m maxTaus(5)], 'c');
      validate(obj.R, [obj.m obj.g maxTaus(6)], 'R');
      validate(obj.Q, [obj.g obj.g maxTaus(7)], 'Q');
    end
    
    function sOut = compileStruct(obj, varargin) %#ok<INUSL>
      % Combines variables passed as arguments into a struct
      % struct = compileStruct(a, b, c) will place the variables a, b, & c
      % in the output variable using the variable names as the field names.
      sOut = struct;
      for iV = 1:nargin-1
        sOut.(inputname(iV+1)) = varargin{iV};
      end
    end
    
    function [Finv, logDetF] = pseudoinv(obj, F, tol) %#ok<INUSL>
      tempF = 0.5 * (F + F');
      
      [PSVD, DSDV, PSVDinv] = svd(tempF);
      
      % Truncate small singular values
      firstZero = find(diag(DSDV) < tol, 1, 'first');
      
      if ~isempty(firstZero)
        PSVD = PSVD(:, 1:firstZero - 1);
        PSVDinv = PSVDinv(:, 1:firstZero - 1);
        DSDV = DSDV(1:firstZero - 1, 1:firstZero - 1);
      end
      
      Finv = PSVD * (DSDV\eye(length(DSDV))) * PSVDinv';
      logDetF = sum(log(diag(DSDV)));
    end
    
    function commutation = genCommutation(obj, m) %#ok<INUSL>
      % Generate commutation matrix
      % A commutation matrix is "a suqare mn-dimensional matrix partitioned
      % into mn sub-matricies of order (n, m) such that the ijth submatrix
      % as a 1 in its jith position and zeros elsewhere."
      E = @(i, j) [zeros(i-1, m); ...
        zeros(1, j-1), 1, zeros(1, m-j); zeros(m-i, m)];
      commutation = zeros(m^2);
      for iComm = 1:m
        for jComm = 1:m
          commutation = commutation + kron(E(iComm, jComm), E(iComm, jComm)');
        end
      end
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
  
  %% Display
  methods (Access = protected)
    function displayScalarObject(obj)
      if ~isempty(obj.accumulator)
        type = 'Mixed-Frequency';
      elseif ~obj.timeInvariant
        type = 'Time-varying parameter';
      else
        type = 'Time-invariant';
      end
      fprintf('%s state space model\n', type);
      
      space = '    ';
      fprintf(['%sObservation dimenstion : %d\n' ...
        '%sState dimension        : %d\n' ...
        '%sShocks                 : %d\n'], ...
        space, obj.p, space, obj.m, space, obj.g);
      if ~obj.timeInvariant
        fprintf('%sData dimension          : %d\n', obj.t);
      end
      
      allParamValues = [obj.Z(:); obj.d(:); obj.H(:); ...
        obj.T(:); obj.c(:); obj.R(:); obj.Q(:)];
      if any(isnan(allParamValues))
        fprintf('%sParameter values to be estimated: %d\n', space, ...
          sum(isnan(allParamValues)));
      end
    end
    
    function lineLen = iterDisplay(obj, iter, logli, logli0, prevLen)
      lineLen = [];
      if ~obj.verbose
        return
      end
      
      lineLen = 46;
      line = @(char) repmat(char, [1 lineLen]);
      
      if nargin < 2 || isempty(iter)
        algoTitle = 'StateSpace Maximum Likelihood Estimation';
        fprintf('\n%s\n', algoTitle);
        fprintf('%s\n  Iteration |  Log-likelihood |  Improvement\n%s\n', ...
          line('='), line('-'));
        return
      end
      if iter < 0
        % End of iter
        fprintf('%s\n', line('-'));
        return
      end
      
      gap = logli - logli0;
      tolLvl = num2str(abs(floor(log10(obj.iterTol)))+1);
      screenOutFormat = ['%11.0d | %16.8g | %12.' tolLvl 'g\n'];
      
      if iter <=2
        bspace = [];
      else
        bspace = repmat('\b', [1 prevLen]*desktop('-inuse'));
      end
      screenOut = sprintf(screenOutFormat, iter, logli, gap);
      fprintf([bspace screenOut]);
      
      lineLen = length(screenOut);
    end
  end

  
end