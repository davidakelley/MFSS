classdef StateSpace < matlab.mixin.CustomDisplay
  % Mixed-frequency state space model
  %
  % Includes filtering/smoothing algorithms and maximum likelihood
  % estimation of parameters with restrictions.
  %
  % ss = StateSpace(Z, d, H, T, c, R, Q)
  %
  % ss = StateSpace(Z, d, H, T, C, R, Q, Harvey)
  %
  % ss = StateSpace(Z, d, H, T, C, R, Q, Harvey, a0, P0)
  %
  % ss.filter(y)
  % ss.filter(y, a0, P0)
  % ss.smooth(y)
  % ss.smooth(y, a0, P0)
  % ss.estimate(y)
  % ss.em_estimate(y)
  
  % David Kelley, 2016
  %
  % Questions I need to answer:
  %     For the gradient: Missing data? Time varrying parameters?
  %     Is there any easy way to package the class file with the mex files?
  
  properties
    Z, d, H
    T, c, R, Q
    tau
    a0, P0
    Harvey
  end
  
  properties(Hidden = true)
    % Dimensions
    p         % Number of observed series
    m         % Number of states
    g         % Number of shocks
    n         % Time periods
    
    % General options
    useMex = true;    % Use mex versions. Set to false if no mex compiled.
    systemParam = {'Z', 'd', 'H', 'T', 'c', 'R', 'Q', 'a0', 'P0'};
    kappa = 1e6;
    
    % Object specific properties
    timeInvariant     % Indicator for TVP
    filterUni         % Use univarite filter if appropriate (H is diagonal)
    
    % ML Estimation parameters
    thetaMap
    useGrad = false;
    tol = 1e-6;
    verbose = true;
  end
  
  methods
    
    function obj = StateSpace(Z, d, H, T, c, R, Q, Harvey)
      % StateSpace constructor
      % Pass state parameters to construct new object (or pass a structure
      % containing the neccessary parameters)
      
      if nargin == 1
        % Structure of parameter values was passed, contains all parameters
        Harvey = Z.Harvey;
        parameters = rmfield(Z, 'Harvey');
      elseif nargin >= 8
        parameters = obj.compileStruct(Z, d, H, T, c, R, Q);
      else
        error('Input error.');
      end
      
      obj = obj.systemParameters(parameters);
      obj = obj.addAccumulators(Harvey);
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
        % No multivariate mex. Plan to implement Cholesky and use uni.
        [a, logli, filterOut] = obj.filter_multi_m(y);
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
        [alpha, smootherOut] = obj.smoother_multi_m(y, filterOut);
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
      
      % Restrict H and Q to be positive
      estimInd = find(obj.thetaMap.estimated);
      varPos = [obj.thetaMap.elem.H, obj.thetaMap.elem.Q];
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
      options = optimoptions(@fmincon, ...
        'Algorithm', 'interior-point', ...
        'SpecifyObjectiveGradient', obj.useGrad, ...
        'Display', 'iter-detailed', ...
        'MaxFunctionEvaluations', 50000, ...
        'MaxIterations', 10000, ...
        'PlotFcns', plotFcns);
      
      [thetaHat, ~, flag] = fmincon(minfunc, theta0, [], [], [], [], ...
        thetaPositiveLB, [], [], options);
      
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
      assert(exist('kfilter_uni') == 3, ...
        'MEX file not found. See .\mex\make.m'); %#ok<EXIST>
      
      [LogL, a, P, v, F, M, K, L] = kfilter_uni(y, ...
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
      w       = zeros(obj.m, obj.n);
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
    
    function [alpha, smootherOut] = smoother_uni_m(obj, y, fOut)
      % Univariate smoother
      
      alpha = zeros(obj.m, obj.n);
      eta   = zeros(obj.g, obj.n);
      r     = zeros(obj.m, obj.n);
      N     = zeros(obj.m, obj.m, obj.n+1);
      V     = zeros(obj.m, obj.m, obj.n);
      J     = zeros(obj.m, obj.m, obj.n);
      
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
        
        V(:,:,ii) = fOut.P(:,:,ii) - ...
          fOut.P(:,:,ii) * N(:,:,ii) * fOut.P(:,:,ii);
        J(:,:,ii) = fOut.P(:,:,ii) * fOut.L(:,:,ii)' * ...
          eye(obj.m) * (eye(obj.m) - N(:,:,ii+1) * fOut.P(:,:,ii+1));
        
        rti = obj.T(:,:,obj.tau.T(ii))' * rti;
        Nti = obj.T(:,:,obj.tau.T(ii))' * Nti * obj.T(:,:,obj.tau.T(ii));
      end
      a0tilde = obj.a0 + obj.P0 * rti;
      
      smootherOut = obj.compileStruct(alpha, eta, r, N, V, J, a0tilde);
    end
    
    function [alpha, smootherOut] = smoother_uni_mex(obj, y, fOut)
      % Call mex function kfilter_uni
      assert(exist('ksmoother_uni') == 3, ...
        'MEX file not found. See .\mex\make.m'); %#ok<EXIST>
      
      [alpha, eta, r, N, V, J] = ksmoother_uni(y, ...
        obj.Z, obj.tau.Z, obj.H, obj.tau.H, ...
        obj.T, obj.tau.T, obj.R, obj.tau.R, obj.Q, obj.tau.Q, ...
        fOut.a, fOut.P, fOut.v, fOut.F, fOut.M, fOut.L, ...
        obj.a0, obj.P0);

      smootherOut = obj.compileStruct(alpha, eta, r, N, V, J);
     end
    
    function [alpha, smootherOut] = smoother_multi_m(obj, y, fOut)
      % Multivariate smoother
      
      alpha   = zeros(obj.m, obj.n);
      eta     = zeros(obj.m, obj.n);
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
      smootherOut = obj.compileStruct(alpha, eta, epsilon, r, N, V, J, a0tilde);
    end
    
    function [logli, gradient] = gradient_multi_filter(obj, y)
      % Gradient algorithm from Nagakura (SSRN # 1634552).
      
      [~, logli, fOut] = obj.filter_multi_m(y);
      
      parameters = {obj.Z, obj.d, obj.H, obj.T, obj.c, obj.R, obj.Q, obj.a0, obj.P0};
      nParamElem = cellfun(@numel, parameters);
      
      paramNames = [obj.systemParam(1:7), {'a', 'P'}];
      G = struct;
      for iP = 1:length(paramNames)
        prevElem = sum(nParamElem(1:iP-1));
        G.(paramNames{iP}) = zeros(sum(nParamElem), nParamElem(iP));
        G.(paramNames{iP})(prevElem+1:prevElem+nParamElem(iP), :) = eye(nParamElem(iP));
      end
      
      % Generate commutation matrix
      % A commutation matrix is "a suqare mn-dimensional matrix partitioned
      % into mn sub-matricies of order (n, m) such that the ijth submatrix
      % as a 1 in its jith position and zeros elsewhere."
      E = @(i, j) [zeros(i-1, obj.m); ...
        zeros(1, j-1), 1, zeros(1, obj.m-j); zeros(obj.m-i, obj.m)];
      commutation = zeros(obj.m^2);
      for iComm = 1:obj.m
        for jComm = 1:obj.m
          commutation = commutation + kron(E(iComm, jComm), E(iComm, jComm)');
        end
      end
      
      vec = @(M) reshape(M, [], 1);
      
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
      
      % Initialize G_\theta(a_0) = 1, G_\theta(P_0) = 1 (?)
      grad = zeros(sum(nParamElem), obj.n);
      for ii = 1:obj.n
        W = eye(obj.p);
        ind = ~isnan(y(:,ii));
        W = W((ind==1),:);
        
        Zii = W * obj.Z(:, :, obj.tau.Z(ii));
        
        grad(:, ii) = G.a * Zii' * fOut.w(ind,ii) + ...
          0.5 * G.P * vec(Zii' * fOut.w(ind,ii) * fOut.w(ind,ii)' * Zii - ...
          Zii' * fOut.Finv(ind,ind,ii) * Zii) + ...
          G.d * W' * fOut.w(ind,ii) + ...
          G.Z * vec(W' * (fOut.w(ind,ii) * fOut.a(:,ii)' + ...
          fOut.w(ind,ii) * fOut.v(ind,ii)' * fOut.M(:,ind,ii)' - ...
          fOut.M(:,ind,ii)')) + ...
          0.5 * G.H * kron(W', W') * vec(fOut.w(ind,ii) * fOut.w(ind,ii)' - ...
          fOut.Finv(ind,ind,ii));
        
        % Set t+1 values
        G.a = G.a * fOut.L(:,:,ii)' + ...
          G.P * kron(Zii' * fOut.w(ind,ii), fOut.L(:,:,ii)') + ...
          G.c + ...
          G.d * fOut.K(:,:,ii)' + ...
          G.Z * (kron(fOut.P(:,:,ii) * fOut.L(:,:,ii)', fOut.w(:,ii)) - ...
          kron(fOut.a(:,ii) + ...
          fOut.M(:,:,ii) * fOut.v(:, ii), fOut.K(:,:,ii)')) - ...
          G.H * kron(fOut.w(:,ii), fOut.K(:,:,ii)') + ...
          G.T * kron(fOut.a(:,ii) + ...
          fOut.M(:,:,ii) * fOut.v(:,ii), eye(obj.m));
        G.P = G.P * kron(fOut.L(:,:,ii)', fOut.L(:,:,ii)') + ...
          G.H * kron(fOut.K(:,:,ii)', fOut.K(:,:,ii)') + ...
          G.Q * kron(obj.R(:,:,obj.tau.R(ii+1))', obj.R(:,:,obj.tau.R(ii+1))') + ...
          (G.T * kron(fOut.P(:,:,ii) * fOut.L(:,:,ii)', eye(obj.m)) + ...
          G.Z * kron(fOut.P(:,:,ii) * fOut.L(:,:,ii)', fOut.K(:,:,ii)') + ...
          G.R * kron(obj.Q(:,:,obj.tau.Q(ii+1)) * ...
          obj.R(:,:,obj.tau.R(ii+1))', eye(obj.m))) * ...
          (eye(obj.m^2) + commutation);
      end
      
      gradient = sum(grad, 2);
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
        obj = obj.setTimeVarrying(length(parameters.d.tauZ));
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
        obj = obj.setTimeVarrying(length(parameters.H.tauZ));
        obj.tau.H = parameters.H.tauH;
        obj.H = parameters.H.Ht;
      else
        obj.H = parameters.H;
      end
      
      % T system matrix
      if isstruct(obj.T)
        obj = obj.setTimeVarrying(length(parameters.T.tauZ) - 1);
        obj.tau.T = parameters.T.tauT;
        obj.T = parameters.T.Tt;
      else
        obj.T = parameters.T;
      end
      
      % c system matrix
      if isstruct(parameters.c)
        obj = obj.setTimeVarrying(length(parameters.c.tauZ) - 1);
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
        obj = obj.setTimeVarrying(length(parameters.R.tauZ) - 1);
        obj.tau.R = parameters.R.tauR;
        obj.R = parameters.R.Rt;
      else
        obj.R = parameters.R;
      end
      
      % Q system matrix
      if isstruct(parameters.Q)
        obj = obj.setTimeVarrying(length(parameters.Q.tauZ) - 1);
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
    
    function obj = setTimeVarrying(obj, n)
      if obj.timeInvariant
        obj.timeInvariant = false;
        obj.n = n;
      else
        assert(obj.n == n, 'TVP calendar length mismatch.');
      end
    end
    
    function obj = addAccumulators(obj, Harvey)
      % Augment system parameters with Harvey accumulators
      
      % Augment Standard State Space with Harvey Accumulator
      % $\xi$  - $\xi_{h}$ indice vector of each series accumulated
      % $A$    - selection matrix
      % $\psi$ - full history of accumulator indicator (for each series)
      % $\tau$ - number of unique accumulator types
      %        - Two Cases:'flow'    = 1 Standard Flow variable;
      %                    'average' = 0 Time-averaged stock variable
      
      % Andrew Butters, 2013
      
      if isempty(Harvey)
        return
      else
        obj.Harvey = Harvey;
      end
      
      xi      = Harvey.xi;
      psi     = Harvey.psi;
      Horizon = Harvey.Horizon;
      
      type = any((psi==0),2);
      [s,r] = find((any((obj.Z(xi,:,:)~=0),3))');
      
      % ===================================================================
      % Horizon
      %  Checking have the correct number of lags of the states
      %  that need accunulation, to be compatible with Horizon
      %  Note: move outside code
      states = unique(s);
      maxHor = max(Horizon,[],2);
      AddLags = zeros(length(states),1);
      ColAdd  = zeros(length(states),max(maxHor));
      RowPos  = zeros(length(states),max(maxHor));
      mnew = obj.m;
      for jj=1:length(states)
        mHor = max(maxHor(r(states(jj)==s)));
        [numlagsjj,lagposjj] = LagsInState(states(jj), ...
          obj.T(:,:, obj.tau.T(1)), obj.R(:,:,obj.tau.R(1))); 
        
        AddLags(jj) = max(mHor-1-numlagsjj-1,0);
        RowPos(jj,1:numlagsjj+1) = [states(jj) lagposjj'];
        if AddLags(jj) > 0
          ColAdd(jj,numlagsjj+2) = lagposjj(end);
          if AddLags(jj)>1
            ColAdd(jj,numlagsjj+3:numlagsjj+1+AddLags(jj)) = mnew:mnew+AddLags(jj)-1;
          end
        end
        RowPos(jj,numlagsjj+2:numlagsjj+1+AddLags(jj)) = mnew+1:mnew+AddLags(jj);
        mnew = mnew+AddLags(jj);
      end
      
      if sum(AddLags)==0
        newZ0 = Zt;
        newT0 = Tt;
        newc0 = ct;
        newR0 = Rt;
      else
        newZ0 = zeros(size(Zt,1),size(Zt,2)+sum(AddLags),size(Zt,3));
        newZ0(:,1:size(Zt,2),:) = Zt;
        newT0 = zeros(size(Tt,1)+sum(AddLags),size(Tt,2)+sum(AddLags),size(Tt,3));
        newT0(1:size(Tt,1),1:size(Tt,2),:) = Tt;
        newT0(sub2ind(size(newT0),(obj.m+1:mnew)',reshape(ColAdd(ColAdd>0),[],1))) = 1;
        newc0 = zeros(size(ct,1)+sum(AddLags),size(ct,2));
        newc0(1:size(ct,1),:) = ct;
        newR0 = zeros(size(Rt,1)+sum(AddLags),size(Rt,2),size(Rt,3));
        newR0(1:size(Rt,1),:,:) = Rt;
        obj.m = mnew;
      end
      % ===================================================================
      
      nonZeroZpos = zeros(length(xi), obj.m);
      
      % Indentify unique elements of the state that need accumulation
      [APsi,~,nonZeroZpos(sub2ind([length(xi) obj.m],r,s))] ...
        = unique([s type(r) Horizon(r,:) psi(r,:)],'rows');
      
      nTau  = size(APsi,1);
      st   = APsi(:,1);
      type = APsi(:,2);
      Hor  = APsi(:,3:size(Horizon,2)+2);
      Psi  = APsi(:,size(Horizon,2)+3:end);
      
      Ttypes   = [tauT' Psi' Hor'];
      ctypes   = [tauc' Psi'];
      Rtypes   = [tauR' Psi'];
      
      [uniqueTs,~, newtauT] = unique(Ttypes,'rows');
      [uniquecs,~, newtauc] = unique(ctypes,'rows');
      [uniqueRs,~, newtauR] = unique(Rtypes,'rows');
      
      NumT = size(uniqueTs,1);
      Numc = size(uniquecs,1);
      NumR = size(uniqueRs,1);
      
      newT = zeros(obj.m+nTau, obj.m+nTau,NumT);
      for jj = 1:NumT
        newT(1:obj.m,:,jj) = [newT0(:,:,uniqueTs(jj,1)) zeros(obj.m, nTau)];
        for  h =1:nTau
          if type(h) == 1
            newT(obj.m+h, [1:obj.m obj.m+h],jj) = ...
              [newT0(APsi(h,1),:,uniqueTs(jj,1)) ...
              uniqueTs(jj,1+h)];
          else
            newT(obj.m+h, [1:obj.m obj.m+h],jj) =...
              [(1/uniqueTs(jj,1+h)) * newT0(APsi(h,1),:,uniqueTs(jj,1))...
              (uniqueTs(jj,1+h)-1)/uniqueTs(jj,1+h)];
            if uniqueTs(jj, nTau+1+h) > 1
              cols = RowPos(st(h)==states, 1:uniqueTs(jj,nTau+1+h)-1);
              newT(obj.m+h, cols, jj) = newT(obj.m+h,cols,jj) + (1/uniqueTs(jj,1+h));
            end
          end
        end
      end
      
      newc = zeros(obj.m+nTau, Numc);
      for jj = 1:Numc
        newc(1:obj.m, jj) = newc0(:, uniquecs(jj,1));
        for  h =1:nTau
          if type(h) == 1
            newc(obj.m+h, jj) = newc0(APsi(h,1), uniquecs(jj,1));
          else
            newc(obj.m+h, jj) = (1/uniquecs(jj,1+h))*...
              newc0(APsi(h,1), uniquecs(jj,1));
          end
        end
      end
      
      newR = zeros(obj.m+nTau, obj.g, NumR);
      for jj = 1:NumR
        newR(1:obj.m,1:obj.g,jj) = newR0(:, :, uniqueRs(jj,1));
        for  h =1:nTau
          if type(h) == 1
            newR(obj.m+h,:,jj) = newR0(APsi(h,1), :, uniqueRs(jj,1));
          else
            newR(obj.m+h,:,jj) = (1/uniqueRs(jj, 1+h))*...
              newR0(APsi(h,1), :, uniqueRs(jj,1));
          end
        end
      end
      
      newZ = zeros(obj.p, obj.m+nTau,size(Zt,3));
      for jj = 1:size(Zt,3)
        newZ(:,1:obj.m,jj) = newZ0(:,:,jj);
        newZ(xi,:,jj) = zeros(length(xi), obj.m+nTau);
        for h=1:length(xi)
          cols = find(newZ0(xi(h),:,jj));
          newZ(xi(h), obj.m+nonZeroZpos(h,cols),jj) = newZ0(xi(h),cols,jj);
        end
      end
      
      obj.Z = newZ;
      obj.T = newT;
      obj.c = newc;
      obj.R = newR;
      
      obj.tau.T = newtauT;
      obj.tau.c = newtauc;
      obj.tau.R = newtauR;
      
    end
    
    function obj = checkSample(obj, y)
      assert(size(y, 1) == obj.p, 'Observed series number mismatch.');
      
      if ~obj.timeInvariant
        assert(size(y, 2) == obj.n);
      else
        % FIXME: obj.tau shouldn't be set if it's already defined
        obj.n = size(y, 2);
        taus = [repmat({ones([1 obj.n])}, [3 1]); ...
          repmat({ones([1 obj.n+1])}, [4 1])];
        obj.tau = cell2struct(taus, obj.systemParam(1:7));
      end
    end
    
    function obj = setInitial(obj, a0, P0)
      % Setter for a0 and P0 (or kappa)
      if ~isempty(a0)
        assert(size(a0, 1) == obj.m, 'a0 should be a m X 1 vector');
        obj.a0 = a0;
      end
      if nargin > 2 && ~isempty(P0)
        if size(P0, 1) == 1 % (ok if m == 1)
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
      % Set default a0, P0. Should be run before the filter/smoother after
      % any a0, P0 inputs have already been processed.
      if ~isempty(obj.a0) && ~isempty(obj.P0)
        return
      end
      
      tempT = obj.T(:, :, obj.tau.T(1));
      if all(abs(eig(tempT)) < 1)
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
        if isempty(obj.a0)
          obj.a0 = zeros(obj.m, 1);
        end
        if isempty(obj.P0)
          obj.P0 = obj.kappa * eye(obj.m);
        end
      end
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
    
    %% Maximum likelihood estimation functions
    function [negLogli, gradient] = minimizeFun(obj, theta, y)
      % Get the likelihood of
      [ss1, a0_theta, P0_theta] = obj.theta2system(theta);
      
      if obj.useGrad
        [rawLogli, rawGradient] = ss1.gradient(y, a0_theta, P0_theta);
        gradient = -rawGradient(obj.thetaMap.estimated);
      else
        [~, rawLogli] = ss1.filter(y, a0_theta, P0_theta);
        gradient = [];
      end
      
      negLogli = -rawLogli;
    end
    
    function [system, a0, P0] = theta2system(obj, theta)
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
        params.(iParam) = reshape(paramVec(obj.thetaMap.elem.(iParam)), ...
          obj.thetaMap.shape.(iParam));
      end
      params.Harvey = obj.Harvey;
      a0 = params.a0;
      P0 = params.P0;
      
      % Create new StateSpace object from parameters
      system = StateSpace(params);
    end
    
    function paramVec = getParamVec(obj)
      % TODO: Handle restrictions on the system parameters passed as non-nan values
      vec = @(M) reshape(M, [], 1);
      paramVec = [vec(obj.Z); vec(obj.d); vec(obj.H); ...
        vec(obj.T); vec(obj.c); vec(obj.R); vec(obj.Q); ...
        vec(obj.a0); vec(obj.P0)];
    end
    
    function obj = generateThetaMap(obj)
      % Generate map from parameters to theta
      parameters = {obj.Z, obj.d, obj.H, obj.T, obj.c, obj.R, obj.Q, ...
        obj.a0, obj.P0}; % Do a0, P0 need to be ss0? I don't think so.
      
      obj.thetaMap = struct;
      
      elems = cellfun(@numel, parameters);
      elemMap = arrayfun(@(x) sum(elems(1:x-1))+1:sum(elems(1:x)), ...
        1:length(elems), 'Uniform', false);
      obj.thetaMap.elem = cell2struct(elemMap', obj.systemParam');
      
      shapes = cellfun(@size, parameters, 'Uniform', false);
      obj.thetaMap.shape = cell2struct(shapes', obj.systemParam');
      
      paramVec = obj.getParamVec();
      obj.thetaMap.estimated = isnan(paramVec);
      obj.thetaMap.constrained = paramVec;
    end
    
    function [logli, gradient] = gradient(obj, y, a0, P0)
      % Returns the likelihood and the change in the likelihood given the
      % change in any system parameters that are currently set to nans.
      % a0 and P0 must be previously specified as paramters.
      
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
  
  methods (Access = protected)
    %% Display
    function displayScalarObject(obj)
      if ~isempty(obj.Harvey)
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
      
      %       propgroup = getPropertyGroups(obj);
      %       matlab.mixin.CustomDisplay.displayPropertyGroups(obj,propgroup)
    end
    
    function propgrp = getPropertyGroups(obj)
      if ~isscalar(obj)
        propgrp = getPropertyGroups@matlab.mixin.CustomDisplay(obj);
      else
        gTitle1 = 'Public Info';
        gTitle2 = 'Personal Info';
        propList1 = {'Name','JobTitle'};
        pd(1:length(obj.Password)) = '*';
        level = round(obj.Salary/100);
        propList2 = struct('Salary',...
          ['Level: ',num2str(level)],...
          'Password',pd);
        propgrp(1) = matlab.mixin.util.PropertyGroup(propList1,gTitle1);
        propgrp(2) = matlab.mixin.util.PropertyGroup(propList2,gTitle2);
      end
    end
  end
  
end