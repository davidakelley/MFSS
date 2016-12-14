classdef StateSpace < AbstractStateSpace
  % State estimation of models with known parameters 
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
  
  % David Kelley, 2016
  %
  % TODO (12/9/16)
  % ---------------
  %   - TVP in gradient function 
  %   - mex version of the gradient function (?)
  %   - Add filter/smoother weight decompositions
  
  properties
    % Use univarite filter if appropriate (H is diagonal)
    filterUni
  end
  
  methods (Static)
    %% Static properties
    function returnVal = useMex(newVal)
      % Static function to mimic a static class property of whether the mex 
      % functions should be used (avoids overhead of checking for them every time)
      persistent useMex_persistent;
      if nargin > 0 && ~isempty(newVal)
        useMex_persistent = newVal;
      end
      
      if isempty(useMex_persistent)
        % Check mex files exist
        mexMissing = any([isempty(which('ss_mex.kfilter_uni'));
          isempty(which('ss_mex.kfilter_multi'));
          isempty(which('ss_mex.kfilter_uni'));
          isempty(which('ss_mex.kfilter_multi'))]);
        if mexMissing
          useMex_persistent = false;
          warning('MEX files not found. See .\mex\make.m');
        else
          useMex_persistent = true;
        end
      end

      returnVal = useMex_persistent;
    end
  end
  
  methods
    %% Constructor
    function obj = StateSpace(Z, d, H, T, c, R, Q)
      % StateSpace constructor
      % Pass state parameters to construct new object (or pass a structure
      % containing the neccessary parameters)
      obj = obj@AbstractStateSpace(Z, d, H, T, c, R, Q);
      
      obj.validateKFilter();
      
      % Check if we can use the univariate filter
      diagH = true;
      for iH = 1:size(obj.H, 3)
        if ~isdiag(obj.H(:,:,iH))
          diagH = false;
        end
      end
      obj.filterUni = diagH;
    end
    
    %% State estimation methods 
    function [a, logli, filterOut] = filter(obj, y, a0, P0)
      % Estimate the filtered state
      
      % Make sure data matches observation dimensions
      obj = obj.checkSample(y);
      
      % Set initial values
      if nargin > 3
        obj = obj.setInitial(a0, P0);
      elseif nargin > 2
        obj = obj.setInitial(a0);
      end
      obj = setDefaultInitial(obj);

      % Determine which version of the filter to run
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
      % Estimate the smoothed state
      
      % Make sure data matches observation dimensions
      obj = obj.checkSample(y);
      
      % Set initial values
      if nargin > 3
        obj = obj.setInitial(a0, P0);
      elseif nargin > 2
        obj = obj.setInitial(a0);
      end
      obj = setDefaultInitial(obj);
      
      % Get the filtered estimates for use in the smoother
      [~, logli, filterOut] = obj.filter(y);
      
      % Determine which version of the smoother to run
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
    
    function [logli, gradient] = gradient(obj, y, tm, a0, P0)
      % Returns the likelihood and the change in the likelihood given the
      % change in any system parameters that are currently set to nans.
      
      assert(isa(tm, 'ThetaMap'));
      
      obj = obj.checkSample(y);
      if nargin > 3
        obj = obj.setInitial(a0, P0);
      end
      obj = setDefaultInitial(obj);
      
      theta = tm.system2theta(obj);
      
      [logli, gradient] = obj.gradient_multi_filter(y, tm, theta);
    end
  end
  
  methods(Hidden)
    %% Filter/smoother Helper Methods
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
        
        firstTau = structfun(@(x) x(1), obj.tau);
        assert(all(firstTau == 1), 'First element of tau must be 1.');
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
      
       % Make sure all of the parameters are known (non-nan)
      assert(~any(cellfun(@(x) any(any(any(isnan(x)))), obj.parameters)), ...
        ['All parameter values must be known. To estimate unknown '...
        'parameters, see StateSpaceEstimation']);
    end
    
    %% Filter/smoother/gradient mathematical methods
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
    
    function [logli, gradient] = gradient_multi_filter(obj, y, tm, theta)
      % Gradient algorithm from Diasuke Nagakura (SSRN # 1634552).
      
      % TODO: First think about G.x carefully.
      assert(obj.timeInvariant, 'TVP gradient not developed yet.'); 
      
      obj.filterUni = false;
      [~, logli, fOut] = obj.filter(y);
      
      G = tm.parameterGradients(theta);
      [Ga0, GP0] = tm.initialValuesGradients(theta, G);
      if ~isempty(Ga0)
        G.a = Ga0;
      else
        G.a = G.a0;
        G = rmfield(G, 'a0');
      end
      if ~isempty(GP0)
        G.P = GP0;
      else
        G.P = G.P0;
        G = rmfield(G, 'P0');
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
      W_base = logical(sparse(eye(obj.p)));
      
      gradient = zeros(tm.nTheta, 1);
      for ii = 1:obj.n
        ind = ~isnan(y(:,ii));
        W = W_base((ind==1),:);
        kronWW = kron(W', W');
        
        Zii = W * obj.Z(:, :, obj.tau.Z(ii));
        
        ww = fOut.w(ind,ii) * fOut.w(ind,ii)';
        Mv = fOut.M(:,:,ii) * fOut.v(:, ii);
        
        gradient = gradient + ...
          G.a * Zii' * fOut.w(ind,ii) + ...
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
    end
    
    %% Initialization
    function obj = setDefaultInitial(obj)
      % Set default a0 and P0.
      % Run before filter/smoother after a0 & P0 inputs have been processed
      if ~obj.usingDefaulta0 && ~obj.usingDefaultP0
        % User provided a0 and P0.
        return
      end
      
      tempT = obj.T(:, :, obj.tau.T(1));
      if all(abs(eig(tempT)) < 1)
        % System is stationary. Compute unconditional estimate of state
        if obj.usingDefaulta0
          obj.a0 = (eye(obj.m) - tempT) \ obj.c(:, obj.tau.c(1));
        end
        if obj.usingDefaultP0
          tempR = obj.R(:, :, obj.tau.R(1));
          tempQ = obj.Q(:, :, obj.tau.Q(1));
          try
            obj.P0 = reshape((eye(obj.m^2) - kron(tempT, tempT)) \ ...
              reshape(tempR * tempQ * tempR', [], 1), ...
              obj.m, obj.m);
          catch ex
            % If the state is large, try making it sparse
            if strcmpi(ex.identifier, 'MATLAB:array:SizeLimitExceeded')
              tempT = sparse(tempT);
              obj.P0 = full(reshape((speye(obj.m^2) - kron(tempT, tempT)) \ ...
                reshape(tempR * tempQ * tempR', [], 1), ...
                obj.m, obj.m));
            else
              rethrow(ex);
            end
          end
            
        end
      else
        % Nonstationary case: use large kappa diffuse initialization
        if obj.usingDefaulta0
          obj.a0 = zeros(obj.m, 1);
        end
        if obj.usingDefaultP0
          obj.P0 = obj.kappa * eye(obj.m);
        end
      end
    end
  end
end
