classdef StateSpace < AbstractStateSpace
  % State estimation of models with known parameters 
  %
  % Includes filtering/smoothing algorithms and maximum likelihood
  % estimation of parameters with restrictions.
  %
  % StateSpace Properties:
  %   Z, d, H - Observation equation parameters
  %   T, c, R, Q - Transition equation parameters
  %   a0, P0 - Initial value parameters
  %
  % StateSpace Methods:
  %   
  % Object construction
  % -------------------
  %   ss = StateSpace(Z, d, H, T, c, R, Q)
  %
  % The d & c parameters may be entered as empty for convenience.
  %
  % Time varrying parameters may be passed as structures with a field for
  % the parameter values and a vector indicating the timing. Using Z as an
  % example the field names would be Z.Zt and Z.tauZ.
  %
  % Filtering & smoothing
  % ---------------------
  %   [a, logl] = ss.filter(y)
  %   alpha = ss.smooth(y)
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
  
  % David Kelley, 2016-2017
  %
  % TODO (1/17/17)
  % ---------------
  %   - Add filter/smoother weight decompositions
  %   - Add IRF/historical decompositions
  
  methods (Static)
    %% Static properties
    function returnVal = useMex(newVal)
      % Static function to mimic a static class property of whether the mex 
      % functions should be used (avoids overhead of checking for them every time)
      persistent useMex_persistent;
      
      % Setter
      if nargin > 0 && ~isempty(newVal)
        useMex_persistent = newVal;
      end
      
      % Default setter
      if isempty(useMex_persistent)
        % Check mex files exist
        mexMissing = any([...
          isempty(which('mfss_mex.kfilter_uni'));
          isempty(which('mfss_mex.kfilter_multi'));
          isempty(which('mfss_mex.ksmoother_uni'));
          isempty(which('mfss_mex.ksmoother_multi'));
          isempty(which('mfss_mex.gradient_multi'))]);
        if mexMissing
          useMex_persistent = false;
          warning('MEX files not found. See .\mex\make.m');
        else
          useMex_persistent = true;
        end
      end

      % Getter
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
      
      obj.validateStateSpace();
      
      % Check if we can use the univariate filter
      slicesH = num2cell(obj.H, [1 2]);
      obj.filterUni = ~any(~cellfun(@isdiag, slicesH));
    end
    
    %% State estimation methods 
    function [a, logli, filterOut] = filter(obj, y)
      % FILTER Estimate the filtered state
      % 
      % a = StateSpace.FILTER(y) returns the filtered state given the data y. 
      %
      % a = StateSpace.FILTER(y, a0) 
      % a = StateSpace.FILTER(y, a0, P0) returns the filtered state given the 
      % data y and initial state estimates a0 and P0. 
      %
      % [a, logli] = StateSpace.FILTER(...) also returns the log-likelihood of
      % the data. 
      % [a, logli, filterOut] = StateSpace.FILTER(...) returns an additional
      % structure of intermediate computations useful in other functions. 
      
      % Make sure data matches observation dimensions
      obj.validateKFilter();
      obj = obj.checkSample(y);
      
      % Set initial values
      obj = setDefaultInitial(obj);

      % Determine which version of the filter to run
      if obj.filterUni
        if isempty(obj.A0)
          % No need for exact initial
          if obj.useMex
            [a, logli, filterOut] = obj.filter_uni_mex(y);
          else
            [a, logli, filterOut] = obj.filter_uni_m(y);
          end
        else
          % Exact initial
          [a, logli, filterOut] = obj.filter_exact_m(y);
        end
      else
        assert(isempty(obj.A0), 'Must use univarite filter with exact initialization.');
        if obj.useMex
          [a, logli, filterOut] = obj.filter_multi_mex(y);
        else
          [a, logli, filterOut] = obj.filter_multi_m(y);
        end
      end
    end
    
    function [alpha, smootherOut] = smooth(obj, y)
      % Estimate the smoothed state
      
      % Make sure data matches observation dimensions
      obj.validateKFilter();
      obj = obj.checkSample(y);
      
      % Set initial values
      obj = setDefaultInitial(obj);
      
      % Get the filtered estimates for use in the smoother
      [~, logli, filterOut] = obj.filter(y);
      
      % Determine which version of the smoother to run
      if obj.filterUni
        if isempty(obj.A0)
          % No need for exact initial
          if obj.useMex
            [alpha, smootherOut] = obj.smoother_uni_mex(y, filterOut);
          else
            [alpha, smootherOut] = obj.smoother_uni_m(y, filterOut);
          end
        else
          % Exact initial
          [alpha, smootherOut] = obj.smoother_exact_m(y, filterOut);
        end
      else
        assert(isempty(obj.A0), 'Must use univarite smoother with exact initialization.');
        if obj.useMex
          [alpha, smootherOut] = obj.smoother_multi_mex(y, filterOut);
        else
          [alpha, smootherOut] = obj.smoother_multi_m(y, filterOut);
        end
      end
      smootherOut.logli = logli;
    end
    
    function [logli, gradient] = gradient(obj, y, tm, theta)
      % Returns the likelihood and the change in the likelihood given the
      % change in any system parameters that are currently set to nans.
      
      obj.validateKFilter();
      assert(isa(tm, 'ThetaMap'));
      
      obj = obj.checkSample(y);

      obj = setDefaultInitial(obj);
      
      if nargin < 4
        theta = tm.system2theta(obj);
      end
      
      % Generate parameter gradient structure
      G = tm.parameterGradients(theta);
      [Ga0, GP0] = tm.initialValuesGradients(theta, G);
      if isempty(G.a0)
        G.a0 = Ga0;
      end
      
      if isempty(G.P0)
        G.P0 = GP0;
      end
      
      obj.filterUni = false;
      [~, logli, fOut] = obj.filter(y);
            
      if obj.useMex
        gradient = obj.gradient_multi_filter_mex(y, G, fOut);
      else
        gradient = obj.gradient_multi_filter_m(y, G, fOut);
      end
        
    end
  end
  
  methods (Hidden)
    %% Filter/smoother Helper Methods
    function obj = checkSample(obj, y)
      assert(size(y, 1) == obj.p, ...
        'Number of series does not match observation equation.');
      % TODO: check that we're not observing accumulated series before the end
      % of a period.
      
      if ~obj.timeInvariant
        % System with TVP, make sure length of taus matches data.
        assert(size(y, 2) == obj.n);
      else
        % No TVP, set n then set tau as ones vectors of that length.
        obj.n = size(y, 2);
        obj = obj.setInvariantTau();
      end
    end
    
    function validateStateSpace(obj)
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
    end
    
    function validateKFilter(obj)
      obj.validateStateSpace();
      
      % Make sure all of the parameters are known (non-nan)
      assert(~any(cellfun(@(x) any(any(any(isnan(x)))), obj.parameters)), ...
        ['All parameter values must be known. To estimate unknown '...
        'parameters, see StateSpaceEstimation']);
    end
    
    %% Filter/smoother/gradient mathematical methods
    function [a, logli, filterOut] = filter_exact_m(obj, y)
      % Filter using exact initial conditions
              
      assert(isdiag(obj.H), 'Univarite only!');
      
      % Preallocate
      % Note Pd is the "diffuse" P matrix (P_\infty).
      a = nan(obj.m, obj.n+1);
      v = nan(obj.p, obj.n);
      
      Pd = nan(obj.m, obj.m, obj.n+1);
      Pstar = nan(obj.m, obj.m, obj.n+1);
      Fd = nan(obj.p, obj.n);
      Fstar = nan(obj.p, obj.n);
      
      Md = nan(obj.m, obj.p, obj.n);
      Mstar = nan(obj.m, obj.p, obj.n);
      
      LogL = zeros(obj.p, obj.n);
      
      % Initialize
      a(:,1) = obj.a0;
      Pd(:,:,1) = obj.A0 * obj.A0';
      Pstar(:,:,1) = obj.R0 * obj.Q0 * obj.R0';
      
      ii = 0;
      % Initial recursion
      while ~all(all(Pd(:,:,ii+1) == 0))
        ii = ii + 1;
        ind = find( ~isnan(y(:,ii)) );
        
        ati = a(:,ii);
        Pstarti = Pstar(:,:,ii);
        Pdti = Pd(:,:,ii);
        for jj = ind'
          Zjj = obj.Z(jj,:,obj.tau.Z(ii));
          v(jj,ii) = y(jj, ii) - Zjj * ati - obj.d(jj,obj.tau.d(ii));
          
          Fd(jj,ii) = Zjj * Pdti * Zjj';
          Fstar(jj,ii) = Zjj * Pstarti * Zjj' + obj.H(jj,jj,obj.tau.H(ii));
          
          Md(:,jj,ii) = Pdti * Zjj';
          Mstar(:,jj,ii) = Pstarti * Zjj';
          
          if Fd(jj,ii) ~= 0
            % F diffuse nonsingular
            ati = ati + Md(:,jj,ii) ./ Fd(jj,ii) * v(jj,ii);
            
            Pstarti = Pstarti - Md(:,jj,ii) * Md(:,jj,ii)' * Fstar(jj,ii) * (Fd(jj,ii).^-2) - ...
              (Mstar(:,jj,ii) * Md(:,jj,ii)' + Md(:,jj,ii) * Mstar(:,jj,ii)') ./ Fd(jj,ii);
            
            Pdti = Pdti - Md(:,jj,ii) ./ Fd(jj,ii) * Md(:,jj,ii)';
          else
            % F diffuse = 0
            ati = ati + Mstar(:,jj,ii) ./ Fstar(jj,ii) * v(jj,ii);
            
            Pstarti = Pstarti - Mstar(:,jj,ii) ./ Fstar(jj,ii) * Mstar(:,jj,ii)';
          end
        end
        
        LogL(jj,ii) = log(Fstar(jj,ii) + Fd(jj,ii)) + (v(jj,ii)^2) / Fstar(jj,ii);
        
        Tii = obj.T(:,:,obj.tau.T(ii));
        a(:,ii+1) = Tii * ati + obj.c(:,obj.tau.c(ii));
        
        Pd(:,:,ii+1)  = Tii * Pdti * Tii';
        Pstar(:,:,ii+1) = Tii * Pstarti * Tii' + ...
          obj.R(:,:,obj.tau.R(ii)) * obj.Q(:,:,obj.tau.Q(ii)) * obj.R(:,:,obj.tau.R(ii))';
      end
      
      dt = ii;
      
      F = Fstar;
      M = Mstar;
      P = Pstar;
      
      % Normal kalman filter
      for ii = dt+1:obj.n
        ind = find( ~isnan(y(:,ii)) );
        ati    = a(:,ii);
        Pti    = P(:,:,ii);
        for jj = ind'
          Zjj = obj.Z(jj,:,obj.tau.Z(ii));
          
          v(jj,ii) = y(jj,ii) - Zjj * ati - obj.d(jj,obj.tau.d(ii));
          
          F(jj,ii) = Zjj * Pti * Zjj' + obj.H(jj,jj,obj.tau.H(ii));
          M(:,jj,ii) = Pti * Zjj';
          
          LogL(jj,ii) = (log(F(jj,ii)) + (v(jj,ii)^2) / F(jj,ii));
          
          ati = ati + M(:,jj,ii) / F(jj,ii) * v(jj,ii);
          Pti = Pti - M(:,jj,ii) / F(jj,ii) * M(:,jj,ii)';
        end
        
        Tii = obj.T(:,:,obj.tau.T(ii));
        
        a(:,ii+1) = Tii * ati + obj.c(:,obj.tau.c(ii));
        P(:,:,ii+1) = Tii * Pti * Tii' + ...
          obj.R(:,:,obj.tau.R(ii)) * obj.Q(:,:,obj.tau.Q(ii)) * obj.R(:,:,obj.tau.R(ii))';
      end

      logli = -(0.5 * sum(sum(isfinite(y)))) * log(2 * pi) - 0.5 * sum(sum(LogL));
      
      filterOut = obj.compileStruct(a, P, v, F, M, Pd, Fd, Md, dt);
    end
    
    function [alpha, smootherOut] = smoother_exact_m(obj, y, fOut)
      % Univariate smoother
      
      alpha = zeros(obj.m, obj.n);
      eta   = zeros(obj.g, obj.n);
      r     = zeros(obj.m, obj.n);
      N     = zeros(obj.m, obj.m, obj.n+1);
      
      rti = zeros(obj.m,1);
      Nti = zeros(obj.m,obj.m);
      for ii = obj.n:-1:fOut.dt+1
        ind = find( ~isnan(y(:,ii)) );
        
        for jj = ind'
          Lti = eye(obj.m) - fOut.M(:,jj,ii) * ...
            obj.Z(jj,:,obj.tau.Z(ii)) / fOut.F(jj,ii);
          rti = obj.Z(jj,:,obj.tau.Z(ii))' / ...
            fOut.F(jj,ii) * fOut.v(jj,ii) + Lti' * rti;
          Nti = obj.Z(jj,:,obj.tau.Z(ii))' / ...
            fOut.F(jj,ii) * obj.Z(jj,:,obj.tau.Z(ii)) ...
            + Lti' * Nti * Lti;
        end
        r(:,ii) = rti;
        N(:,:,ii) = Nti;
        
        alpha(:,ii) = fOut.a(:,ii) + fOut.P(:,:,ii) * r(:,ii);
        eta(:,ii) = obj.Q(:,:,obj.tau.Q(ii)) * obj.R(:,:,obj.tau.R(ii))' * r(:,ii);
        
        rti = obj.T(:,:,obj.tau.T(ii))' * rti;
        Nti = obj.T(:,:,obj.tau.T(ii))' * Nti * obj.T(:,:,obj.tau.T(ii));
      end
      
      r0 = r;
      r1 = zeros(obj.m, fOut.dt+1);
      
      % Exact initial smoother
      for ii = fOut.dt:-1:1
        r0ti = r0(:,ii+1);
        r1ti = r1(:,ii+1);
        
        ind = find( ~isnan(y(:,ii)) );
        for jj = ind'
          Zjj = obj.Z(jj,:,obj.tau.Z(ii));
          
          if fOut.Fd(jj,ii) ~= 0 % ~isequal(Finf(ind(jj),ii),0)
            % Diffuse case
            Ldti = eye(obj.m) - fOut.Md(:,jj,ii) * Zjj / fOut.Fd(jj,ii);
            L0ti = (fOut.Md(:,jj,ii) * fOut.F(jj,ii) / fOut.Fd(jj,ii) + ...
              fOut.M(:,jj,ii)) * Zjj / fOut.Fd(jj,ii);
            
            r1ti = Zjj' / fOut.Fd(jj,ii) * fOut.v(jj,ii) - L0ti' * r0ti + Ldti' * r1ti;
            
            r0ti = Ldti' * r0ti;
          else
            % Known
            Lstarti = eye(m) - fOut.M(:,jj,ii) * Zjj / fOut.F(ind(jj),ii);
            r0ti = Zjj' / fOut.F(jj,ii) * v(jj,ii) + Lstarti' * r0ti;
          end
        end
        r0(:,ii) = r0ti;
        r1(:,ii) = r1ti;
        
        alpha(:,ii) = fOut.a(:,ii) + fOut.P(:,:,ii) * r0(:,ii) + ...
          fOut.Pd(:,:,ii) * r1(:,ii);
        
        eta(:,ii) = obj.Q(:,:,obj.tau.Q(ii)) * obj.R(:,:,obj.tau.R(ii))' * r0(:,ii);
        
        r0ti = obj.T(:,:,obj.tau.T(ii))' * r0ti;
        r1ti = obj.T(:,:,obj.tau.T(ii))' * r1ti;
      end
      
      Pstar0 = obj.R0 * obj.Q0 * obj.R0';
      if fOut.dt > 0
        Pd0 = obj.A0 * obj.A0';
        a0tilde = obj.a0 + Pstar0 * r0ti + Pd0 * r1ti;
      else
        a0tilde = obj.a0 + P0 * rti;
      end
      
      smootherOut = obj.compileStruct(alpha, eta, r, N, a0tilde);
    end
    
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
      
      P0 = obj.R0 * obj.Q0 * obj.R0';
      
      a(:,1) = obj.T(:,:,obj.tau.T(1)) * obj.a0 + obj.c(:,obj.tau.c(1));
      P(:,:,1) = obj.T(:,:,obj.tau.T(1)) * P0 * obj.T(:,:,obj.tau.T(1))' ...
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
      P0 = obj.R0 * obj.Q0 * obj.R0';
      
      [LogL, a, P, v, F, M, K, L] = mfss_mex.kfilter_uni(y, ...
        obj.Z, obj.tau.Z, obj.d, obj.tau.d, obj.H, obj.tau.H, ...
        obj.T, obj.tau.T, obj.c, obj.tau.c, obj.R, obj.tau.R, obj.Q, obj.tau.Q, ...
        obj.a0, P0);
      
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

      P0 = obj.R0 * obj.Q0 * obj.R0';
      
      a(:,1) = obj.T(:,:,obj.tau.T(1)) * obj.a0 + obj.c(:,obj.tau.c(1));
      P(:,:,1) = obj.T(:,:,obj.tau.T(1)) * P0 * obj.T(:,:,obj.tau.T(1))'...
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
      P0 = obj.R0 * obj.Q0 * obj.R0';
      
      [LogL, a, P, v, F, M, K, L, w, Finv] = mfss_mex.kfilter_multi(y, ...
        obj.Z, obj.tau.Z, obj.d, obj.tau.d, obj.H, obj.tau.H, ...
        obj.T, obj.tau.T, obj.c, obj.tau.c, obj.R, obj.tau.R, obj.Q, obj.tau.Q, ...
        obj.a0, P0);
      
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
      P0 = obj.R0 * obj.Q0 * obj.R0';
      
      a0tilde = obj.a0 + P0 * rti;
      
      smootherOut = obj.compileStruct(alpha, eta, r, N, a0tilde);
    end
    
    function [alpha, smootherOut] = smoother_uni_mex(obj, y, fOut)
      % Call mex function kfilter_uni
      P0 = obj.R0 * obj.Q0 * obj.R0';
      
      [alpha, eta, r, N, a0tilde] = mfss_mex.ksmoother_uni(y, ...
        obj.Z, obj.tau.Z, obj.H, obj.tau.H, ...
        obj.T, obj.tau.T, obj.R, obj.tau.R, obj.Q, obj.tau.Q, ...
        fOut.a, fOut.P, fOut.v, fOut.F, fOut.M, fOut.L, ...
        obj.a0, P0);
      
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
      
      P0 = obj.R0 * obj.Q0 * obj.R0';
      a0tilde = obj.a0 + P0 * obj.T(:,:,obj.tau.T(1))'*r(:,1);
      smootherOut = obj.compileStruct(alpha, eta, epsilon, r, N, a0tilde, V, J);
    end
    
    function [alpha, smootherOut] = smoother_multi_mex(obj, y, fOut)
      % Call mex function kfilter_uni
      P0 = obj.R0 * obj.Q0 * obj.R0';
      
      [alpha, eta, epsilon, r, N, a0tilde, V, J] = mfss_mex.ksmoother_multi(y, ...
        obj.Z, obj.tau.Z, obj.H, obj.tau.H, ...
        obj.T, obj.tau.T, obj.R, obj.tau.R, obj.Q, obj.tau.Q, ...
        fOut.a, fOut.P, fOut.v, fOut.M, fOut.K, fOut.L, fOut.Finv, ...
        obj.a0, P0);
      
      smootherOut = obj.compileStruct(alpha, eta, epsilon, r, N, a0tilde, V, J);
    end
    
    function gradient = gradient_multi_filter_m(obj, y, G, fOut)
      % Gradient algorithm from Diasuke Nagakura (SSRN # 1634552).
      %
      % Note that G.x is 3D for everything except a and P (and a0 and P0). 
      % G.a and G.P denote the one-step ahead gradient (i.e., G_\theta(a_{t+1}))
      
      nTheta = size(G.T, 1);
      
      Nm = (eye(obj.m^2) + obj.genCommutation(obj.m));
      vec = @(M) reshape(M, [], 1);
      
      % Compute partial results that have less time-variation (even with TVP)
      kronRR = zeros(obj.g*obj.g, obj.m*obj.m, max(obj.tau.R));
      for iR = 1:max(obj.tau.R)
        kronRR(:, :, iR) = kron(obj.R(:,:,iR)', obj.R(:,:,iR)');
      end
      
      [tauQRrows, ~, tauQR] = unique([obj.tau.R obj.tau.Q], 'rows');
      kronQRI = zeros(obj.g * obj.m, obj.m * obj.m, max(tauQR));
      for iQR = 1:max(tauQR)
        kronQRI(:, :, iQR) = kron(obj.Q(:,:,tauQRrows(iQR, 2)) * obj.R(:,:,tauQRrows(iQR, 1))', ...
          eye(obj.m));
      end
      
      % Initial period: G.a and G.P capture effects of a0, T
      P0 = obj.R0 * obj.Q0 * obj.R0';

      G.a = G.a0 * obj.T(:,:,obj.tau.T(1))' + ...
        G.c(:, :, obj.tau.c(1)) + ... % Yes, G.c is 3D.
        G.T(:,:,obj.tau.T(1)) * kron(obj.a0, eye(obj.m));
      G.P = G.P0 * kron(obj.T(:,:,obj.tau.T(1))', obj.T(:,:,obj.tau.T(1))') + ...
        G.Q(:,:,obj.tau.Q(1)) * kron(obj.R(:,:,obj.tau.R(1))', obj.R(:,:,obj.tau.R(1))') + ...
        (G.T(:,:,obj.tau.T(1)) * kron(P0 * obj.T(:,:,obj.tau.T(1))', eye(obj.m)) + ...
        G.R(:,:,obj.tau.R(1)) * kron(obj.Q(:,:,obj.tau.Q(1)) * ...
        obj.R(:,:,obj.tau.R(1))', eye(obj.m))) * ...
          Nm;
      
      % Recursion through time periods
      W_base = logical(sparse(eye(obj.p)));
      
      grad = zeros(obj.n, nTheta);
      for ii = 1:obj.n
        ind = ~isnan(y(:,ii));
        W = W_base((ind==1),:);
        kronWW = kron(W', W');
        
        Zii = W * obj.Z(:, :, obj.tau.Z(ii));
        
        ww = fOut.w(ind,ii) * fOut.w(ind,ii)';
        Mv = fOut.M(:,:,ii) * fOut.v(:, ii);
        
        grad(ii, :) = G.a * Zii' * fOut.w(ind,ii) + ...
          0.5 * G.P * vec(Zii' * ww * Zii - Zii' * fOut.Finv(ind,ind,ii) * Zii) + ...
          G.d(:,:,obj.tau.d(ii)) * W' * fOut.w(ind,ii) + ...
          G.Z(:,:,obj.tau.Z(ii)) * vec(W' * (fOut.w(ind,ii) * fOut.a(:,ii)' + ...
            fOut.w(ind,ii) * Mv' - fOut.M(:,ind,ii)')) + ...
          0.5 * G.H(:,:,obj.tau.H(ii)) * kronWW * vec(ww - fOut.Finv(ind,ind,ii));
        
        % Set t+1 values
        PL = fOut.P(:,:,ii) * fOut.L(:,:,ii)';
        
        kronZwL = kron(Zii' * fOut.w(ind,ii), fOut.L(:,:,ii)');
        kronPLw = kron(PL, fOut.w(:,ii));
        kronaMvK = kron(fOut.a(:,ii) + Mv, fOut.K(:,:,ii)');
        kronwK = kron(fOut.w(:,ii), fOut.K(:,:,ii)');
        kronAMvI = kron(fOut.a(:,ii) + Mv, eye(obj.m));
        
        G.a = G.a * fOut.L(:,:,ii)' + ...
          G.P * kronZwL + ...
          G.c(:,:,obj.tau.c(ii+1)) - ...
          G.d(:,:,obj.tau.d(ii)) * fOut.K(:,:,ii)' + ...
          G.Z(:,:,obj.tau.Z(ii)) * (kronPLw - kronaMvK) - ...
          G.H(:,:,obj.tau.H(ii)) * kronwK + ...
          G.T(:,:,obj.tau.T(ii+1)) * kronAMvI;
        
        kronLL = kron(fOut.L(:,:,ii)', fOut.L(:,:,ii)');
        kronKK = kron(fOut.K(:,:,ii)', fOut.K(:,:,ii)');
        kronPLI = kron(PL, eye(obj.m));
        kronPLK = kron(PL, fOut.K(:,:,ii)');
        
        G.P = G.P * kronLL + ...
          G.H(:,:,obj.tau.H(ii)) * kronKK + ...
          G.Q(:,:,obj.tau.Q(ii+1)) * kronRR(:,:, obj.tau.R(ii+1)) + ...
          (G.T(:,:,obj.tau.T(ii+1)) * kronPLI - ...
            G.Z(:,:,obj.tau.Z(ii)) * kronPLK + ...
            G.R(:,:,obj.tau.R(ii+1)) * kronQRI(:, :, tauQR(ii+1))) * ...
            Nm;
      end
      
      gradient = sum(grad, 1)';
    end
    
    function gradient = gradient_multi_filter_mex(obj, y, G, fOut)
      P0 = obj.R0 * obj.Q0 * obj.R0';
      
      ssStruct = struct('Z', obj.Z, 'd', obj.d, 'H', obj.H, ...
        'T', obj.T, 'c', obj.c, 'R', obj.R, 'Q', obj.Q, ...
        'a0', obj.a0, 'P0', P0, ...
        'tau', obj.tau, ...
        'p', obj.p, 'm', obj.m, 'g', obj.g, 'n', obj.n);
      
      gradient = mfss_mex.gradient_multi(y, ssStruct, G, fOut);
    end
    
    %% Initialization
    function obj = setDefaultInitial(obj)
      % Set default a0 and P0.
      % Run before filter/smoother after a0 & P0 inputs have been processed
      
      if ~obj.usingDefaulta0 && ~obj.usingDefaultP0
        % User provided a0 and P0.
        return
      end
      
      % Find stationary states and compute the unconditional mean and variance
      % of them using the parts of T, c, R and Q. For the nonstationary states,
      % set up the A0 selection matrix. 
      [stationary, nonstationary] = obj.findStationaryStates();
      
      tempT = obj.T(stationary, stationary, obj.tau.T(1));
      assert(all(abs(eig(tempT)) < 1));
      
      select = eye(obj.m);
      mStationary = length(stationary);

      if obj.usingDefaulta0
        obj.a0 = zeros(obj.m, 1);
        
        a0temp = (eye(mStationary) - tempT) \ obj.c(stationary, obj.tau.c(1));
        obj.a0(stationary) = a0temp;
      end
      
      if obj.usingDefaultP0
        obj.A0 = select(:, nonstationary);
        obj.R0 = select(:, stationary);
        
        tempR = obj.R(stationary, :, obj.tau.R(1));
        tempQ = obj.Q(:, :, obj.tau.Q(1));
        try
          obj.Q0 = reshape((eye(mStationary^2) - kron(tempT, tempT)) \ ...
            reshape(tempR * tempQ * tempR', [], 1), ...
            mStationary, mStationary);
        catch ex
          % If the state is large, try making it sparse
          if strcmpi(ex.identifier, 'MATLAB:array:SizeLimitExceeded')
            tempT = sparse(tempT);
            obj.Q0 = full(reshape((speye(mStationary^2) - kron(tempT, tempT)) \ ...
              reshape(tempR * tempQ * tempR', [], 1), ...
              mStationary, mStationary));
          else
            rethrow(ex);
          end
        end
        
      end
    end
    
    function [stationary, nonstationary] = findStationaryStates(obj)
      % Find which states have stationary distributions given the T matrix.
      [V, D] = eig(obj.T(:,:,1));
      bigEigs = abs(diag(D)) >= 1;
      
      nonstationary = find(any(V(:, bigEigs), 2));
      
      % I think we don't need a loop here to find other states that have 
      % loadings on the nonstationary states (the eigendecomposition does this 
      % for us) but I'm not sure.
      stationary = setdiff(1:obj.m, nonstationary);      

      assert(all(abs(eig(obj.T(stationary,stationary,1))) < 1), ...
        ['Stationary section of T isn''t actually stationary. \n' ... 
        'Likely development error.']);
    end
  end
end
