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
      if nargin == 0
        superArgs = {};
      else
        superArgs = {Z, d, H, T, c, R, Q};
      end
      obj = obj@AbstractStateSpace(superArgs{:});
      if nargin == 0
        return;
      end
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
      % [a, logli] = StateSpace.FILTER(...) also returns the log-likelihood of
      % the data. 
      % [a, logli, filterOut] = StateSpace.FILTER(...) returns an additional
      % structure of intermediate computations useful in other functions. 
      
      [obj, y] = obj.prepareFilter(y);
      
      assert(~any(obj.H(1:obj.p+1:end) < 0), 'Negative error variance.');
 
      % Call the filter
      if obj.useMex
        [a, logli, filterOut] = obj.filter_mex(y);
      else
        [a, logli, filterOut] = obj.filter_m(y);
      end
    end
    
    function [alpha, smootherOut, filterOut] = smooth(obj, y)
      % SMOOTH Estimate the smoothed state
      % 
      % alpha = StateSpace.SMOOTH(y) returns the smoothed state given the data y. 
      % [alpha, smootherOut] = StateSpace.SMOOTH(...) also returns additional 
      % quantities computed in the smoothed state calculation. 
      % [alpha, smootherOut, filterOut] = StateSpace.SMOOTH(...) returns 
      % additional quantities computed in the filtered state calculation.
        
      [obj, y] = obj.prepareFilter(y);

      % Get the filtered estimates for use in the smoother
      [~, logli, filterOut] = obj.filter(y);
      
      % Determine which version of the smoother to run
      if obj.useMex
        [alpha, smootherOut] = obj.smoother_mex(y, filterOut);
      else
        [alpha, smootherOut] = obj.smoother_m(y, filterOut);
      end
      
      smootherOut.logli = logli;
    end
    
    function [logli, gradient] = gradient(obj, y, tm, theta)
      % Returns the likelihood and the change in the likelihood given the
      % change in any system parameters that are currently set to nans.

      % Handle inputs
      assert(isa(tm, 'ThetaMap'), 'tm must be a ThetaMap.');
      if nargin < 4
        theta = tm.system2theta(obj);
      end
      assert(all(size(theta) == [tm.nTheta 1]), ...
        'theta must be a nTheta x 1 vector.');

      ssMulti = obj;
      [obj, yUni, factorC] = obj.prepareFilter(y);
      
      % Generate parameter gradient structure
      GMulti = tm.parameterGradients(theta);
      [GMulti.a1, GMulti.P1] = tm.initialValuesGradients(theta, GMulti);

      % Transform to univariate filter gradients
      if factorC ~= eye(obj.p)
        GUni = obj.factorGradient(GMulti, ssMulti, factorC);
      else
        GUni = GMulti;
      end
      
      % Run filter with extra outputs, comput gradient
      [~, logli, fOut, ftiOut] = obj.filter_detail_m(yUni);      
      gradient = obj.gradient_filter_m(GUni, fOut, ftiOut);
    end
    
    function [dataDecomposition, constContrib] = decompose_smoothed(obj, y, decompPeriods)
      % Decompose the smoothed states by data contributions
      %
      % Output ordered by (state, observation, contributingPeriod, effectPeriod)
      
      [obj, y] = obj.prepareFilter(y);

      if nargin < 3
        decompPeriods = 1:obj.n;
      end
      
      [~, ~, fOut] = obj.filter(y);
      if obj.useMex
        [alpha, sOut] = obj.smoother_mex(y, fOut);
      else
        [alpha, sOut] = obj.smoother_m(y, fOut);
      end
      
      [alphaW, constContrib] = obj.smoother_weights(y, fOut, sOut, decompPeriods);
      
      % Apply weights to data
      cleanY = y;
      cleanY(isnan(y)) = 0;
      
      nDecomp = length(decompPeriods);
      dataDecomposition = zeros(obj.m, obj.p, obj.n, nDecomp);
      % Loop over periods effected
      for iT = 1:nDecomp
        % Loop over contributing periods
        for cT = 1:obj.n 
          dataDecomposition(:,:,cT,iT) = alphaW(:,:,cT,iT) .* repmat(cleanY(:, cT)', [obj.m, 1]);
        end
      end
      
      % Weights come out ordered (state, observation, origin, effect) so
      % collapsing the 2nd and 3rd dimension should give us a time-horizontal
      % sample. 
      dataContrib = squeeze(sum(sum(dataDecomposition, 2), 3));
      if obj.m == 1
        dataContrib = dataContrib';
      end
      alpha_test = dataContrib + constContrib;
      err = alpha(:,decompPeriods) - alpha_test;
      assert(max(max(abs(err))) < 0.001, 'Did not recover data from decomposition.');
    end
    
    %% Utilties
    function obj = setDefaultInitial(obj, reset)
      % Set default a0 and P0.
      % Run before filter/smoother after a0 & P0 inputs have been processed
      
      if nargin < 2 
        % Option to un-set initial values. 
        reset = false;
      end
      if reset
        obj.usingDefaulta0 = true;
        obj.usingDefaultP0 = true;
        obj.a0 = [];
        obj.A0 = [];
        obj.R0 = [];
        obj.Q0 = [];
      end        
      
      if ~obj.usingDefaulta0 && ~obj.usingDefaultP0
        % User provided a0 and P0.
        return
      end
      
      if isempty(obj.tau)
        obj = obj.setInvariantTau();
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
  end
  
  methods (Hidden)
    %% Filter/smoother Helper Methods
    function [obj, y, factorC] = prepareFilter(obj, y)
      % Make sure data matches observation dimensions
      obj.validateKFilter();
      obj = obj.checkSample(y);
      
      % Set initial values
      obj = setDefaultInitial(obj);

      % Handle multivariate series
      [obj, y, factorC] = obj.factorMultivariate(y);
    end
    
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
    
    function [ssUni, yUni, factorC] = factorMultivariate(obj, y)
      % Compute new Z and H matricies so the univariate treatment can be applied
      
      % If H is already diagonal, do nothing
      if arrayfun(@(x) isdiag(obj.H(:,:,x)), 1:size(obj,3))
        ssUni = obj;
        ssUni.filterUni = true;
        yUni = y;
        factorC = eye(obj.p);
        return
      end
      
      [uniqueOut, ~, newTauH] = unique([obj.tau.H ~isnan(y')], 'rows');
      oldTauH = uniqueOut(:,1);
      obsPattern = uniqueOut(:,2:end);
      
      % Create factorizations
      maxTauH = max(newTauH);
      factorC = zeros(size(obj.H, 1), size(obj.H, 2), maxTauH);
      newHmat = zeros(size(obj.H, 1), size(obj.H, 2), maxTauH);
      for iH = 1:maxTauH
        ind = logical(obsPattern(iH, :));
        [factorC(ind,ind,iH), newHmat(ind,ind,iH)] = ldl(obj.H(ind,ind,oldTauH(iH)), 'lower');
        assert(isdiag(newHmat(ind,ind,iH)), 'ldl returned non-diagonal d matrix.');
      end
      newH = struct('Ht', abs(newHmat), 'tauH', newTauH);      
      
      yUni = nan(size(y));
      inds = logical(obsPattern(newTauH, :));
      for iT = 1:size(y,2)
        % Transform observations
        yUni(inds(iT,:),iT) = factorC(inds(iT,:),inds(iT,:),newTauH(iT)) * y(inds(iT,:),iT);
      end
      
      % We may need to create more slices of Z. 
      % If a given slice of Z is used when two different H matricies are used,
      % we need to use the correct C matrix to factorize it at each point. 
      [uniqueOut, ~, newTauZ] = unique([obj.tau.Z newTauH ~isnan(y')], 'rows');
      oldTauZ = uniqueOut(:,1);
      correspondingNewHOldZ = uniqueOut(:,2);
      obsPattern = uniqueOut(:,3:end);
      
      newZmat = zeros([size(obj.Z, 1) size(obj.Z, 2), max(newTauZ)]);
      for iZ = 1:max(newTauZ)
        ind = logical(obsPattern(iZ,:));
        newZmat(ind,:,iZ) = factorC(ind,ind,correspondingNewHOldZ(iZ)) \ obj.Z(ind,:,oldTauZ(iZ));
      end
      newZ = struct('Zt', newZmat, 'tauZ', newTauZ);
      
      % Same thing for d
      [uniqueOut, ~, newTaud] = unique([obj.tau.d newTauH ~isnan(y')], 'rows');
      oldTaud = uniqueOut(:,1);
      correspondingNewHOldd = uniqueOut(:,2);
      obsPattern = uniqueOut(:,3:end);

      newdmat = zeros([size(obj.d, 1) max(newTaud)]);
      for id = 1:max(newTaud)
        ind = logical(obsPattern(id,:));
        newdmat(ind,id) = factorC(ind,ind,correspondingNewHOldd(id)) \ obj.d(ind,oldTaud(id));
      end
      newd = struct('dt', newdmat, 'taud', newTaud);
      
      [~, ~, ~, T, c, R, Q] = obj.getInputParameters();
      
      ssUni = StateSpace(newZ, newd, newH, T, c, R, Q);
      
      % Set initial values
      P0 = obj.R0 * obj.Q0 * obj.R0';
      P0(obj.A0 * obj.A0' == 1) = Inf;
      ssUni = ssUni.setInitial(obj.a0, P0);
    end
    
    function Guni = factorGradient(obj, GMulti, ssMulti, factorC)
      % factorGradient transforms the gradient of the multivariate model to the
      % univarite model given the univariate parameters and the factorzation
      % matrix. 
      
      assert(size(GMulti.H, 3) == 1, 'TVP not yet developed.');
      
      Guni = GMulti;
      
      % Get the indexes we'll need to separate out GC and GHstar
      indexMat = reshape(1:obj.p^2, [obj.p obj.p]);
      diagCols = diag(indexMat);
      lowerTril = tril(indexMat, -1);
      lowerDiagCols = lowerTril(lowerTril ~= 0);
      
      % Find GC and GHstar 
      Kp = AbstractSystem.genCommutation(obj.p);
      Np = eye(obj.p^2) + Kp;
      Cmult = kron(obj.H * factorC', eye(obj.p)) * Np;
      Hstarmult = kron(factorC', factorC');
      
      Wtilde = [Cmult(lowerDiagCols,:); Hstarmult(diagCols,:)];
      Gtilde = GMulti.H * pinv(Wtilde);
      
      GC = zeros(size(GMulti.H));
      GC(:, lowerDiagCols) = Gtilde(:,1:length(lowerDiagCols));
      GHstar = zeros(size(GMulti.H));
      GHstar(:, diagCols) = Gtilde(:,length(lowerDiagCols)+1:end);
      
      Guni.H = GHstar;

      % Check
      error = GC * Cmult + GHstar * Hstarmult - GMulti.H;
      assert(max(max(abs(error))) < 1e-10, 'Development error');
      
      factorCinv = inv(factorC);
      kronCinv = kron(factorCinv, factorCinv);
      Guni.Z = -GC * kronCinv * kron(ssMulti.Z, eye(obj.p)) + GMulti.Z * kron(eye(obj.m), factorCinv);
      Guni.d = -GC * kronCinv * kron(ssMulti.Z, eye(obj.p)) + GMulti.d * factorCinv;
    end
    
    function [obsErr, stateErr] = getErrors(obj, y, state, a0)
      % Get the errors epsilon & eta given an estimate of the state 
      % Either the filtered or smoothed estimates can be calculated by passing
      % either the filtered state (a) or smoothed state (alphaHat).
      
      % With the state already estimated, we simply have to back out the errors.
      
      % Make sure the object is set up correctly but DO NOT factor as is done 
      % for the univariate filter. 
      if isempty(obj.n)
        obj.n = size(state, 2);
        obj = obj.setInvariantTau();
      else
        assert(obj.n == size(state, 2), ...
          'Size of state doesn''t match time dimension of StateSpace.');
      end
      
      % Iterate through observations
      obsErr = nan(obj.p, obj.n);
      for iT = 1:obj.n
        obsErr(:,iT) = y(:,iT) - ...
          obj.Z(:,:,obj.tau.Z(iT)) * state(:,iT) - obj.d(:,obj.tau.d(iT));        
      end
      
      % Potentially return early
      if nargout < 2
        stateErr = [];
        return
      end
      
      % Iterate through states
      Rbar = nan(obj.g, obj.m, size(obj.R, 3));
      for iR = 1:size(obj.R, 3)
        Rbar(:,:,iR) = (obj.R(:,:,iR)' * obj.R(:,:,iR)) \ obj.R(:,:,iR)';
      end
      
      stateErr = nan(obj.g, obj.n);
      stateErr(:,1) = Rbar(:,:,obj.tau.R(iT)) * (state(:,1) - ...
        obj.T(:,:,obj.tau.T(1)) * a0 - obj.c(:,obj.tau.c(1)));
      for iT = 2:obj.n
        stateErr(:,iT) = Rbar(:,:,obj.tau.R(iT)) * (state(:,iT) - ...
          obj.T(:,:,obj.tau.T(iT)) * state(:,iT-1) - obj.c(:,obj.tau.c(iT)));
      end
    end
    
    function [V, J, D] = getErrorVariances(obj, fOutNew, sOut, factorC)
      % Get the smoothed state variance and covariance matricies
      % Produces V = Var(alpha | Y_n) and J = Cov(alpha_{t+1}, alpha_t | Y_n)
      % and D = Var(epsilon_t | Y_n)
      
      I = eye(obj.m);
      Hinv = nan(obj.p, obj.p, size(obj.H, 3));
      for iH = 1:size(obj.H, 3)
        Hinv(:,:,iH) = AbstractSystem.pseudoinv(obj.H(:,:,iH), 1e-12);
      end
            
      V = nan(obj.m, obj.m, obj.n);
      J = nan(obj.m, obj.m, obj.n);
      D = nan(obj.p, obj.p, obj.n);
      for iT = obj.n:-1:1
        iP = fOutNew.P(:,:,iT);
        iC = factorC(:,:,obj.tau.H(iT));
        
        V(:,:,iT) = iP - iP * sOut.N(:,:,iT) * iP;
        
        L = obj.T(:,:,obj.tau.T(iT)) - fOutNew.K(:,:,iT) * obj.Z(:,:,obj.tau.Z(iT));
        J(:,:,iT) = iP * L' * (I - sOut.N(:,:,iT+1) * fOutNew.P(:,:,iT+1));
        
        % TODO: Can we do this without the F inverses or Ks?
        D(:,:,iT) = iC' * (fOutNew.Finv(:,:,iT) + ...
          fOutNew.K(:,:,iT)' * sOut.N(:,:,iT) * fOutNew.K(:,:,iT)) * iC;
      end
    end
        
    %% Filter/smoother/gradient mathematical methods
    function [a, logli, filterOut] = filter_m(obj, y)
      % Filter using exact initial conditions
      %
      % Note that the quantities v, F and K are those that come from the
      % univariate filter and cannot be transfomed via the Cholesky
      % factorization to get the quanties from the multivariate filter. 
      %
      % See "Fast Filtering and Smoothing for Multivariate State Space Models",
      % Koopman & Durbin (2000) and Durbin & Koopman, sec. 7.2.5.
              
      assert(all(arrayfun(@(iH) isdiag(obj.H(:,:,iH)), 1:size(obj.H, 3))), 'Univarite only!');
      
      % Preallocate
      % Note Pd is the "diffuse" P matrix (P_\infty).
      a = zeros(obj.m, obj.n+1);
      v = zeros(obj.p, obj.n);
      
      Pd = zeros(obj.m, obj.m, obj.n+1);
      Pstar = zeros(obj.m, obj.m, obj.n+1);
      Fd = zeros(obj.p, obj.n);
      Fstar = zeros(obj.p, obj.n);
      
      Kd = zeros(obj.m, obj.p, obj.n);
      Kstar = zeros(obj.m, obj.p, obj.n);
      
      LogL = zeros(obj.p, obj.n);
      
      % Initialize - Using the FRBC timing 
      ii = 0;
      Tii = obj.T(:,:,obj.tau.T(ii+1));
      a(:,ii+1) = Tii * obj.a0 + obj.c(:,obj.tau.c(ii+1));
      
      Pd0 = obj.A0 * obj.A0';
      Pstar0 = obj.R0 * obj.Q0 * obj.R0';
      Pd(:,:,ii+1)  = Tii * Pd0 * Tii';
      Pstar(:,:,ii+1) = Tii * Pstar0 * Tii' + ...
        obj.R(:,:,obj.tau.R(ii+1)) * obj.Q(:,:,obj.tau.Q(ii+1)) * obj.R(:,:,obj.tau.R(ii+1))';

      % Initial recursion
      while ~all(all(Pd(:,:,ii+1) == 0))
        if ii >= obj.n
          error(['Degenerate model. ' ...
          'Exact initial filter unable to transition to standard filter.']);
        end
        
        ii = ii + 1;
        ind = find(~isnan(y(:,ii)));
        
        ati = a(:,ii);
        Pstarti = Pstar(:,:,ii);
        Pdti = Pd(:,:,ii);
        for jj = ind'
          Zjj = obj.Z(jj,:,obj.tau.Z(ii));
          v(jj,ii) = y(jj, ii) - Zjj * ati - obj.d(jj,obj.tau.d(ii));
          
          Fd(jj,ii) = Zjj * Pdti * Zjj';
          Fstar(jj,ii) = Zjj * Pstarti * Zjj' + obj.H(jj,jj,obj.tau.H(ii));
          
          Kd(:,jj,ii) = Pdti * Zjj';
          Kstar(:,jj,ii) = Pstarti * Zjj';
          
          if Fd(jj,ii) ~= 0
            % F diffuse nonsingular
            ati = ati + Kd(:,jj,ii) ./ Fd(jj,ii) * v(jj,ii);
            
            Pstarti = Pstarti + Kd(:,jj,ii) * Kd(:,jj,ii)' * Fstar(jj,ii) * (Fd(jj,ii).^-2) - ...
              (Kstar(:,jj,ii) * Kd(:,jj,ii)' + Kd(:,jj,ii) * Kstar(:,jj,ii)') ./ Fd(jj,ii);
            
            Pdti = Pdti - Kd(:,jj,ii) .* Kd(:,jj,ii)' ./ Fd(jj,ii);
            
            LogL(jj,ii) = log(Fd(jj,ii));
          else
            % F diffuse = 0
            ati = ati + Kstar(:,jj,ii) ./ Fstar(jj,ii) * v(jj,ii);
            
            Pstarti = Pstarti - Kstar(:,jj,ii) ./ Fstar(jj,ii) * Kstar(:,jj,ii)';
            
            % Pdti = Pdti;
              
            LogL(jj,ii) = (log(Fstar(jj,ii)) + (v(jj,ii)^2) ./ Fstar(jj,ii));
          end
        end
        
        Tii = obj.T(:,:,obj.tau.T(ii+1));
        a(:,ii+1) = Tii * ati + obj.c(:,obj.tau.c(ii+1));
        
        Pd(:,:,ii+1)  = Tii * Pdti * Tii';
        Pstar(:,:,ii+1) = Tii * Pstarti * Tii' + ...
          obj.R(:,:,obj.tau.R(ii+1)) * obj.Q(:,:,obj.tau.Q(ii+1)) * obj.R(:,:,obj.tau.R(ii+1))';
      end
      
      dt = ii;
      
      F = Fstar;
      K = Kstar;
      P = Pstar;
      
      % Standard Kalman filter recursion
      for ii = dt+1:obj.n
        ind = find(~isnan(y(:,ii)));
        ati    = a(:,ii);
        Pti    = P(:,:,ii);
        for jj = ind'
          Zjj = obj.Z(jj,:,obj.tau.Z(ii));
          
          v(jj,ii) = y(jj,ii) - Zjj * ati - obj.d(jj,obj.tau.d(ii));
          
          F(jj,ii) = Zjj * Pti * Zjj' + obj.H(jj,jj,obj.tau.H(ii));
          K(:,jj,ii) = Pti * Zjj';
          
          LogL(jj,ii) = (log(F(jj,ii)) + (v(jj,ii)^2) / F(jj,ii));
          
          ati = ati + K(:,jj,ii) / F(jj,ii) * v(jj,ii);
          Pti = Pti - K(:,jj,ii) / F(jj,ii) * K(:,jj,ii)';
        end
        
        Tii = obj.T(:,:,obj.tau.T(ii+1));
        
        a(:,ii+1) = Tii * ati + obj.c(:,obj.tau.c(ii+1));
        P(:,:,ii+1) = Tii * Pti * Tii' + ...
          obj.R(:,:,obj.tau.R(ii+1)) * obj.Q(:,:,obj.tau.Q(ii+1)) * obj.R(:,:,obj.tau.R(ii+1))';
      end
      
      % Consider changing to 
      %   sum(sum(F(:,d+1:end)~=0)) + sum(sum(Fstar(:,1:d)~=0))
      logli = -(0.5 * sum(sum(isfinite(y)))) * log(2 * pi) - 0.5 * sum(sum(LogL));
      
      filterOut = obj.compileStruct(a, P, Pd, v, F, Fd, K, Kd, dt);
    end
    
    function [a, logli, filterOut, detailOut] = filter_detail_m(obj, y)
      % Filter using exact initial conditions
      %
      % Note that the quantities v, F and K are those that come from the
      % univariate filter and cannot be transfomed via the Cholesky
      % factorization to get the quanties from the multivariate filter. 
      %
      % See "Fast Filtering and Smoothing for Multivariate State Space Models",
      % Koopman & Durbin (2000) and Durbin & Koopman, sec. 7.2.5.
              
      assert(all(arrayfun(@(iH) isdiag(obj.H(:,:,iH)), 1:size(obj.H, 3))), 'Univarite only!');
      
      % Preallocate
      % Note Pd is the "diffuse" P matrix (P_\infty).
      a = zeros(obj.m, obj.n+1);
      ati = zeros(obj.m, obj.n+1, obj.p+1);
      v = zeros(obj.p, obj.n);
      
      Pd = zeros(obj.m, obj.m, obj.n+1);
      Pdti = zeros(obj.m, obj.m, obj.n+1, obj.p+1);
      Pstar = zeros(obj.m, obj.m, obj.n+1);
      Pstarti = zeros(obj.m, obj.m, obj.n+1, obj.p+1);
      Fd = zeros(obj.p, obj.n);
      Fstar = zeros(obj.p, obj.n);
      
      Kd = zeros(obj.m, obj.p, obj.n);
      Kstar = zeros(obj.m, obj.p, obj.n);
      
      LogL = zeros(obj.p, obj.n);
      
      % Initialize - Using the FRBC timing 
      iT = 0;
      Tii = obj.T(:,:,obj.tau.T(iT+1));
      a(:,iT+1) = Tii * obj.a0 + obj.c(:,obj.tau.c(iT+1));
      
      Pd0 = obj.A0 * obj.A0';
      Pstar0 = obj.R0 * obj.Q0 * obj.R0';
      
      Pd(:,:,iT+1)  = Tii * Pd0 * Tii';
      Pstar(:,:,iT+1) = Tii * Pstar0 * Tii' + ...
        obj.R(:,:,obj.tau.R(iT+1)) * obj.Q(:,:,obj.tau.Q(iT+1)) * obj.R(:,:,obj.tau.R(iT+1))';

      % Initial recursion
      while ~all(all(Pd(:,:,iT+1) == 0))
        if iT >= obj.n
          error(['Degenerate model. ' ...
          'Exact initial filter unable to transition to standard filter.']);
        end
        
        iT = iT + 1;
       
        ati(:,iT,1) = a(:,iT);
        Pstarti(:,:,iT,1) = Pstar(:,:,iT);
        Pdti(:,:,iT,1) = Pd(:,:,iT);
        for iP = 1:obj.p 
          if isnan(y(:,iT))
            ati(:,iT,iP+1) = ati(:,iT,iP);
            Pdti(:,:,iT,iP+1) = Pdti(:,:,iT,iP);
            Pstarti(:,:,iT,iP+1) = Pstarti(:,:,iT,iP);
            continue
          end
          
          Zjj = obj.Z(iP,:,obj.tau.Z(iT));
          v(iP,iT) = y(iP, iT) - Zjj * ati(:,iT,iP) - obj.d(iP,obj.tau.d(iT));
          
          Fd(iP,iT) = Zjj * Pdti(:,:,iT,iP) * Zjj';
          Fstar(iP,iT) = Zjj * Pstarti(:,:,iT,iP) * Zjj' + obj.H(iP,iP,obj.tau.H(iT));
          
          Kd(:,iP,iT) = Pdti(:,:,iT,iP) * Zjj';
          Kstar(:,iP,iT) = Pstarti(:,:,iT,iP) * Zjj';
          
          if Fd(iP,iT) ~= 0
            % F diffuse nonsingular
            ati(:,iT,iP+1) = ati(:,iT,iP) + ...
              Kd(:,iP,iT) ./ Fd(iP,iT) * v(iP,iT);
            
            Pstarti(:,:,iT,iP+1) = Pstarti(:,:,iT,iP) + ...
              Kd(:,iP,iT) * Kd(:,iP,iT)' * Fstar(iP,iT) * (Fd(iP,iT).^-2) - ...
              (Kstar(:,iP,iT) * Kd(:,iP,iT)' + Kd(:,iP,iT) * Kstar(:,iP,iT)') ./ Fd(iP,iT);
            
            Pdti(:,:,iT,iP+1) = Pdti(:,:,iT,iP) - ...
              Kd(:,iP,iT) .* Kd(:,iP,iT)' ./ Fd(iP,iT);
            
            LogL(iP,iT) = log(Fd(iP,iT));
          else
            % F diffuse = 0
            ati(:,iT,iP+1) = ati(:,iT,iP) + ...
              Kstar(:,iP,iT) ./ Fstar(iP,iT) * v(iP,iT);
            
            Pstarti(:,:,iT,iP+1) = Pstarti(:,:,iT,iP) - ...
              Kstar(:,iP,iT) ./ Fstar(iP,iT) * Kstar(:,iP,iT)';
            
            Pdti(:,:,iT,iP+1) = Pdti(:,:,iT,iP);
              
            LogL(iP,iT) = (log(Fstar(iP,iT)) + (v(iP,iT)^2) ./ Fstar(iP,iT));
          end
        end
        
        Tii = obj.T(:,:,obj.tau.T(iT+1));
        a(:,iT+1) = Tii * ati(:,iT,obj.p+1) + obj.c(:,obj.tau.c(iT+1));
        
        Pd(:,:,iT+1)  = Tii * Pdti(:,:,iT,obj.p+1) * Tii';
        Pstar(:,:,iT+1) = Tii * Pstarti(:,:,iT,obj.p+1) * Tii' + ...
          obj.R(:,:,obj.tau.R(iT+1)) * obj.Q(:,:,obj.tau.Q(iT+1)) * obj.R(:,:,obj.tau.R(iT+1))';
      end
      
      dt = iT;
      
      F = Fstar;
      K = Kstar;
      P = Pstar;
      Pti = Pstarti;
      
      % Standard Kalman filter recursion
      for iT = dt+1:obj.n
        ind = find(~isnan(y(:,iT)));
        ati(:,iT,1) = a(:,iT);
        Pti(:,:,iT,1) = P(:,:,iT);
        for iP = ind'
          Zjj = obj.Z(iP,:,obj.tau.Z(iT));
          
          v(iP,iT) = y(iP,iT) - Zjj * ati(:,iT,iP) - obj.d(iP,obj.tau.d(iT));
          
          F(iP,iT) = Zjj * Pti(:,:,iT,iP) * Zjj' + obj.H(iP,iP,obj.tau.H(iT));
          K(:,iP,iT) = Pti(:,:,iT,iP) * Zjj';
          
          LogL(iP,iT) = (log(F(iP,iT)) + (v(iP,iT)^2) / F(iP,iT));
          
          ati(:,iT,iP+1) = ati(:,iT,iP) + K(:,iP,iT) / F(iP,iT) * v(iP,iT);
          Pti(:,:,iT,iP+1) = Pti(:,:,iT,iP) - K(:,iP,iT) / F(iP,iT) * K(:,iP,iT)';
        end
        
        Tii = obj.T(:,:,obj.tau.T(iT+1));
        
        a(:,iT+1) = Tii * ati(:,iT,obj.p+1) + obj.c(:,obj.tau.c(iT+1));
        P(:,:,iT+1) = Tii * Pti(:,:,iT,obj.p+1) * Tii' + ...
          obj.R(:,:,obj.tau.R(iT+1)) * obj.Q(:,:,obj.tau.Q(iT+1)) * obj.R(:,:,obj.tau.R(iT+1))';
      end
      
      ati(:,obj.n+1,:) = repmat(a(:,obj.n+1), [1 1 obj.p+1]);
      Pti(:,:,obj.n+1,:) = repmat(P(:,:,obj.n+1), [1 1 1 obj.p+1]);
          
      % Consider changing to 
      %   sum(sum(F(:,d+1:end)~=0)) + sum(sum(Fstar(:,1:d)~=0))
      logli = -(0.5 * sum(sum(isfinite(y)))) * log(2 * pi) - 0.5 * sum(sum(LogL));
      
      filterOut = obj.compileStruct(a, P, Pd, v, F, Fd, K, Kd, dt);
      detailOut = obj.compileStruct(ati, Pti, Pdti);
    end
    
    function [a, logli, filterOut] = filter_mex(obj, y)
      % Call mex function filter_uni
      ssStruct = struct('Z', obj.Z, 'd', obj.d, 'H', obj.H, ...
        'T', obj.T, 'c', obj.c, 'R', obj.R, 'Q', obj.Q, ...
        'a0', obj.a0, 'A0', obj.A0, 'R0', obj.R0, 'Q0', obj.Q0, ...
        'tau', obj.tau);
      if isempty(ssStruct.R0)
        ssStruct.R0 = zeros(obj.m, 1);
        ssStruct.Q0 = 0;
      end
      if isempty(ssStruct.A0)
        ssStruct.A0 = zeros(obj.m, 1);
      end
      
      [a, logli, P, Pd, v, F, Fd, K, Kd, dt] = mfss_mex.filter_uni(y, ssStruct);
      filterOut = obj.compileStruct(a, P, Pd, v, F, Fd, K, Kd, dt);
    end
    
    function [alpha, smootherOut] = smoother_m(obj, y, fOut)
      % Univariate smoother
      
      alpha = zeros(obj.m, obj.n);
      V     = zeros(obj.m, obj.m, obj.n);
      eta   = zeros(obj.g, obj.n);
      r     = zeros(obj.m, obj.n);
      N     = zeros(obj.m, obj.m, obj.n);
      
      rti = zeros(obj.m,1);
      Nti = zeros(obj.m,obj.m);
      for ii = obj.n:-1:fOut.dt+1
        ind = flipud(find(~isnan(y(:,ii))));
        
        for jj = ind'
          Lti = eye(obj.m) - fOut.K(:,jj,ii) * ...
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
        V(:,:,ii) = fOut.P(:,:,ii) - fOut.P(:,:,ii) * N(:,:,ii) * fOut.P(:,:,ii);
        eta(:,ii) = obj.Q(:,:,obj.tau.Q(ii+1)) * obj.R(:,:,obj.tau.R(ii+1))' * r(:,ii); 
        
        rti = obj.T(:,:,obj.tau.T(ii))' * rti;
        Nti = obj.T(:,:,obj.tau.T(ii))' * Nti * obj.T(:,:,obj.tau.T(ii));
      end
      
      r1 = zeros(obj.m, fOut.dt+1);
      N1 = zeros(obj.m, obj.m, fOut.dt+1);
      N2 = zeros(obj.m, obj.m, fOut.dt+1);

      % Note: r0 = r and N0 = N
      r0ti = r(:,fOut.dt+1);
      r1ti = r1(:,fOut.dt+1);
      N0ti = N(:,:,fOut.dt+1);
      N1ti = N1(:,:,fOut.dt+1);
      N2ti = N2(:,:,fOut.dt+1);
      
      % Exact initial smoother
      for ii = fOut.dt:-1:1
        ind = flipud(find(~isnan(y(:,ii))));
        for jj = ind'
          Zjj = obj.Z(jj,:,obj.tau.Z(ii));
          
          if fOut.Fd(jj,ii) ~= 0
            % Diffuse case
            Ldti = eye(obj.m) - fOut.Kd(:,jj,ii) * Zjj / fOut.Fd(jj,ii);
            % NOTE: minus sign!
            L0ti = (fOut.Kd(:,jj,ii) * fOut.F(jj,ii) / fOut.Fd(jj,ii) - ... 
              fOut.K(:,jj,ii)) * Zjj / fOut.Fd(jj,ii);
            
            r0ti = Ldti' * r0ti;
            % NOTE: plus sign!
            r1ti = Zjj' / fOut.Fd(jj,ii) * fOut.v(jj,ii) + L0ti' * r0ti + Ldti' * r1ti; 
            
            N0ti = Ldti' * N0ti * Ldti;
            N1ti = Zjj' / fOut.Fd(jj,ii) * Zjj + Ldti' * N0ti * L0ti + Ldti' * N1ti * Ldti;
            N2ti = Zjj' * fOut.Fd(jj,ii)^(-2) * Zjj * fOut.F(jj,ii) + ...
              L0ti' * N1ti * L0ti + Ldti' * N1ti * L0ti + ...
              L0ti' * N1ti * Ldti + Ldti' * N2ti * Ldti;
          else
            % Known
            Lstarti = eye(obj.m) - fOut.K(:,jj,ii) * Zjj / fOut.F(jj,ii);
            r0ti = Zjj' / fOut.F(jj,ii) * fOut.v(jj,ii) + Lstarti' * r0ti;
            
            N0ti = Zjj' / fOut.F(jj,ii) * Zjj + Lstarti' * N0ti * Lstarti;
          end
        end
        
        r(:,ii) = r0ti;
        r1(:,ii) = r1ti;
        N(:,:,ii) = N0ti;
        N1(:,:,ii) = N1ti;
        N2(:,:,ii) = N2ti;        
        
        % What here needs tau_{ii+1}?
        alpha(:,ii) = fOut.a(:,ii) + fOut.P(:,:,ii) * r(:,ii) + ...
          fOut.Pd(:,:,ii) * r1(:,ii);
        V(:,:,ii) = fOut.P(:,:,ii) - ...
          fOut.P(:,:,ii) * N(:,:,ii) * fOut.P(:,:,ii) - ...
          (fOut.Pd(:,:,ii) * N1(:,:,ii) * fOut.P(:,:,ii))' - ...
          fOut.P(:,:,ii) * N1(:,:,ii) * fOut.Pd(:,:,ii) - ...
          fOut.Pd(:,:,ii) * N2(:,:,ii) * fOut.Pd(:,:,ii);

        eta(:,ii) = obj.Q(:,:,obj.tau.Q(ii)) * obj.R(:,:,obj.tau.R(ii))' * r(:,ii);
        
        r0ti = obj.T(:,:,obj.tau.T(ii))' * r0ti;
        r1ti = obj.T(:,:,obj.tau.T(ii))' * r1ti;
        
        N0ti = obj.T(:,:,obj.tau.T(ii))' * N0ti * obj.T(:,:,obj.tau.T(ii));
        N1ti = obj.T(:,:,obj.tau.T(ii))' * N1ti * obj.T(:,:,obj.tau.T(ii));
        N2ti = obj.T(:,:,obj.tau.T(ii))' * N2ti * obj.T(:,:,obj.tau.T(ii));
      end
      
      Pstar0 = obj.R0 * obj.Q0 * obj.R0';
      if fOut.dt > 0
        Pd0 = obj.A0 * obj.A0';
        a0tilde = obj.a0 + Pstar0 * r0ti + Pd0 * r1ti;
      else
        a0tilde = obj.a0 + Pstar0 * rti;
      end
      
      smootherOut = obj.compileStruct(alpha, V, eta, r, N, N1, N2, a0tilde);
    end
    
    function [alpha, smootherOut] = smoother_mex(obj, y, fOut)
      ssStruct = struct('Z', obj.Z, 'd', obj.d, 'H', obj.H, ...
        'T', obj.T, 'c', obj.c, 'R', obj.R, 'Q', obj.Q, ...
        'a0', obj.a0, 'A0', obj.A0, 'R0', obj.R0, 'Q0', obj.Q0, ...
        'tau', obj.tau);
      if isempty(ssStruct.R0)
        ssStruct.R0 = zeros(obj.m, 1);
        ssStruct.Q0 = 0;
      end
      if isempty(ssStruct.A0)
        ssStruct.A0 = zeros(obj.m, 1);
      end
      
      [alpha, eta, r, N, a0tilde] = mfss_mex.smoother_uni(y, ssStruct, fOut);
      smootherOut = obj.compileStruct(alpha, eta, r, N, a0tilde);
    end
    
    function gradient = gradient_filter_m(obj, G, fOut, ftiOut)
      nTheta = size(G.T, 1);
      
      Nm = (eye(obj.m^2) + obj.genCommutation(obj.m));
      
      Gati = zeros(nTheta, obj.m, obj.n, obj.p+1);
      Gati(:,:,1,1) = G.a1;
      
      GPti = zeros(nTheta, obj.m^2, obj.n, obj.p+1);
      GPti(:,:,1,1) = G.P1;
      
      Gvti = zeros(nTheta, obj.n, obj.p);
      GFti = zeros(nTheta, obj.n, obj.p);
      GKti = zeros(nTheta, obj.m, obj.n, obj.p);
      
      grad = zeros(nTheta, obj.n, obj.p);
      for iT = fOut.dt+1:obj.n
        for iP = 1:obj.p
          if fOut.v(iP,iT) == 0
            continue 
          end
          
          % Fetch commonly used quantities
          Zti = obj.Z(iP,:,obj.tau.Z(iT));
          GZti = G.Z(:, iP:obj.p:(obj.p*obj.m), obj.tau.Z(iT));
          GHind = 1 + (iP-1)*(obj.p+1); 
          GHti = G.H(:, GHind, obj.tau.H(iT)); % iP-th diagonal element of H
          
          % Compute basics
          Gvti(:,iT,iP) = -GZti * ftiOut.ati(:,iT,iP) - ...
            Gati(:,:,iT,iP) * Zti' - G.d(:,iP,obj.tau.d(iT));
          GFti(:,iT,iP) = 2 * GZti * (ftiOut.Pti(:,:,iT,iP) * Zti') + ...
            GPti(:,:,iT,iP) * kron(Zti', Zti') + GHti;
          GKti(:,:,iT,iP) = GPti(:,:,iT,iP) * kron(Zti' * fOut.F(iP,iT)^(-1), eye(obj.m)) + ...
            GZti * ftiOut.Pti(:,:,iT,iP) * fOut.F(iP,iT)^(-1) - ...
            (GFti(:,iT,iP) * fOut.F(iP,iT)^(-2) * Zti * ftiOut.Pti(:,:,iT,iP)); 
          
          % Comptue the period contribution to the gradient
          grad(:,iT,iP) = 0.5 * GFti(:,iT,iP) * ...
            (1./fOut.F(iP,iT) - (fOut.v(iP,iT)^2 ./ fOut.F(iP,iT)^2)) + ...
            Gvti(:,iT,iP) * fOut.v(iP,iT) ./ fOut.F(iP,iT);
          
          % Transition from i to i+1
          Gati(:,:,iT,iP+1) = Gati(:,:,iT,iP) + ...
            GKti(:,:,iT,iP) * fOut.v(iP,iT) + ...
            Gvti(:,iT,iP) * fOut.K(:,iP,iT)';
          GKFK = GKti(:,:,iT,iP) * kron(fOut.F(iP,iT) * fOut.K(:,iP,iT)', eye(obj.m)) * Nm + ...
            GFti(:,iT,iP) * kron(fOut.K(:,iP,iT)', fOut.K(:,iP,iT)');
          GPti(:,:,iT,iP+1) = GPti(:,:,iT,iP) - GKFK;
        end
        
        % Transition from time t to t+1
        Tt = obj.T(:,:,obj.tau.T(iT+1));
        Rt = obj.R(:,:,obj.tau.R(iT+1));
        Qt = obj.Q(:,:,obj.tau.Q(iT+1));
        
        Gati(:,:,iT+1,1) = G.T(:,:,obj.tau.T(iT+1)) * kron(ftiOut.ati(:,iT,obj.p+1), eye(obj.m)) + ...
          Gati(:,:,iT,obj.p+1) * Tt' + G.c(:,:,obj.tau.c(iT+1));
        GTPT = G.T(:,:,obj.tau.T(iT+1)) * ...
          kron(ftiOut.Pti(:,:,iT,obj.p+1) * Tt', eye(obj.m)) * Nm + ...
          GPti(:,:,iT,obj.p+1) * kron(Tt', Tt');
        GRQR = G.R(:,:,obj.tau.R(iT+1)) * kron(Qt * Rt', eye(obj.m)) * Nm + ...
          G.Q(:,:,obj.tau.Q(iT+1)) * kron(Rt', Rt');
        GPti(:,:,iT+1,1) = GTPT + GRQR;
      end
      
      gradient = -sum(sum(grad, 3), 2);
    end
    
    %% Decomposition mathematical methods
    function [alphaTweights, alphaTconstant] = smoother_weights(obj, y, fOut, sOut, decompPeriods)
      % Generate weights (effectively via multivariate smoother) 
      %
      % Weights ordered (state, observation, data period, state effect period)
      % Constant contributions are ordered (state, effect period)
      
      % pre-allocate W(j,t) where a(:,t) = \sum_{j=1}^{t-1} W(j,t)*y(:,j)
      alphaTweights = zeros(obj.m, obj.p, obj.n, length(decompPeriods)); 
      alphaTconstant = zeros(obj.m, length(decompPeriods)); 
      
      eyeP = eye(obj.p);
      genF = @(ind, iT) obj.Z(ind,:,obj.tau.Z(iT)) * fOut.P(:,:,iT) * obj.Z(ind,:,obj.tau.Z(iT))' ...
        + obj.H(ind,ind,obj.tau.H(iT));
      genK = @(indJJ, jT) obj.T(:,:,obj.tau.T(jT+1)) * fOut.P(:,:,jT) * ...
            obj.Z(indJJ,:,obj.tau.Z(jT))' * AbstractSystem.pseudoinv(genF(indJJ,jT), 1e-12);
      genL = @(ind, iT) obj.T(:,:,obj.tau.T(iT+1)) - ...
        genK(ind,iT) * (eyeP(ind,:) * obj.Z(:,:,obj.tau.Z(iT)));
      
      wb = waitbar(0, 'Creating smoother weights.');
      
      for iPer = 1:length(decompPeriods)
        iT = decompPeriods(iPer);
        ind = ~isnan(y(:,iT));
        
        iL = genL(ind, iT);
        lWeights = iL;
        % Loop through t,t+1,...,n to calculate weights and constant adjustment for j >= t
        for jT = iT:obj.n 
          indJJ = ~isnan(y(:,jT));
          Zjj = obj.Z(indJJ,:,obj.tau.Z(jT));
          Kjj = genK(indJJ, jT); % fOut.K(:,indJJ,jT);
          jL = genL(indJJ,jT);
          
          Finvjj = AbstractSystem.pseudoinv(genF(indJJ,jT), 1e-12);
          % Slight alternative calculation for the boundary condition for j == t
          if jT == iT
            PFZKNL = fOut.P(:,:,jT) * (Finvjj * Zjj - ...
              Kjj' * sOut.N(:,:,jT+1) * jL)';
            
            alphaTweights(:, indJJ, jT, iPer) = PFZKNL;            
            alphaTconstant(:,iPer) = -PFZKNL * ...
               obj.d(indJJ, obj.tau.d(jT)) + ...
              (eye(obj.m) - fOut.P(:,:,jT) * sOut.N(:,:,jT)) * obj.c(:,obj.tau.c(jT));
          else
            % weight and constant adjustment calculations for j >> t
            PL = fOut.P(:,:,iT) * lWeights;
            PLFZKNL = PL * (Finvjj * Zjj - ...
              Kjj' * sOut.N(:,:,jT+1) * jL)';
            
            alphaTweights(:,indJJ,jT,iPer) = PLFZKNL;
            alphaTconstant(:,iPer) = alphaTconstant(:,iPer) - ...
              PLFZKNL * obj.d(indJJ, obj.tau.d(jT)) - ...
              PL * sOut.N(:,:,jT) * obj.c(:,obj.tau.c(jT));
            lWeights = lWeights * jL';
          end
        end
        
        if iT > 1
          lWeights = (eye(obj.m) - fOut.P(:,:,iT) * sOut.N(:,:,iT));
        end
        % Loop through j = t-1,t-2,...,1 to calculate weights and constant adjustment for j < t
        for jT = iT-1:-1:1
          % find non-missing observations for time period "jj"
          indJJ = ~isnan(y(:,jT));
          jL = genL(ind,jT);

          if iT > jT
            lK =  lWeights * genK(indJJ,jT);
            alphaTweights(:,indJJ,jT,iPer) = lK;
            alphaTconstant(:,iPer) = alphaTconstant(:,iPer) + ...
              lWeights * jL * obj.c(:,obj.tau.c(jT)) - ...
              lK * obj.d(indJJ,obj.tau.d(jT));
            lWeights = lWeights * jL;
          end
        end
        
        waitbar(iPer ./ length(decompPeriods), wb);
      end

      delete(wb);
    end
    
    %% General utilities
    function [stationary, nonstationary] = findStationaryStates(obj)
      % Find which states have stationary distributions given the T matrix.
      [V, D] = eig(obj.T(:,:,obj.tau.T(1)));
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
    
    function [Z, d, H, T, c, R, Q] = getInputParameters(obj)
      % Get parameters to input to constructor
      
      if ~isempty(obj.tau)
        Z = struct('Tt', obj.Z, 'tauT', obj.tau.Z);
        d = struct('ct', obj.d, 'tauc', obj.tau.d);
        H = struct('Rt', obj.H, 'tauR', obj.tau.H);
        
        T = struct('Tt', obj.T, 'tauT', obj.tau.T);
        c = struct('ct', obj.c, 'tauc', obj.tau.c);
        R = struct('Rt', obj.R, 'tauR', obj.tau.R);
        Q = struct('Qt', obj.Q, 'tauQ', obj.tau.Q);
      else
        Z = obj.Z;
        d = obj.d;
        H = obj.H;
        
        T = obj.T;
        c = obj.c;
        R = obj.R;
        Q = obj.Q;
      end
    end
  end
  
  methods (Static, Hidden)
    %% General Utility Methods
    function setSS = setAllParameters(ss, value)
      % Create a StateSpace with all paramters set to value
      
      % Set all parameter values equal to the scalar provided
      for iP = 1:length(ss.systemParam)
        ss.(ss.systemParam{iP})(:) = value;
      end
      
      % Needs to be a StateSpace since we don't want a ThetaMap
      setSS = StateSpace(ss.Z(:,:,1), ss.d(:,1), ss.H(:,:,1), ...
        ss.T(:,:,1), ss.c(:,1), ss.R(:,:,1), ss.Q(:,:,1));
      paramNames = ss.systemParam;
      for iP = 1:length(paramNames)
        setSS.(paramNames{iP}) = ss.(paramNames{iP});
      end
      setSS.tau = ss.tau;
      setSS.timeInvariant = ss.timeInvariant;
      setSS.n = ss.n;
      
      % Do I want to set initial values? 
      if ~isempty(ss.a0)
        a0value = repmat(value, [ss.m, 1]);
      else
        a0value = [];
      end
      setSS = setSS.setInitial(a0value);
      
      % Do I want to account for diffuse states? 
      if ~isempty(ss.Q0)
        setSS.A0 = ss.A0;
        setSS.R0 = ss.R0;
        
        Q0value = repmat(value, size(ss.Q0));  
        setSS = setSS.setQ0(Q0value);
      end
    end    
  end
end
