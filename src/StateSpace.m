classdef StateSpace < AbstractStateSpace
  % State estimation of models with known parameters
  %
  % See documentation in the function reference.
  
  % David Kelley, 2016-2018
  
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
      assert(isvector(newa0) || isempty(newa0));
      obj.a0Private = newa0;
    end
    
    function P0 = get.P0(obj)
      P0 = obj.P0Private;
    end
    
    function obj = set.P0(obj, newP0)
      assert(ismatrix(newP0)|| isempty(newP0));
      obj.P0Private = newP0;
    end
  end
  
  methods
    function obj = StateSpace(Z, H, T, Q, varargin)
      % StateSpace constructor
      %
      % Arguments:
      %   Z (matrix): Observation loadings
      %   H (matrix): Observation error covariances
      %   T (matrix): State transition coefficients
      %   Q (matrix): State error covariances
      % Optional Arguments: 
      %   d (matrix): Observation constants
      %   beta (matrix): Exogenous measurement series loadings
      %   c (matrix): State constants
      %   gamma (matrix): Exogenous state series loadings
      %   R (matrix): Error selection
      %   a0 (vector): initial estimate of the state
      %   P0 (matrix): variance of initial estimate of the state
      %
      % Returns:
      %   obj (StateSpace): a StateSpace object
      
      inP = inputParser();
      inP.addParameter('d', []); 
      inP.addParameter('beta', []);
      inP.addParameter('c', []);
      inP.addParameter('gamma', []);
      inP.addParameter('R', []);
      inP.addParameter('a0', []);
      inP.addParameter('P0', []);
      inP.parse(varargin{:});
      parsed = inP.Results;
      
      if nargin == 0
        superArgs = {};
      else
        superArgs = {Z, parsed.d, parsed.beta, H, T, parsed.c, parsed.gamma, parsed.R, Q};
      end
      obj = obj@AbstractStateSpace(superArgs{:});
      if nargin == 0
        return;
      end
      if ~isempty(parsed.a0)
        obj.a0 = parsed.a0;
      end
      if ~isempty(parsed.P0)
        obj.P0 = parsed.P0;
      end
      obj.validateStateSpace();
    end
  end
  
  methods
    function [a, logli, filterOut] = filter(obj, y, x, w)
      % Estimate the filtered state
      %
      % Arguments:
      %   y (double): observed data (p x T)
      %   x (double): exogenous measreument data (k x T)
      %   w (double): exogenous state data (l x T)
      %
      % Returns:
      %   a (double) : filtered state (m x [T+1])
      %   logli (double) : log-likelihood
      %   filterOut (struct) : structure of additional intermeidate quantites
      
      if nargin < 3
        x = [];
      end
      if nargin < 4
        w = [];
      end
      
      [obj, yCheck, xCheck, wCheck] = obj.prepareFilter(y, x, w);
      
      assert(~any(obj.H(1:obj.p+1:end) < 0), 'Negative error variance.');
      
      % Call the filter
      if obj.useMex
        [a, logli, filterOut] = obj.filter_mex(yCheck, xCheck, wCheck);
      else
        [a, logli, filterOut] = obj.filter_m(yCheck, xCheck, wCheck);
      end
      
      if ~all(size(y) == size(yCheck))
        % The data were transposed prior to running the filter, make a match shape
        a = a';
      end
    end
    
    function [alpha, smootherOut, filterOut] = smooth(obj, y, x, w)
      % Estimate the smoothed state
      %
      % Arguments:
      %   y (double): observed data (p x T)
      %   x (double): exogenous measreument data (k x T)
      %   w (double): exogenous state data (l x T)
      %
      % Returns:
      %   alpha (double) : smoothed state (m x T)
      %   smootherOut (struct) : structure of additional intermeidate quantites
      %   filterOut (struct) : structure of additional intermeidate quantites
      
      if nargin < 3
        x = [];
      end
      if nargin < 4
        w = [];
      end
      
      [obj, yCheck, xCheck, wCheck] = obj.prepareFilter(y, x, w);
      
      % Get the filtered estimates for use in the smoother
      [~, logli, filterOut] = obj.filter(yCheck, xCheck, wCheck);
      
      % Determine which version of the smoother to run
      if obj.useMex
        [alpha, smootherOut] = obj.smoother_mex(yCheck, filterOut);
      else
        [alpha, smootherOut] = obj.smoother_m(yCheck, filterOut);
      end
      
      if ~all(size(y) == size(yCheck))
        % The data were transposed prior to running the smoother, make alpha match shape
        alpha = alpha';
      end
      smootherOut.logli = logli;
    end
    
    function [logli, gradient, filterOut] = gradient(obj, y, x, w, tm, theta)
      % Evaluate the likelihood and gradient of a set of parameters
      %
      % Arguments:
      %   y (double): observed data (p x T)
      %   x (double): exogenous measreument data (k x T)
      %   w (double): exogenous state data (l x T)
      %   tm (ThetaMap): ThetaMap that builds StateSpace system from parameter vector
      %   theta (double): parameters to evaluate likelihood and gradient (nTheta x 1)
      %
      % Returns:
      %   logli (double) : log-likelihood
      %   gradient (double) : vector of change in likelihood given change in theta
      %   filterOut (struct) : structure of additional intermeidate quantites
      
      % Handle inputs
      assert(isa(tm, 'ThetaMap'), 'tm must be a ThetaMap.');
      if nargin < 6 || isempty(theta)
        theta = tm.system2theta(obj);
      end
      assert(all(size(theta) == [tm.nTheta 1]), ...
        'theta must be a nTheta x 1 vector.');
      
      if obj.useParallel
        [logli, gradient, filterOut] = obj.gradFinDiff_parallel(y, x, w, tm, theta);
      else
        [logli, gradient, filterOut] = obj.gradFinDiff(y, x, w, tm, theta);
      end
    end
    
    function [dataContr, paramContr, exogMContr, exogSContr, weights] = ...
        decompose_filtered(obj, y, x, w)
      % Decompose the filtered states by contribution effect at each time. 
      %
      % Arguments:
      %   y (double): observed data (p x T)
      %   x (double): exogenous measreument data (k x T)
      %   w (double): exogenous state data (l x T)
      %
      % Returns:
      %   dataContr (double) : contrib. from the data (m x p x T+1)
      %   paramContr (double) : contrib. from the parameters (m x T+1)
      %   exogMContr (double) : contrib. from the exogenous measurement data (m x k x T+1)
      %   exogSContr (double) : contrib. from the exogenous state data (m x l x T+1)
      %   weights (struct) : contrib. separated by weight function. Cell arrays for each
      %     weight ordered by contributing time then effect time. The first dimension 
      %     of each cell element is ordered by state effected then quantity contributnig.
      
      if nargin < 3
        x = [];
      end
      if nargin < 4
        w = [];
      end
      [obj, y, x, w] = obj.checkSample(y, x, w);
      
      % Get quantities from filter
      [~, ~, fOut] = obj.filter(y, x, w);

      % Transform to the univariate form of the state space
      obj.validateKFilter();
      obj = obj.checkSample(y, x, w);
      ssMulti = obj;
      [obj, ~, ~, ~, C] = obj.prepareFilter(y, x, w);
      
      % Compute recursion
      weights = obj.filter_weights(y, x, w, fOut, ssMulti, C);
      
      % Weights are ordered (state, observation, effect, origin) so we need to collapse 
      % the 4th dimension for the data and the 2nd and 4th dimensions for the parameters. 
      dataContr = zeros(obj.m, obj.p, obj.n+1);
      paramContr = zeros(obj.m, obj.n+1);
      exogMContr = zeros(obj.m, obj.k, obj.n+1);
      exogSContr = zeros(obj.m, obj.l, obj.n+1);
      for iT = 1:obj.n+1
        if any(~cellfun(@isempty, weights.y(iT,:)))
          dataContr(:,:,iT) = sum(cat(3, ...
            weights.y{iT, ~cellfun(@isempty, weights.y(iT,:))}), 3);
        end
        
        if any(~cellfun(@isempty, weights.d(iT,:)))
          paramContr(:,iT) = ...
            sum(sum(cat(3, weights.d{iT, ~cellfun(@isempty, weights.d(iT,:))}), 3), 2);
        end
        
        if any(~cellfun(@isempty, weights.c(iT,:)))
          paramContr(:,iT) = paramContr(:,iT) + ...
            sum(sum(cat(3, weights.c{iT, ~cellfun(@isempty, weights.c(iT,:))}), 3), 2);
        end
        
        if ~isempty(weights.a0{iT})
          paramContr(:,iT) = paramContr(:,iT) + sum(weights.a0{iT}, 2);
        end
        
        if any(~cellfun(@isempty, weights.x(iT,:)))
          exogMContr(:,:,iT) = sum(cat(3, ...
            weights.x{iT, ~cellfun(@isempty, weights.x(iT,:))}), 3);
        end
        
        if any(~cellfun(@isempty, weights.w(iT,:)))
          exogSContr(:,:,iT) = sum(cat(3, ...
            weights.w{iT, ~cellfun(@isempty, weights.w(iT,:))}), 3);
        end
      end
    end
    
    function [dataContr, paramContr, exogMContr, exogSContr, weights] = ...
        decompose_smoothed(obj, y, x, w)
      % Decompose the smoothed states by contribution effect at each time. 
      %
      % Arguments:
      %   y (double): observed data (p x T)
      %   x (double): exogenous measreument data (k x T)
      %   w (double): exogenous state data (l x T)
      %
      % Returns:
      %   dataContr (double) : contrib. from the data (m x p x T)
      %   paramContr (double) : contrib. from the parameters (m x T)
      %   exogMContr (double) : contrib. from the exogenous measurement data (m x k x T)
      %   exogSContr (double) : contrib. from the exogenous state data (m x l x T)
      %   weights (struct) : contrib. separated by weight function. Cell arrays for each
      %     weight ordered by contributing time then effect time. The first dimension 
      %     of each cell element is ordered by state effected then quantity contributnig.
      
      if nargin < 3
        x = [];
      end
      if nargin < 4
        w = [];
      end
      [obj, y, x, w] = obj.checkSample(y, x, w);

      % Get output from the filter
      [~, ~, fOut] = obj.filter(y, x, w);

      % Transform to the univariate form of the state space
      obj.validateKFilter();
      obj = obj.checkSample(y, x, w);
      ssMulti = obj;
      [obj, ~, ~, ~, C] = obj.prepareFilter(y, x, w);
      
      weights = obj.smoother_weights(y, x, w, fOut, ssMulti, C);
      
      % Weights come out ordered (state, observation, effect, origin) so we need
      % to collapse the 4th dimension for the data and the 2nd and 4th
      % dimensions for the parameters. 
      dataContr = zeros(obj.m, obj.p, obj.n);
      paramContr = zeros(obj.m, obj.n);
      exogMContr = zeros(obj.m, obj.k, obj.n);
      exogSContr = zeros(obj.m, obj.l, obj.n);
      for iT = 1:obj.n
        if any(~cellfun(@isempty, weights.y(iT,:)))
          dataContr(:,:,iT) = sum(cat(3, ...
            weights.y{iT, ~cellfun(@isempty, weights.y(iT,:))}), 3);
        end
        
        if any(~cellfun(@isempty, weights.d(iT,:)))
          paramContr(:,iT) = ...
            sum(sum(cat(3, weights.d{iT, ~cellfun(@isempty, weights.d(iT,:))}), 3), 2);
        end
        
        if any(~cellfun(@isempty, weights.c(iT,:)))
          paramContr(:,iT) = paramContr(:,iT) + ...
            sum(sum(cat(3, weights.c{iT, ~cellfun(@isempty, weights.c(iT,:))}), 3), 2);
        end
        
        if ~isempty(weights.a0{iT})
          paramContr(:,iT) = paramContr(:,iT) + sum(weights.a0{iT}, 2);
        end
        
        if any(~cellfun(@isempty, weights.x(iT,:)))
          exogMContr(:,:,iT) = sum(cat(3, ...
            weights.x{iT, ~cellfun(@isempty, weights.x(iT,:))}), 3);
        end
                
        if any(~cellfun(@isempty, weights.w(iT,:)))
          exogSContr(:,:,iT) = sum(cat(3, ...
            weights.w{iT, ~cellfun(@isempty, weights.w(iT,:))}), 3);
        end
      end
    end
    
    function [alphaTilde, mleVariance] = stateSample(obj, y, x, w, tm, nSamples)
      % Get samples of the state taking into account parameter uncertainty around MLE
      
      % Handle inputs
      if nargin < 6
        nSamples = 1;
      end
      assert(isa(tm, 'ThetaMap'), 'tm must be a ThetaMap.');
      theta = tm.system2theta(obj);
      assert(all(size(theta) == [tm.nTheta 1]), ...
        'theta must be a nTheta x 1 vector.');

      % Get smoothed state
      alphaHat = obj.smooth(y, x, w);
      fisherInformation = -obj.gradFinDiffSecond(y, x, w, tm, theta);
      mleVariance = inv(fisherInformation * obj.n);
      alphaTilde = nan([size(alphaHat), nSamples]);

      % Check if outer product of gradients is positive semi-definite
      try
          chol(mleVariance);
      catch
          [q,D] = eig(mleVariance);
          d= diag(D);
          d(d <= eps) = 1e-10;
          mleVariance = q*diag(d)*q';
      end

      thetaSamples = mvnrnd(theta, mleVariance, nSamples)'; 
      iThetaSS = cell(nSamples, 1);
      parfor iSample = 1:nSamples
        iThetaSS{iSample} = tm.theta2system(thetaSamples(:,iSample)); %#ok<PFOUS,PFBNS> 
        alphaTilde(:,:,iSample) = iThetaSS{iSample}.smoothSample(y, x, w, alphaHat');        
      end
    end
    
    function [alphaTilde, ssLogli] = smoothSample(obj, y, x, w, alphaHat, nSamples)
      % Durbin & Koopman simulation smoother for draws of the smoothed state
      %
      % See Jarocinski (2015) for details on deviation from Durbin & Koopman.
      %
      % Simulation smoothing by mean corrections: 
      % Step 0 - Run the smoother on y to get alphaHat. 
      % Step 1 - Simulate alpha+ and y+ from draws of epsilon & eta using system parameters.
      % Step 2 - Construct y* = y - y+.
      % Step 3 - Compute alpha* from Kalman smoother on y*
      % Step 4 - Comute alphaTilde = alphaHat + alpha+ - alpha*, a draw of the state. 
      
      % Step 0 - Run the smoother on y to get alphaHat.
      if nargin < 3 
        x = [];
      end
      if nargin < 4
        w = [];
      end
      
      [obj, y, x, w] = obj.checkSample(y, x, w);

      if nargin < 5 || isempty(alphaHat)
        alphaHat = obj.smooth(y, x, w);
      end
      if nargin < 6
        nSamples = 1;
      end
      
      alphaTilde = nan([size(alphaHat) nSamples]);
      ssLogli = nan(nSamples, 1);
      for iS = 1:nSamples
        % Step 1 - Generate alpha+ and y+
        [yPlus, alphaPlus] = generateSimulationData(obj);
        
        % Step 2 - Construct y* = y - y+.
        yStar = y - yPlus;
        
        % Step 3 - Compute alpha* from Kalman smoother on y*
        [alphaStar, sOut] = obj.smooth(yStar, zeros(size(x)));
        ssLogli(iS) = sOut.logli;
        
        % Step 4 - Comute alphaTilde = alphaHat + alpha+ - alpha*, a draw of the state.
        alphaTilde(:,:,iS) = alphaStar + alphaPlus;
      end
      if size(alphaTilde, 2) ~= obj.p
        alphaTilde = permute(alphaTilde, [2 1 3]);
      end
    end
    
    function irf = impulseState(obj, nPeriods)
      % Impulse response functions for the states to a one standard deviation shock.
      % 
      % Arguments:
      %   nPeriods (double): observed data (p x T)
      %
      % Returns:
      %   irf (double): impulse response, ordered by (state, period, shock).
      %
      % The calendar used for the parameters assumes the shock occurs in period 1 of the
      % data sample. 
      
      if ~isscalar(nPeriods)
        irfTau = nPeriods;
      else
        irfTau = 1:nPeriods;
      end
      
      if obj.timeInvariant
        obj.n = nPeriods;
        obj = obj.setInvariantTau();
      end
      
      irf = nan(obj.m, nPeriods, obj.g);
      for iShock = 1:obj.g
        irf(:,1,iShock) = obj.R(:,:,obj.tau.R(irfTau(1))) * obj.Q(:,iShock);
        for iPeriod = 2:nPeriods
          irf(:,iPeriod,iShock) = obj.T(:,:,obj.tau.T(irfTau(iPeriod))) * ...
            irf(:,iPeriod-1,iShock);
        end
      end
    end
    
    function [stateErr, hd] = decompState(obj, y, x, w, state, a0)
      % Historical decompositions for the states to the structural shocks.
      %
      % Arguments:
      %   y (double): observed data (p x T)
      %   x (double): exogenous measreument data (k x T)
      %   w (double): exogenous state data (l x T)
      %   state (double): state estimate (either a or alphaHat)
      %   a0 (double): initial state estimate
      %
      % Returns:
      %   stateErr (double): state errors, estimate of eta
      %   hd (double): historical decomposition, ordered by (period, state, shock)

      [~, stateErr] = getErrors(obj, y, x, w, state, a0);

      hd = zeros(obj.n,obj.m,obj.g);
      for iShock=1:obj.g
        vtilde = [stateErr; zeros(obj.m-obj.g,obj.n)];
        vtilde(iShock,:) = zeros(1,obj.n);
        xi_tilde = obj.c(:,obj.tau.c(1)) + obj.T(:,:,obj.tau.T(1))*a0+vtilde(:,1);
        hd(1,:,iShock) = xi_tilde(1:obj.m);
        for iPeriod=2:obj.n
            xi_tilde = obj.c(:,obj.tau.c(iPeriod)) + obj.T(:,:,obj.tau.T(iPeriod))*xi_tilde + vtilde(:,iPeriod);
            hd(iPeriod,:,iShock) = xi_tilde(1:obj.m);
        end
      end
    end

    function [obsErr, stateErr] = getErrors(obj, y, x, w, state, a0)
      % Get the errors epsilon & eta given an estimate of the state. 
      % Either the filtered or smoothed estimates can be calculated by passing
      % either the filtered state (a) or smoothed state (alphaHat).
      %
      % Arguments:
      %   y (double): observed data (p x T)
      %   x (double): exogenous measreument data (k x T)
      %   w (double): exogenous state data (l x T)
      %   state (double): state estimate (either a or alphaHat)
      %   a0 (double): initial state estimate
      %
      % Returns:
      %   obsErr (double): observation errors, estimates of epsilon
      %   stateErr (double): state errors, estimate of eta
      
      % Make sure the object is set up correctly but DO NOT factor as is done
      % for the univariate filter.
      if isempty(obj.n)
        obj.n = size(state, 2);
        obj = obj.setInvariantTau();
      else
        assert(obj.n == size(state, 2), ...
          'Size of state doesn''t match time dimension of StateSpace.');
      end
      
      % We need to index by time on the second dimension of x and w, so make sure they're
      % the right size.
      if isempty(x)
        x = zeros(0, obj.n);
      end
      if isempty(w)
        w = zeros(0, obj.n+1);
      end
      
      % Iterate through observations
      obsErr = nan(obj.p, obj.n);
      for iT = 1:obj.n
        obsErr(:,iT) = y(:,iT) - ...
          obj.Z(:,:,obj.tau.Z(iT)) * state(:,iT) - ...
          obj.d(:,obj.tau.d(iT)) - ...
          obj.beta(:,:,obj.tau.beta(iT)) * x(:,iT);
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
          obj.T(:,:,obj.tau.T(iT)) * state(:,iT-1) - ...
          obj.c(:,obj.tau.c(iT)) - ...
          obj.gamma(:,:,obj.tau.gamma(iT)) * w(:,iT));
      end
    end
    
    function [V, J] = getErrorVariances(obj, y, fOut, sOut)
      % Get the smoothed state variance and covariance matricies
      % Either the filtered or smoothed estimates can be calculated by passing
      % either the filtered state (a) or smoothed state (alphaHat).
      %
      % Arguments:
      %   y (double): observed data (p x T)
      %   fOut (struct): structure of additional quantites from the filter
      %   sOut (struct): structure of additional quantites from the smoother
      %
      % Returns:
      %   V (double): Var(alpha | Y_n). (m x m x T)
      %   J (double): Cov(alpha_{t+1}, alpha_t | Y_n).  (m x m x T)
      
      Ldagger = obj.build_Ldagger(y, fOut);
      
      I = eye(obj.m);
      V = nan(obj.m, obj.m, obj.n);
      J = nan(obj.m, obj.m, obj.n);
      
      sOut.N(:,:,obj.n+1) = 0;
      
      for iT = obj.n:-1:1
        iP = fOut.P(:,:,iT);
        V(:,:,iT) = iP - iP * sOut.N(:,:,iT) * iP;   
        J(:,:,iT) = iP * Ldagger(:,:,iT)' * ...
          (I - sOut.N(:,:,iT+1) * fOut.P(:,:,iT+1));
      end
    end
    
    function obj = setDefaultInitial(obj)
      % Set default a0 and P0.
      % Run before filter/smoother after a0 & P0 inputs have been processed.
      
      if ~isempty(obj.a0) && ~isempty(obj.P0)
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
      
      if isempty(obj.a0)
        obj.a0 = zeros(obj.m, 1);
        
        a0temp = (eye(mStationary) - tempT) \ obj.c(stationary, obj.tau.c(1));
        obj.a0(stationary) = a0temp;
      end
      
      if isempty(obj.P0)
        newA0 = select(:, nonstationary);
        newR0 = select(:, stationary);
        
        tempR = obj.R(stationary, :, obj.tau.R(1));
        tempQ = obj.Q(:, :, obj.tau.Q(1));
        try
          newQ0 = reshape((eye(mStationary^2) - kron(tempT, tempT)) \ ...
            reshape(tempR * tempQ * tempR', [], 1), ...
            mStationary, mStationary);
        catch ex
          % If the state is large, try making it sparse
          if strcmpi(ex.identifier, 'MATLAB:array:SizeLimitExceeded')
            tempT = sparse(tempT);
            newQ0 = full(reshape((speye(mStationary^2) - kron(tempT, tempT)) \ ...
              reshape(tempR * tempQ * tempR', [], 1), ...
              mStationary, mStationary));
          else
            rethrow(ex);
          end
        end
        
        diffuseP0 = newA0 * newA0';
        diffuseP0(diffuseP0 ~= 0) = Inf;
        newP0 = diffuseP0 + newR0 * newQ0 * newR0';
        obj.P0 = newP0;
      end
    end
  end
  
  methods (Hidden)
    %% Filter/smoother Helper Methods
    function [obj, y, x, w, factorC, oldTau] = prepareFilter(obj, y, x, w)
      % Common setup tasks needed for the filter/smoother.
      
      % Make sure data matches observation dimensions
      obj.validateKFilter();
      [obj, y, x, w] = obj.checkSample(y, x, w);
      
      % Set initial values
      obj = obj.setDefaultInitial();
      
      % Handle multivariate series
      [obj, y, factorC, oldTau] = obj.factorMultivariate(y);
    end
    
    function validateKFilter(obj)
      % Validate parameters to ensure we can compute the likelihood.
      
      obj.validateStateSpace();
      
      % Make sure all of the parameters are known (non-nan)
      assert(~any(cellfun(@(par) any(any(any(isnan(par)))), obj.parameters)), ...
        ['All parameter values must be known. To estimate unknown '...
        'parameters, see StateSpaceEstimation']);
    end
    
    function [ssUni, yUni, factorC, oldTau] = factorMultivariate(obj, y)
      % Compute new Z and H matricies so the univariate treatment can be applied
      %
      % Arguments:
      %   y : observed data
      %
      % Returns:
      %   ssUni : StateSpace model with diagonal measurement error
      %   yUni : data modified for ssUni
      %   factorC : diagonalizing matrix(s) for original H matrix
      %   oldTau : structure of mapping from original tau to new tau
            
      [uniqueOut, ~, newTauH] = unique([obj.tau.H ~isnan(y')], 'rows');
      oldTauH = uniqueOut(:,1);
      obsPatternH = uniqueOut(:,2:end);
      
      % Create factorizations
      maxTauH = max(newTauH);
      factorC = zeros(size(obj.H, 1), size(obj.H, 2), maxTauH);
      newHmat = zeros(size(obj.H, 1), size(obj.H, 2), maxTauH);
      factorCinv = cell(maxTauH, 1);
      for iH = 1:maxTauH
        ind = logical(obsPatternH(iH, :));
        [factorC(ind,ind,iH), newHmat(ind,ind,iH)] = ldl(obj.H(ind,ind,oldTauH(iH)), 'lower');
        factorCinv{iH} = inv(factorC(ind,ind,iH));
        assert(isdiag(newHmat(ind,ind,iH)), 'ldl returned non-diagonal d matrix.');
      end
      newH = struct('Ht', abs(newHmat), 'tauH', newTauH);
      
      inds = logical(obsPatternH(newTauH, :));
      yUni = nan(size(y));
      for iT = 1:size(y,2)
        % Transform observations
        yUni(inds(iT,:),iT) = factorCinv{newTauH(iT)} * y(inds(iT,:),iT);
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
      
      % Same thing for beta
      [uniqueOut, ~, newTaubeta] = unique([obj.tau.beta newTauH ~isnan(y')], 'rows');
      oldTaubeta = uniqueOut(:,1);
      correspondingNewHOldbeta = uniqueOut(:,2);
      obsPattern = uniqueOut(:,3:end);
      
      newbetamat = zeros([size(obj.beta, 1) size(obj.beta, 2) max(newTaubeta)]);
      for ibeta = 1:max(newTaubeta)
        ind = logical(obsPattern(ibeta,:));
        newbetamat(ind,:, ibeta) = factorC(ind,ind,correspondingNewHOldbeta(ibeta)) \ ...
          obj.beta(ind,:,oldTaubeta(ibeta));
      end
      newbeta = struct('betat', newbetamat, 'taubeta', newTaubeta);
      
      [~, ~, T, Q, ~, ~, c, gamma, R] = obj.getInputParameters();
      
      ssUni = StateSpace(newZ, newH, T, Q, ...
        'd', newd, 'beta', newbeta, 'c', c, 'gamma', gamma, 'R', R);
      
      % Set initial values
      ssUni.a0 = obj.a0;
      ssUni.P0 = obj.P0;
      
      oldTau = struct('H', oldTauH, 'Z', oldTauZ, 'd', oldTaud, 'beta', oldTaubeta, ...
        'correspondingNewHOldZ', correspondingNewHOldZ, ...
        'correspondingNewHOldd', correspondingNewHOldd, ...
        'correspondingNewHOldbeta', correspondingNewHOldbeta, ...
        'obsPattern', obsPatternH);
    end
    
    %% General utilities
    function [stationary, nonstationary] = findStationaryStates(obj)
      % Find which states have stationary distributions given the T matrix.
      
      [V, D] = eig(obj.T(:,:,obj.tau.T(1)));
      bigEigs = abs(diag(D)) >= 1 - 10 * eps; % Fudge factor of 10 * eps
      
      nonstationary = find(any(V(:, bigEigs), 2));
      
      % I think we don't need a loop here to find other states that have
      % loadings on the nonstationary states (the eigendecomposition does this
      % for us) but I'm not sure.
      stationary = setdiff(1:obj.m, nonstationary);
      
      assert(all(abs(eig(obj.T(stationary,stationary,1))) < 1), ...
        ['Stationary section of T isn''t actually stationary. \n' ...
        'Likely development error.']);
    end
    
    function [Z, H, T, Q, d, beta, c, gamma, R] = getInputParameters(obj)
      % Helper function to get parameters for constructor
      
      if ~isempty(obj.tau)
        Z = struct('Tt', obj.Z, 'tauT', obj.tau.Z);
        d = struct('dt', obj.d, 'taud', obj.tau.d);
        beta = struct('betat', obj.beta, 'taubeta', obj.tau.beta);
        H = struct('Rt', obj.H, 'tauR', obj.tau.H);
        
        T = struct('Tt', obj.T, 'tauT', obj.tau.T);
        c = struct('ct', obj.c, 'tauc', obj.tau.c);
        gamma = struct('gammat', obj.gamma, 'taugamma', obj.tau.gamma);
        R = struct('Rt', obj.R, 'tauR', obj.tau.R);
        Q = struct('Qt', obj.Q, 'tauQ', obj.tau.Q);
      else
        Z = obj.Z;
        d = obj.d;
        beta = obj.beta;
        H = obj.H;
        
        T = obj.T;
        c = obj.c;
        gamma = obj.gamma;
        R = obj.R;
        Q = obj.Q;
      end
    end
  
    %% Derivative functions
    function [ll, grad, fOut] = gradFinDiff(obj, y, x, w, tm, theta)
      % Compute numeric gradient using central differences
      
      [~, ll, fOut] = obj.filter(y, x, w);
      
      nTheta = tm.nTheta;
      grad = nan(nTheta, 1);
      
      stepSize = 0.5 * obj.delta;
      for iTheta = 1:nTheta
        % For each element of theta, compute finite differences derivative
        try
          thetaDown = theta - [zeros(iTheta-1,1); stepSize; zeros(nTheta-iTheta,1)];
          ssDown = tm.theta2system(thetaDown);
          [~, llDown] = ssDown.filter(y, x, w);
          
          thetaUp = theta + [zeros(iTheta-1,1); stepSize; zeros(nTheta-iTheta,1)];
          ssUp = tm.theta2system(thetaUp);
          [~, llUp] = ssUp.filter(y, x, w);
          
          if obj.numericGradPrec == 1
            grad(iTheta) = (llUp - llDown) ./ (2 * stepSize);
          else
            thetaDown2 = theta - [zeros(iTheta-1,1); 2 * stepSize; zeros(nTheta-iTheta,1)];
            ssDown2 = tm.theta2system(thetaDown2);
            [~, llDown2] = ssDown2.filter(y, x, w);
            
            thetaUp2 = theta + [zeros(iTheta-1,1); 2 * stepSize; zeros(nTheta-iTheta,1)];
            ssUp2 = tm.theta2system(thetaUp2);
            [~, llUp2] = ssUp2.filter(y, x, w);
            
            grad(iTheta) = (llDown2 - 8 * llDown + 8 * llUp - llUp2) ./ (12 * stepSize);
          end
        catch ex
          % We might get a bad model. If we do, ignore it and move on - it'll
          % get a -Inf gradient. Other errors may be of concern though.
          if ~strcmp(ex.identifier, 'StateSpace:filter:degenerate')
            rethrow(ex);
          end
        end
      end
      
      % Handle bad evaluations
      grad(imag(grad) ~= 0 | isnan(grad)) = -Inf;
    end
    
    function [ll, grad, fOut] = gradFinDiff_parallel(obj, y, x, w, tm, theta)
      % Compute numeric gradient using central differences
      
      [~, ll, fOut] = obj.filter(y, x, w);
      nTheta = tm.nTheta;
      
      if obj.numericGradPrec ~= 1
        % No reason for this other than maintainability
        error('numericGradPrec = 2 not suported in parallel.')
      end
      
      % Compile likelihoods
      grad = nan(nTheta, 1);
      stepSize = 0.5 * obj.delta;
      parfor iTheta = 1:nTheta
        thetaDown = theta - [zeros(iTheta-1,1); stepSize; zeros(nTheta-iTheta,1)];
        ssDown = tm.theta2system(thetaDown); %#ok<PFBNS>
        [~, llDown] = ssDown.filter(y, x, w);
        
        thetaUp = theta + [zeros(iTheta-1,1); stepSize; zeros(nTheta-iTheta,1)];
        ssUp = tm.theta2system(thetaUp);
        [~, llUp] = ssUp.filter(y, x, w);
        
        grad(iTheta) = (llUp - llDown) ./ (2 * stepSize);
      end
      
      % Handle bad evaluations
      grad(imag(grad) ~= 0 | isnan(grad)) = -Inf;
    end
    
    function hessian = gradFinDiffSecond(obj, y, x, w, tm, theta)
      % Get the matrix of 2nd derivatives
      
      nTheta = tm.nTheta;
      hessian = nan(nTheta);
      
      stepSize = 0.5 * sqrt(obj.delta);
      stepMat = stepSize * eye(nTheta);
      
      parfor iTheta = 1:nTheta
        for jTheta = 1:nTheta
          % For each element of theta, compute finite differences derivative
          try
            thetaUpUp = theta + stepMat(:,iTheta) + stepMat(:,jTheta); %#ok<PFBNS> 
            ssUpUp = tm.theta2system(thetaUpUp); %#ok<PFBNS> 
            [~, llUpUp] = ssUpUp.filter(y, x, w);
            
            thetaUpDown = theta + stepMat(:,iTheta) - stepMat(:,jTheta);
            ssUpDown = tm.theta2system(thetaUpDown);
            [~, llUpDown] = ssUpDown.filter(y, x, w);
            
            thetaDownUp = theta - stepMat(:,iTheta) + stepMat(:,jTheta);
            ssDownUp = tm.theta2system(thetaDownUp);
            [~, llDownUp] = ssDownUp.filter(y, x, w);
            
            thetaDownDown = theta - stepMat(:,iTheta) - stepMat(:,jTheta);
            ssDownDown = tm.theta2system(thetaDownDown);
            [~, llDownDown] = ssDownDown.filter(y, x, w);
            
            hessian(iTheta, jTheta) = (llUpUp - llUpDown - llDownUp + llDownDown) / ...
              (4 * stepSize * stepSize);
            
          catch ex
            % We might get a bad model. If we do, ignore it and move on - it'll
            % get a -Inf gradient. Other errors may be of concern though.
            if ~strcmp(ex.identifier, 'StateSpace:filter:degenerate')
              rethrow(ex);
            end
          end
        end
      end
      
      % Handle bad evaluations
      hessian(imag(hessian) ~= 0 | isnan(hessian)) = -Inf;
    end
    
  end
  
  methods (Hidden)
    %% Filter/smoother/gradient mathematical methods
    function [a, logli, filterOut] = filter_m(obj, y, x, w)
      % Filter using exact initial conditions
      %
      % See "A Practitioner's Guide and Matlab Toolbox for Mixed Frequency State Space 
      % Models", Appendix A. 
      
      assert(all(arrayfun(@(iH) isdiag(obj.H(:,:,iH)), ...
        1:size(obj.H, 3))), 'Univarite only!');
      
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
      
      % Initialize
      iT = 0;
      Tii = obj.T(:,:,obj.tau.T(iT+1));
      a(:,iT+1) = Tii * obj.a0 + obj.c(:,obj.tau.c(iT+1)) + ...
        obj.gamma(:,:,obj.tau.gamma(iT+1)) * w(:,iT+1);
      
      Pd0 = obj.A0 * obj.A0';
      Pstar0 = obj.R0 * obj.Q0 * obj.R0';
      Pd(:,:,iT+1)  = Tii * Pd0 * Tii';
      Pstar(:,:,iT+1) = Tii * Pstar0 * Tii' + ...
        obj.R(:,:,obj.tau.R(iT+1)) * obj.Q(:,:,obj.tau.Q(iT+1)) * obj.R(:,:,obj.tau.R(iT+1))';
      
      % Initial recursion
      while ~all(all(Pd(:,:,iT+1) < eps))
        if iT >= obj.n
          error('StateSpace:filter:degenerate', ['Degenerate model. ' ...
            'Exact initial filter unable to transition to standard filter.']);
        end
        
        iT = iT + 1;
        ind = find(~isnan(y(:,iT)));
        
        ati = a(:,iT);
        Pstarti = Pstar(:,:,iT);
        Pdti = Pd(:,:,iT);
        for iP = ind'
          Zjj = obj.Z(iP,:,obj.tau.Z(iT));
          v(iP,iT) = y(iP, iT) - Zjj * ati - obj.d(iP,obj.tau.d(iT)) ...
            - obj.beta(iP,:,obj.tau.beta(iT)) * x(:,iT);
          
          Fd(iP,iT) = Zjj * Pdti * Zjj';
          Fstar(iP,iT) = Zjj * Pstarti * Zjj' + obj.H(iP,iP,obj.tau.H(iT));
          
          Kd(:,iP,iT) = Pdti * Zjj' ./ Fd(iP,iT);
          Kstar(:,iP,iT) = Pstarti * Zjj' ./ Fstar(iP,iT);
          
          if abs(Fd(iP,iT)) > eps
            % F diffuse nonsingular
            ati = ati + Kd(:,iP,iT) * v(iP,iT);
            
            Pstarti = Pstarti - ...
              (Kstar(:,iP,iT) * Kd(:,iP,iT)' + Kd(:,iP,iT) * Kstar(:,iP,iT)' ...
              - Kd(:,iP,iT) * Kd(:,iP,iT)') .* Fstar(iP,iT);
            
            Pdti = Pdti - Kd(:,iP,iT) .* Kd(:,iP,iT)' .* Fd(iP,iT);
            
            LogL(iP,iT) = log(Fd(iP,iT));
          else
            % F diffuse = 0
            ati = ati + Kstar(:,iP,iT) .* v(iP,iT);
            
            Pstarti = Pstarti - Kstar(:,iP,iT) * Kstar(:,iP,iT)' .* Fstar(iP,iT);
            
            % Pdti = Pdti;
            
            LogL(iP,iT) = (log(Fstar(iP,iT)) + (v(iP,iT)^2) ./ Fstar(iP,iT));
          end
        end
        
        Tii = obj.T(:,:,obj.tau.T(iT+1));
        a(:,iT+1) = Tii * ati + obj.c(:,obj.tau.c(iT+1)) + ...
          obj.gamma(:,:,obj.tau.gamma(iT+1)) * w(:,iT+1);
        
        Pd(:,:,iT+1)  = Tii * Pdti * Tii';
        Pstar(:,:,iT+1) = Tii * Pstarti * Tii' + ...
          obj.R(:,:,obj.tau.R(iT+1)) * obj.Q(:,:,obj.tau.Q(iT+1)) * obj.R(:,:,obj.tau.R(iT+1))';
      end
      
      dt = iT;
      
      F = Fstar;
      K = Kstar;
      P = Pstar;
      
      % Standard Kalman filter recursion
      for iT = dt+1:obj.n
        ind = find(~isnan(y(:,iT)));
        ati    = a(:,iT);
        Pti    = P(:,:,iT);
        for iP = ind'
          Zjj = obj.Z(iP,:,obj.tau.Z(iT));
          
          v(iP,iT) = y(iP,iT) - Zjj * ati - obj.d(iP,obj.tau.d(iT)) ...
            - obj.beta(iP,:,obj.tau.beta(iT)) * x(:,iT);
          
          F(iP,iT) = Zjj * Pti * Zjj' + obj.H(iP,iP,obj.tau.H(iT));
          K(:,iP,iT) = Pti * Zjj' / F(iP,iT);
          
          LogL(iP,iT) = (log(F(iP,iT)) + (v(iP,iT)^2) / F(iP,iT));
          
          ati = ati + K(:,iP,iT) * v(iP,iT);
          Pti = Pti - K(:,iP,iT) * F(iP,iT) * K(:,iP,iT)';
        end
        
        Tii = obj.T(:,:,obj.tau.T(iT+1));
        
        a(:,iT+1) = Tii * ati + obj.c(:,obj.tau.c(iT+1)) + ...
          obj.gamma(:,:,obj.tau.gamma(iT+1)) * w(:,iT+1);
        P(:,:,iT+1) = AbstractSystem.enforceSymmetric(Tii * Pti * Tii' + ...
          obj.R(:,:,obj.tau.R(iT+1)) * obj.Q(:,:,obj.tau.Q(iT+1)) * obj.R(:,:,obj.tau.R(iT+1))');
      end
      
      logli = -(0.5 * sum(sum(isfinite(y)))) * log(2 * pi) - 0.5 * sum(sum(LogL));      
      filterOut = struct('a', a, 'P', P, 'Pd', Pd, 'v', v, 'F', F, 'Fd', Fd, ...
        'K', K, 'Kd', Kd, 'dt', dt);
    end
    
    function [a, logli, filterOut] = filter_mex(obj, y, x, w)
      % Call mex function filter_uni
      %
      % Provides identical output to filter_m but is roughtly 15x faster.
      
      if isempty(obj.beta)
        % Correction to pass arrays to mex 
        obj.beta = obj.beta(:,:,1);
        obj.tau.beta = ones(size(obj.tau.beta));
      end
      
      ssStruct = struct('Z', obj.Z, 'd', obj.d, 'beta', obj.beta, 'H', obj.H, ...
        'T', obj.T, 'c', obj.c, 'gamma', obj.gamma, 'R', obj.R, 'Q', obj.Q, ...
        'a0', obj.a0, 'A0', obj.A0, 'R0', obj.R0, 'Q0', obj.Q0, ...
        'tau', obj.tau);
      if isempty(ssStruct.R0)
        ssStruct.R0 = zeros(obj.m, 1);
        ssStruct.Q0 = 0;
      end
      if isempty(ssStruct.A0)
        ssStruct.A0 = zeros(obj.m, 1);
      end
      
      [a, logli, P, Pd, v, F, Fd, K, Kd, dt] = mfss_mex.filter_uni(y, x, w, ssStruct);
      filterOut = struct('a', a, 'P', P, 'Pd', Pd, 'v', v, 'F', F, 'Fd', Fd, ...
        'K', K, 'Kd', Kd, 'dt', dt);
    end
    
    function [alpha, smootherOut] = smoother_m(obj, y, fOut)
      % Univarite smoother taking exact initial conditions into account
      %
      % See "A Practitioner's Guide and Matlab Toolbox for Mixed Frequency State Space 
      % Models", Appendix A. 
      
      % Preallocation
      alpha = zeros(obj.m, obj.n);
      V     = zeros(obj.m, obj.m, obj.n);
      eta   = zeros(obj.g, obj.n);
      r     = zeros(obj.m, obj.n);
      N     = zeros(obj.m, obj.m, obj.n);
      
      Im = eye(obj.m);
      
      rti = zeros(obj.m,1);
      Nti = zeros(obj.m,obj.m);
      for iT = obj.n:-1:fOut.dt+1
        ind = flipud(find(~isnan(y(:,iT))));
        
        for iP = ind'
          % Generate t,i quantities
          Zti = obj.Z(iP,:,obj.tau.Z(iT));
          Lti = Im - fOut.K(:,iP,iT) * Zti;
          
          % Transition to t,i-1
          rti = Zti' ./ fOut.F(iP,iT) .* fOut.v(iP,iT) + Lti' * rti;
          Nti = Zti' ./ fOut.F(iP,iT) * Zti + Lti' * Nti * Lti;
        end
        r(:,iT) = rti;
        N(:,:,iT) = Nti;
        
        alpha(:,iT) = fOut.a(:,iT) + fOut.P(:,:,iT) * r(:,iT);
        V(:,:,iT) = fOut.P(:,:,iT) - fOut.P(:,:,iT) * N(:,:,iT) * fOut.P(:,:,iT);
        eta(:,iT) = obj.Q(:,:,obj.tau.Q(iT+1)) * obj.R(:,:,obj.tau.R(iT+1))' * r(:,iT);
        
        rti = obj.T(:,:,obj.tau.T(iT))' * rti;
        Nti = AbstractSystem.enforceSymmetric(...
          obj.T(:,:,obj.tau.T(iT))' * Nti * obj.T(:,:,obj.tau.T(iT)));
      end
      
      r1 = zeros(obj.m, fOut.dt+1);
      
      % Note: r0 = r and N0 = N
      r0ti = rti;
      r1ti = r1(:,fOut.dt+1);
      N0ti = Nti;
      N1ti = zeros(obj.m, obj.m);
      N2ti = zeros(obj.m, obj.m);
      
      % Exact initial smoother
      for iT = fOut.dt:-1:1
        ind = flipud(find(~isnan(y(:,iT))));
        for iP = ind'
          Zti = obj.Z(iP,:,obj.tau.Z(iT));
          
          if fOut.Fd(iP,iT) ~= 0
            % Diffuse case
            Ldti = Im - fOut.Kd(:,iP,iT) * Zti;
            L0ti = (fOut.Kd(:,iP,iT) - fOut.K(:,iP,iT)) * Zti * fOut.F(iP,iT) ./ fOut.Fd(iP,iT);
            r1ti = Zti' / fOut.Fd(iP,iT) * fOut.v(iP,iT) + L0ti' * r0ti + Ldti' * r1ti;
            % Be sure to compute r0ti after r1ti since it is used in r1ti
            r0ti = Ldti' * r0ti;
            
            N0ti = Ldti' * N0ti * Ldti;
            N1ti = Zti' / fOut.Fd(iP,iT) * Zti + Ldti' * N0ti * L0ti + Ldti' * N1ti * Ldti;
            N2ti = Zti' * fOut.Fd(iP,iT)^(-2) * Zti * fOut.F(iP,iT) + ...
              L0ti' * N1ti * L0ti + Ldti' * N1ti * L0ti + ...
              L0ti' * N1ti * Ldti + Ldti' * N2ti * Ldti;
          else
            % Known
            Lstarti = eye(obj.m) - fOut.K(:,iP,iT) * Zti;
            r0ti = Zti' / fOut.F(iP,iT) * fOut.v(iP,iT) + Lstarti' * r0ti;
            
            N0ti = Zti' / fOut.F(iP,iT) * Zti + Lstarti' * N0ti * Lstarti;
          end
        end
        
        r(:,iT) = r0ti;
        r1(:,iT) = r1ti;
        N(:,:,iT) = N0ti;
        
        alpha(:,iT) = fOut.a(:,iT) + fOut.P(:,:,iT) * r(:,iT) + ...
          fOut.Pd(:,:,iT) * r1(:,iT);
        V(:,:,iT) = fOut.P(:,:,iT) - ...
          fOut.P(:,:,iT) * N0ti * fOut.P(:,:,iT) - ...
          (fOut.Pd(:,:,iT) * N1ti * fOut.P(:,:,iT))' - ...
          fOut.P(:,:,iT) * N1ti * fOut.Pd(:,:,iT) - ...
          fOut.Pd(:,:,iT) * N2ti * fOut.Pd(:,:,iT);
        
        eta(:,iT) = obj.Q(:,:,obj.tau.Q(iT)) * obj.R(:,:,obj.tau.R(iT))' * r(:,iT);
        
        r0ti = obj.T(:,:,obj.tau.T(iT))' * r0ti;
        r1ti = obj.T(:,:,obj.tau.T(iT))' * r1ti;
        
        N0ti = obj.T(:,:,obj.tau.T(iT))' * N0ti * obj.T(:,:,obj.tau.T(iT));
        N1ti = obj.T(:,:,obj.tau.T(iT))' * N1ti * obj.T(:,:,obj.tau.T(iT));
        N2ti = obj.T(:,:,obj.tau.T(iT))' * N2ti * obj.T(:,:,obj.tau.T(iT));
      end
      
      Pstar0 = obj.R0 * obj.Q0 * obj.R0';
      if fOut.dt > 0
        Pd0 = obj.A0 * obj.A0';
        a0tilde = obj.a0 + Pstar0 * r0ti + Pd0 * r1ti;
      else
        a0tilde = obj.a0 + Pstar0 * rti;
      end
      
      smootherOut = struct('alpha', alpha, 'V', V, 'eta', eta, 'r', r, ...
        'N', N, 'a0tilde', a0tilde, 'r1', r1);
    end
    
    function [alpha, smootherOut] = smoother_mex(obj, y, fOut)
      % Call mex function smoother_uni
      %
      % Provides identical output to smoother_m but is roughtly 10x faster.
      
      ssStruct = struct('Z', obj.Z, 'H', obj.H, ...
        'T', obj.T, 'R', obj.R, 'Q', obj.Q, ...
        'a0', obj.a0, 'A0', obj.A0, 'R0', obj.R0, 'Q0', obj.Q0, ...
        'tau', obj.tau);
      if isempty(ssStruct.R0)
        ssStruct.R0 = zeros(obj.m, 1);
        ssStruct.Q0 = 0;
      end
      if isempty(ssStruct.A0)
        ssStruct.A0 = zeros(obj.m, 1);
      end
      
      [alpha, eta, r, N, V, a0tilde] = mfss_mex.smoother_uni(y, ssStruct, fOut);
      smootherOut = struct('alpha', alpha, 'V', V, 'eta', eta, 'r', r, ...
        'N', N, 'a0tilde', a0tilde);
    end
    
    %% Decomposition mathematical methods
    function fWeights = filter_weights(obj, y, x, w, fOut, ssMulti, C)
      % Decompose the effect of the data on the filtered state.
      %
      % Inputs:
      %   y (double): observed data (p x T)
      %   x (double): exogenous measreument data (k x T)
      %   w (double): exogenous state data (l x T)
      %   fOut (double): additional quantities from the filter
      %   ssMulti (double): StateSpace before trasformed to univariate
      %   C (double): Cholesky from univariate transformation
      %
      % Outputs:
      %   omega: (m x p x T+1 x T) array organized as
      %         (state, observation, effectPeriod, contributionPeriod)
      %   omegac: (m x m x T+1 x T+1) array organized as
      %         (state, observation, effectPeriod, contributionPeriod)
      %   omegad: (m  p x T+1 x T) array organized as
      %         (state, observation, effectPeriod, contributionPeriod)
      %   omegax: (m x k x T+1 x T) array organized as
      %        (state, observation, effectPeriod, contributionPeriod)
      %   omegaw: (m x l x T+1 x T) array organized as
      %         (state, observation, effectPeriod, contributionPeriod)
      %   omegaa0: (m x m x T+1) array organized as
      %         (state, observation, effectPeriod, contributionPeriod)
      %
      % See "A Practitioner's Guide and Matlab Toolbox for Mixed Frequency State Space 
      % Models", Appendix C.
      
      % Create cell arrays that are (T+1 x T) where weight matricies will be placed.
      omega = cell(obj.n+1, obj.n);
      omegac = cell(obj.n+1, obj.n+1);
      omegad = cell(obj.n+1, obj.n);
      omegax = cell(obj.n+1, obj.n);
      omegaw = cell(obj.n+1, obj.n+1);
      omegaa0 = cell(obj.n+1, 1);

      Im = eye(obj.m);
      
      % Kstar - the effect of the whole observable on the state
      Kstar = zeros(obj.m, obj.p, obj.n);
      % Lstar - propogation of current state to next period
      Lstar = zeros(obj.m, obj.m, obj.n+1);
      Lstar(:,:,1) = obj.T(:,:,obj.tau.T(1));
      
      eps2 = eps^2;
      
      % Loop over the data and determine effects of observations as they happen
      for iJ = 1:obj.n
        T = obj.T(:,:,obj.tau.T(iJ+1));
        Lproduct = Im;
        KstarTemp = zeros(obj.m, obj.p);
        
        for iP = obj.p:-1:1
          % If we don't observe a y value we don't learn anything about the
          % state (Kstar doesn't get any nonzero values and Lproduct doesn't change so we 
          % can just skip the iteration). 
          if isnan(y(iP,iJ))
            continue;
          end
          
          % Select between diffuse and standar value of K.
          if iJ > fOut.dt || fOut.Fd(iP,iJ) == 0
            K = fOut.K(:,iP,iJ);
          else
            K = fOut.Kd(:,iP,iJ);
          end
          KstarTemp(:,iP) = Lproduct * K;
          
          Lproduct = Lproduct * (Im - K * obj.Z(iP,:,obj.tau.Z(iJ)));
        end
        Kstar(:,:,iJ) = T * KstarTemp;
        Lstar(:,:,iJ+1) = T * Lproduct;
        
        ind = ~isnan(y(:,iJ));
        
        % Determine effect of data/parameters the period after observation
        KCinv = Kstar(:,ind,iJ) / C(ind,ind,obj.tau.H(iJ));
        
        omega_temp = KCinv * diag(y(ind,iJ));
        if any(any(abs(omega_temp) > eps2))
          omega{iJ+1,iJ} = zeros(obj.m, obj.p);
          omega{iJ+1,iJ}(:,ind) = omega_temp;
        end
        
        omegad_temp = -KCinv * diag(ssMulti.d(ind, ssMulti.tau.d(iJ)));
        if any(any(abs(omegad_temp) > eps2))
          omegad{iJ+1,iJ} = zeros(obj.m, obj.p);
          omegad{iJ+1,iJ}(:,ind) = omegad_temp;
        end
        
        omegax_temp = -KCinv * ssMulti.beta(ind, :, ssMulti.tau.beta(iJ)) * diag(x(:,iJ));
        if any(any(abs(omegax_temp) > eps2))
          omegax{iJ+1,iJ} = zeros(obj.m, obj.k);
          omegax{iJ+1,iJ} = omegax_temp;
        end
        
        omegac_temp = diag(obj.c(:,obj.tau.c(iJ)));
        if any(any(abs(omegac_temp) > eps2))
          omegac{iJ,iJ} = omegac_temp;
        end
        
        omegaw_temp = obj.gamma(:,:,obj.tau.gamma(iJ)) * diag(w(:,iJ));
        if any(any(abs(omegaw_temp) > eps2))
          omegaw{iJ,iJ} = omegaw_temp;
        end
      end
      % Get the effect on the T+1 period filtered state
      omegac{obj.n+1,obj.n+1} = diag(obj.c(:,obj.tau.c(obj.n+1)));
      omegaw{obj.n+1,obj.n+1} = obj.gamma(:,:,obj.tau.gamma(obj.n+1)) * diag(w(:,obj.n+1)); 
            
      % Propogate effect forward to other time periods of states
      for iJ = 1:obj.n
        % c (part 1)
        if ~isempty(omegac{iJ,iJ})
          omegac{iJ+1,iJ} = Lstar(:,:,iJ+1) * omegac{iJ,iJ};
        end
        
        % w (part 1)
        if ~isempty(omegaw{iJ,iJ})
          omegaw{iJ+1,iJ} = Lstar(:,:,iJ+1) * omegaw{iJ,iJ};
        end
        
        % y
        for iT = iJ+1:obj.n
          if ~isempty(omega{iT,iJ})
            omega_temp = Lstar(:,:,iT+1)  * omega{iT,iJ};
            if all(abs(omega_temp) < eps2)
              break
            end
            omega{iT+1,iJ} = omega_temp;
          end
        end
        
        % d
        for iT = iJ+1:obj.n
          if ~isempty(omegad{iT,iJ})
            omegad_temp = Lstar(:,:,iT+1) * omegad{iT,iJ};
            if all(abs(omegad_temp) < eps2)
              break
            end
            omegad{iT+1,iJ} = omegad_temp;
          end
        end
        
        % x
        for iT = iJ+1:obj.n
          if ~isempty(omegax{iT,iJ})
            omegax_temp = Lstar(:,:,iT+1) * omegax{iT,iJ};
            if all(abs(omegax_temp) < eps2)
              break
            end
            omegax{iT+1,iJ} = omegax_temp;
          end
        end
        
        % c (part 2)
        for iT = iJ+1:obj.n
          if ~isempty(omegac{iT,iJ})
            omegac_temp = Lstar(:,:,iT+1) * omegac{iT,iJ};
            if all(abs(omegac_temp) < eps2)
              break
            end
            omegac{iT+1,iJ} = omegac_temp;
          end
        end
        
        % w (part 2)
        for iT = iJ+1:obj.n
          if ~isempty(omegaw{iT,iJ})
            omegaw_temp = Lstar(:,:,iT+1) * omegaw{iT,iJ};
            if all(abs(omegaw_temp) < eps2)
              break
            end
            omegaw{iT+1,iJ} = omegaw_temp;
          end
        end
        
      end
      
      % Determine effect of initial conditions
      omegaa0{1} = obj.T(:,:,obj.tau.T(1)) * diag(obj.a0);
      for iT = 2:obj.n+1
        omegaa0_temp = Lstar(:,:,iT) * omegaa0{iT-1};
        if all(abs(omegaa0_temp) < eps2)
          break
        end
        omegaa0{iT} = omegaa0_temp;        
      end
      
      fWeights = struct('y', {omega}, 'd', {omegad}, 'x', {omegax}, ...
        'c', {omegac}, 'w', {omegaw}, 'a0', {omegaa0});
    end
    
    function sWeights = smoother_weights(obj, y, x, w, fOut, ssMulti, C)
      % Decompose the effect of the data on the filtered state.
      % Inputs:
      %   y (double): observed data (p x T)
      %   x (double): exogenous measreument data (k x T)
      %   w (double): exogenous state data (l x T)
      %   fOut (double): additional quantities from the filter
      %   ssMulti (double): StateSpace before trasformed to univariate
      %   C (double): Cholesky from univariate transformation
      %
      % Outputs:
      %   omega: (m x p x T x T) array organized as
      %         (state, observation, effectPeriod, contributionPeriod)
      %   omegac: (m x m x T x T) array organized as
      %         (state, observation, effectPeriod, contributionPeriod)
      %   omegad: (m x p x T x T) array organized as
      %         (state, observation, effectPeriod, contributionPeriod)
      %   omegax: (m x k x T x T) array organized as
      %         (state, observation, effectPeriod, contributionPeriod)
      %   omegaw: (m x l x T x T) array organized as
      %         (state, observation, effectPeriod, contributionPeriod)
      %   omegaa0: (m x m x T) array organized as
      %         (state, observation, effectPeriod, contributionPeriod)      
      %
      % See "A Practitioner's Guide and Matlab Toolbox for Mixed Frequency State Space 
      % Models", Appendix C.
      
      % Filter weights (a_t)
      fWeight = obj.filter_weights(y, x, w, fOut, ssMulti, C);
      [rWeight, r1Weight] = obj.r_weights(y, x, fOut, fWeight, ssMulti, C);
      
      % Calculate smoothed state weights
      omega = cell(obj.n+1, obj.n);
      omegac = cell(obj.n+1, obj.n+1);
      omegad = cell(obj.n+1, obj.n);
      omegax = cell(obj.n+1, obj.n);
      omegaw = cell(obj.n+1, obj.n+1);
      omegaa0 = cell(obj.n+1, 1);
      
      zeroMP = zeros(obj.m, obj.p);
      zeroMK = zeros(obj.m, obj.k);
      zeroMM = zeros(obj.m, obj.m);
      zeroML = zeros(obj.m, obj.l);
      
      % Diffuse filter
      for iT = 1:fOut.dt
        for iJ = 1:obj.n
          % omega
          if ~isempty(fWeight.y{iT,iJ})
            temp_y = fWeight.y{iT,iJ};
          else
            temp_y = zeroMP;            
          end
          if ~isempty(rWeight.y{iT,iJ}) 
            temp_r0 = fOut.P(:,:,iT) * rWeight.y{iT,iJ};
          else
            temp_r0 = zeroMP;
          end
          if ~isempty(r1Weight.y{iT,iJ})
            temp_r1 = fOut.Pd(:,:,iT) * r1Weight.y{iT,iJ};
          else
            temp_r1 = zeroMP;
          end
          omega{iT,iJ} = temp_y + temp_r0 + temp_r1;
          
          % omegad
          if ~isempty(fWeight.d{iT,iJ})
            temp_yd = fWeight.d{iT,iJ};
          else
            temp_yd = zeroMP;            
          end
          if ~isempty(rWeight.d{iT,iJ}) 
            temp_r0d = fOut.P(:,:,iT) * rWeight.d{iT,iJ};
          else
            temp_r0d = zeroMP;
          end
          if ~isempty(r1Weight.d{iT,iJ})
            temp_r1d = fOut.Pd(:,:,iT) * r1Weight.d{iT,iJ};
          else
            temp_r1d = zeroMP;
          end
          omegad{iT,iJ} = temp_yd + temp_r0d + temp_r1d;
          
          % omegax
          if ~isempty(fWeight.x{iT,iJ})
            temp_yx = fWeight.x{iT,iJ};
          else
            temp_yx = zeroMK;            
          end
          if ~isempty(rWeight.x{iT,iJ}) 
            temp_r0x = fOut.P(:,:,iT) * rWeight.x{iT,iJ};
          else
            temp_r0x = zeroMK;
          end
          if ~isempty(r1Weight.x{iT,iJ})
            temp_r1x = fOut.Pd(:,:,iT) * r1Weight.x{iT,iJ};
          else
            temp_r1x = zeroMK;
          end
          omegax{iT,iJ} = temp_yx + temp_r0x + temp_r1x;
          
          % omegac
          if ~isempty(fWeight.c{iT,iJ})
            temp_yc = fWeight.c{iT,iJ};
          else
            temp_yc = zeroMM;            
          end
          if ~isempty(rWeight.c{iT,iJ}) 
            temp_r0c = fOut.P(:,:,iT) * rWeight.c{iT,iJ};
          else
            temp_r0c = zeroMM;
          end
          if ~isempty(r1Weight.c{iT,iJ})
            temp_r1c = fOut.Pd(:,:,iT) * r1Weight.c{iT,iJ};
          else
            temp_r1c = zeroMM;
          end
          omegac{iT,iJ} = temp_yc + temp_r0c + temp_r1c;
          
          % omegaw
          if ~isempty(fWeight.w{iT,iJ})
            temp_yw = fWeight.w{iT,iJ};
          else
            temp_yw = zeroML;            
          end
          if ~isempty(rWeight.w{iT,iJ}) 
            temp_r0w = fOut.P(:,:,iT) * rWeight.w{iT,iJ};
          else
            temp_r0w = zeroML;
          end
          if ~isempty(r1Weight.w{iT,iJ})
            temp_r1w = fOut.Pd(:,:,iT) * r1Weight.w{iT,iJ};
          else
            temp_r1w = zeroML;
          end
          omegaw{iT,iJ} = temp_yw + temp_r0w + temp_r1w;
        end
        
        % omegaa0
        if ~isempty(fWeight.a0{iT})
          temp_ya0 = fWeight.a0{iT};
        else
          temp_ya0 = zeroMM;
        end
        if ~isempty(rWeight.a0{iT})
          temp_r0a0 = fOut.P(:,:,iT) * rWeight.a0{iT};
        else
          temp_r0a0 = zeroMM;
        end
        if ~isempty(r1Weight.a0{iT})
          temp_r1a0 = fOut.Pd(:,:,iT) * r1Weight.a0{iT};
        else
          temp_r1a0 = zeroMM;
        end
        omegaa0{iT} = temp_ya0 + temp_r0a0 + temp_r1a0;
      end

      % Standard Kalman filter
      for iT = fOut.dt+1:obj.n
        for iJ = 1:obj.n
          if ~isempty(fWeight.y{iT,iJ}) && ~isempty(rWeight.y{iT,iJ})
            omega{iT,iJ} = fWeight.y{iT,iJ} + fOut.P(:,:,iT) * rWeight.y{iT,iJ};
          elseif ~isempty(fWeight.y{iT,iJ}) 
            omega{iT,iJ} = fWeight.y{iT,iJ};
          elseif ~isempty(rWeight.y{iT,iJ})
            omega{iT,iJ} = fOut.P(:,:,iT) * rWeight.y{iT,iJ};            
          end
          
          if ~isempty(fWeight.d{iT,iJ}) && ~isempty(rWeight.d{iT,iJ})
            omegad{iT,iJ} = fWeight.d{iT,iJ} + fOut.P(:,:,iT) * rWeight.d{iT,iJ};
          elseif ~isempty(fWeight.d{iT,iJ}) 
            omegad{iT,iJ} = fWeight.d{iT,iJ};
          elseif ~isempty(rWeight.d{iT,iJ})
            omegad{iT,iJ} = fOut.P(:,:,iT) * rWeight.d{iT,iJ};
          end
          
          if ~isempty(fWeight.x{iT,iJ}) && ~isempty(rWeight.x{iT,iJ})
            omegax{iT,iJ} = fWeight.x{iT,iJ} + fOut.P(:,:,iT) * rWeight.x{iT,iJ};
          elseif ~isempty(fWeight.x{iT,iJ})
            omegax{iT,iJ} = fWeight.x{iT,iJ};
          elseif ~isempty(rWeight.x{iT,iJ})
            omegax{iT,iJ} = fOut.P(:,:,iT) * rWeight.x{iT,iJ};
          end
          
          if ~isempty(fWeight.c{iT,iJ}) && ~isempty(rWeight.c{iT,iJ})
            omegac{iT,iJ} = fWeight.c{iT,iJ} + fOut.P(:,:,iT) * rWeight.c{iT,iJ};
          elseif ~isempty(fWeight.c{iT,iJ}) 
            omegac{iT,iJ} = fWeight.c{iT,iJ};
          elseif ~isempty(rWeight.c{iT,iJ})
            omegac{iT,iJ} = fOut.P(:,:,iT) * rWeight.c{iT,iJ};
          end
                    
          if ~isempty(fWeight.w{iT,iJ}) && ~isempty(rWeight.w{iT,iJ})
            omegaw{iT,iJ} = fWeight.w{iT,iJ} + fOut.P(:,:,iT) * rWeight.w{iT,iJ};
          elseif ~isempty(fWeight.w{iT,iJ}) 
            omegaw{iT,iJ} = fWeight.w{iT,iJ};
          elseif ~isempty(rWeight.w{iT,iJ})
            omegaw{iT,iJ} = fOut.P(:,:,iT) * rWeight.w{iT,iJ};
          end
          
        end
        
        if ~isempty(fWeight.a0{iT}) && ~isempty(rWeight.a0{iT})
          omegaa0{iT} = fWeight.a0{iT} + fOut.P(:,:,iT) * rWeight.a0{iT};
        elseif ~isempty(fWeight.a0{iT})
          omegaa0{iT} = fWeight.a0{iT};
        elseif ~isempty(rWeight.a0{iT})
          omegaa0{iT} = fOut.P(:,:,iT) * rWeight.a0{iT};
        end
      end
      
      sWeights = struct('y', {omega}, 'd', {omegad}, 'x', {omegax}, ...
        'c', {omegac}, 'w', {omegaw}, 'a0', {omegaa0});
    end
    
    function [r, r1] = r_weights(obj, y, x, fOut, fWeight, ssMulti, C)
      % Construct weight functions for r
      %
      % Ordered (state, observation, effectPeriod, contributionPeriod)
      %
      % See "A Practitioner's Guide and Matlab Toolbox for Mixed Frequency State Space 
      % Models", Appendix C.
      
      comp = obj.build_smoother_weight_parts(y, fOut);
      
      % Compute the decomposition of r and r^0
      rComp = struct('Ay', comp.Ay, 'Aa', comp.Aa, ...
        'M', comp.Mdagger, 'Lown', comp.Ldagger, 'Lother', []);
      r = r_weight_recursion(obj, y, x, ssMulti, C, fWeight, rComp, obj.n, []);
      
      % Compute the decomposition of r^1
      r1Comp = struct('Ay', comp.Ay, 'Aa', comp.Aa, ...
        'M', comp.Minfty, 'Lown', comp.Linfty, 'Lother', comp.Lzero);
      r1 = r_weight_recursion(obj, y, x, ssMulti, C, fWeight, r1Comp, fOut.dt, r);
    end
    
    function weights = r_weight_recursion(obj, y, x, ssMulti, C, fWeights, sOut2, T, otherOmega)
      % The recursions for the weights are similar regardless of if its for r,
      % r^0 or r^1. This function generalizes it so that they all can all be
      % performed by changing the inputs. 
      %
      % If this is for r, we need to compute the decomposition of r_1 to r_T (since r^0 is
      % stored at the beginning of r). If this is for r^1, we only need r_1 to r_dt.
      %
      % See "A Practitioner's Guide and Matlab Toolbox for Mixed Frequency State Space 
      % Models", Appendix C.
      
      omegar = cell(T, obj.n);
      omegarc = cell(T, obj.n+1);
      omegard = cell(T, obj.n);
      omegarx = cell(T, obj.n);
      omegarw = cell(T, obj.n+1);
      omegara0 = cell(T, 1);
      
      if isempty(sOut2.Lother)
        sOut2.Lother = zeros(size(sOut2.Lown));
      end
       
      zeroMP = zeros(obj.m, obj.p);
      zeroMK = zeros(obj.m, obj.k);
      zeroMM = zeros(obj.m, obj.m);
      zeroML = zeros(obj.m, obj.l);
      
      eps2 = eps^2;
      
      % Iterate over periods affected
      for iT = T:-1:1
        
        % Iterate over observed data
        for iJ = obj.n:-1:1
          % The effect of future v_t on r_t, recursively.
          % In order for this to work we need to make sure we have done
          % omegar_{t+1,j} before omegar_{t,j}. There shouldn't be any concern
          % over doing the j's in any order here.
          if iT ~= T && ~isempty(omegar{iT+1,iJ})
            forwardEffect = sOut2.Lown(:,:,iT)' * omegar{iT+1,iJ};
          else
            forwardEffect = zeroMP;
          end
          if iT ~= T && ~isempty(omegard{iT+1,iJ})
            forwardEffectd = sOut2.Lown(:,:,iT)' * omegard{iT+1,iJ};
          else
            forwardEffectd = zeroMP;
          end
          if iT ~= T && ~isempty(omegarx{iT+1,iJ})
            forwardEffectx = sOut2.Lown(:,:,iT)' * omegarx{iT+1,iJ};
          else
            forwardEffectx = zeroMK;
          end
          if iT ~= T && ~isempty(omegarc{iT+1,iJ})
            forwardEffectc = sOut2.Lown(:,:,iT)' * omegarc{iT+1,iJ};
          else
            forwardEffectc = zeroMM;
          end
          if iT ~= T && ~isempty(omegarw{iT+1,iJ})
            forwardEffectw = sOut2.Lown(:,:,iT)' * omegarw{iT+1,iJ};
          else
            forwardEffectw = zeroML;
          end
          
          % The effect of y on r^(1) via future r^(0) 
          if ~isempty(otherOmega) && iT ~= obj.n && ~isempty(otherOmega.y{iT+1,iJ})
            forwardEffectOther = sOut2.Lother(:,:,iT)' * otherOmega.y{iT+1,iJ};
          else
            forwardEffectOther = zeroMP;
          end
          if ~isempty(otherOmega) && iT ~= obj.n && ~isempty(otherOmega.d{iT+1,iJ})
            forwardEffectOtherd = sOut2.Lother(:,:,iT)' * otherOmega.d{iT+1,iJ};
          else
            forwardEffectOtherd = zeroMP;
          end
          if ~isempty(otherOmega) && iT ~= obj.n && ~isempty(otherOmega.x{iT+1,iJ})
            forwardEffectOtherx = sOut2.Lother(:,:,iT)' * otherOmega.x{iT+1,iJ};
          else
            forwardEffectOtherx = zeroMK;
          end
          if ~isempty(otherOmega) && iT ~= obj.n && ~isempty(otherOmega.c{iT+1,iJ})
            forwardEffectOtherc = sOut2.Lother(:,:,iT)' * otherOmega.c{iT+1,iJ};
          else
            forwardEffectOtherc = zeroMM;
          end
          if ~isempty(otherOmega) && iT ~= obj.n && ~isempty(otherOmega.w{iT+1,iJ})
            forwardEffectOtherw = sOut2.Lother(:,:,iT)' * otherOmega.w{iT+1,iJ};
          else
            forwardEffectOtherw = zeroML;
          end
          
          % The effect of the data on the filtered state estimate, a_t.
          if iT < iJ || isempty(fWeights.y{iT,iJ})
            filterEffect = zeroMP;
          else
            filterEffect = -sOut2.M(:,:,iT) * sOut2.Aa(:,:,iT) * fWeights.y{iT,iJ};
          end
          if iT < iJ || isempty(fWeights.d{iT,iJ})
            filterEffectd = zeroMP;
          else
            filterEffectd = -sOut2.M(:,:,iT) * sOut2.Aa(:,:,iT) * fWeights.d{iT,iJ};
          end
          if iT < iJ || isempty(fWeights.x{iT,iJ})
            filterEffectx = zeroMK;
          else
            filterEffectx = -sOut2.M(:,:,iT) * sOut2.Aa(:,:,iT) * fWeights.x{iT,iJ};
          end
          if iT < iJ || isempty(fWeights.c{iT,iJ})
            filterEffectc = zeroMM;
          else
            filterEffectc = -sOut2.M(:,:,iT) * sOut2.Aa(:,:,iT) * fWeights.c{iT,iJ};
          end
          if iT < iJ || isempty(fWeights.w{iT,iJ})
            filterEffectw = zeroML;
          else
            filterEffectw = -sOut2.M(:,:,iT) * sOut2.Aa(:,:,iT) * fWeights.w{iT,iJ};
          end

          % The effect of the data on the error term, v_t.
          % Note that there's no c here because it was already included in a_t.
          contempEffect = zeroMP;
          contempEffectd = zeroMP;
          contempEffectx = zeroMK;
          if iT == iJ
            validY = ~isnan(y(:,iJ));
            contempEffect(:,validY) = sOut2.M(:,:,iT) * sOut2.Ay(:,validY,iT) ...
              / C(validY,validY,obj.tau.H(iT)) * diag(y(validY,iJ));
            contempEffectd(:,validY) = -sOut2.M(:,:,iT) * sOut2.Ay(:,validY,iT) ...
              / C(validY,validY,obj.tau.H(iT)) * diag(ssMulti.d(validY,ssMulti.tau.d(iJ)));
            contempEffectx(:,:) = -sOut2.M(:,:,iT) * sOut2.Ay(:,validY,iT) ...
              / C(validY,validY,obj.tau.H(iT)) * ...
              ssMulti.beta(validY,:,ssMulti.tau.beta(iJ)) * diag(x(:,iT));
          end
          
          % No filter effect since the filter does nothing when j == t.
          omegar_temp = forwardEffect + forwardEffectOther + filterEffect + contempEffect;
          if any(any(abs(omegar_temp) > eps2))
            omegar{iT,iJ} = omegar_temp;
          end
          omegard_temp = forwardEffectd + forwardEffectOtherd + filterEffectd + contempEffectd;
          if any(any(abs(omegard_temp) > eps2))
            omegard{iT,iJ} = omegard_temp;
          end
          omegarx_temp = forwardEffectx + forwardEffectOtherx + filterEffectx + contempEffectx;
          if any(any(abs(omegarx_temp) > eps2))
            omegarx{iT,iJ} = omegarx_temp;
          end
          omegarc_temp = forwardEffectc + forwardEffectOtherc + filterEffectc;
          if any(any(abs(omegarc_temp) > eps2))
            omegarc{iT,iJ} = omegarc_temp;
          end
          omegarw_temp = forwardEffectw + forwardEffectOtherw + filterEffectw;
          if any(any(abs(omegarw_temp) > eps2))
            omegarw{iT,iJ} = omegarw_temp;
          end
        end
        
        % Weight for a0
        if iT ~= T && ~isempty(omegara0{iT+1})
          forwardEffecta0 = sOut2.Lown(:,:,iT)' * omegara0{iT+1};
        else
          forwardEffecta0 = zeroMM;
        end
        if ~isempty(otherOmega) && iT ~= obj.n && ~isempty(otherOmega.a0{iT+1})
          forwardEffectOthera0 = sOut2.Lother(:,:,iT)' * otherOmega.a0{iT+1};
        else
          forwardEffectOthera0 = zeroMM;
        end
        if ~isempty(fWeights.a0{iT})
          filterEffecta0 = -sOut2.M(:,:,iT) * sOut2.Aa(:,:,iT) * fWeights.a0{iT};
        else 
          filterEffecta0 = zeroMM;
        end
        omegara0_temp = forwardEffecta0 + forwardEffectOthera0 + filterEffecta0;
        if any(any(abs(omegara0_temp) > eps2))
          omegara0{iT} = omegara0_temp;
        end
      end
      
      weights = struct('y', {omegar}, 'd', {omegard}, 'x', {omegarx}, ...
        'c', {omegarc}, 'w', {omegarw}, 'a0', {omegara0});
    end
    
    function components = build_smoother_weight_parts(obj, y, fOut)
      % Build the quantities we need to build the smoother weights
      % A^a, A^y, L^\dagger & M^\dagger
      % 
      % Because we would have to set the last value of all of the * quantities in the
      % diffuse smoother with the corresponding values from the non-diffuse smoother, the
      % * values will stored in the non-diffuse smoother value arrays. 
      %
      % See "A Practitioner's Guide and Matlab Toolbox for Mixed Frequency State Space 
      % Models", Appendix C.
      
      Im = eye(obj.m);
      Ip = eye(obj.p);
      
      % A^a*, A^a\infty, M^*, \tilde{M}^\infty, L^* and L^\infty
      Aa = zeros(obj.p, obj.m, obj.n);
      Ldagger = zeros(obj.m, obj.m, obj.n);
      Mdagger = zeros(obj.m, obj.p, obj.n);

      Aainfty = zeros(obj.p, obj.m, fOut.dt);
      Linfty = zeros(obj.m, obj.m, fOut.dt);
      MinftyTilde = zeros(obj.m, obj.p, fOut.dt);
      
      for iT = 1:fOut.dt
        SstarProd = Im;
        SinftyProd = Im;
        
        for jP = 1:obj.p
          if isnan(y(jP,iT))
            continue
          end
          
          Z = obj.Z(jP,:,obj.tau.Z(iT));
          Aa(jP,:,iT) = Z * SstarProd;
          Aainfty(jP,:,iT) = Z * SinftyProd;
          
          if fOut.Fd(jP,iT) ~= 0
            Sinfty = (Im - fOut.Kd(:,jP,iT) * Z);
            Sstar = Sinfty;
            MinftyTilde(:,jP,iT) = SstarProd' * Z' / fOut.Fd(jP,iT);
          else
            Sstar = (Im - fOut.K(:,jP,iT) * Z);
            Sinfty = Im;
            Mdagger(:,jP,iT) = SstarProd' * Z' / fOut.F(jP,iT);
          end
          
          SstarProd = Sstar * SstarProd;
          SinftyProd = Sinfty * SinftyProd;
        end
        Ldagger(:,:,iT) = obj.T(:,:,obj.tau.T(iT+1)) * SstarProd;
        Linfty(:,:,iT) = obj.T(:,:,obj.tau.T(iT+1)) * SinftyProd;
      end
      
      % A^a, Mdagger and Ldagger
      for iT = fOut.dt+1:obj.n
        Lprod = Im;
        for jP = 1:obj.p
          if isnan(y(jP,iT))
            continue
          end
          
          Z = obj.Z(jP,:,obj.tau.Z(iT));
          K = fOut.K(:,jP,iT);
          
          Aa(jP,:,iT) = Z * Lprod;
          Lprod = (Im - K * Z) * Lprod;
        end
        Ldagger(:,:,iT) = obj.T(:,:,obj.tau.T(iT+1)) * Lprod;
        FinvDiag = zeros(obj.p, obj.p);
        FinvDiag(~isnan(y(:,iT)), ~isnan(y(:,iT))) = diag(1./fOut.F(~isnan(y(:,iT)),iT));
        Mdagger(:,:,iT) = Aa(:,:,iT)' * FinvDiag;
      end
      
      % L^(0) and M^\infty (M^(0) as a component)
      Lzero = zeros(obj.m, obj.m, fOut.dt);
      Minfty = zeros(obj.m, obj.p, fOut.dt);
      for iT = 1:fOut.dt
        Mzero = zeros(obj.m, obj.p);
        Lzerosum = zeros(obj.m, obj.m);
        
        % Prebuild S^* product that runs in reverse
        SstarProd = zeros(obj.m, obj.m, obj.p+1);
        SstarProd(:,:,end) = Im;
        % We only need to do p-1 multiplications here since for the first observation we
        % need the product of the i=2 through p S* matricies.
        for iP = obj.p:-1:1
          Zi = obj.Z(iP, :, obj.tau.Z(iT));
          if fOut.Fd(iP,iT) ~= 0
            Sstar = (Im - fOut.Kd(:,iP,iT) * Zi);
          else
            Sstar = (Im - fOut.K(:,iP,iT) * Zi);
          end
          SstarProd(:,:,iP) = Sstar' * SstarProd(:,:,iP+1);          
        end
        
        % Build M^(0) and L^(0) (and the product of S^\infty as we go)
        SinftyProd = Im;
        for iP = 1:obj.p
          if fOut.Fd(iP,iT) ~= 0
            Zi = obj.Z(iP, :, obj.tau.Z(iT));
            S0ti = (fOut.Kd(:,iP,iT) - fOut.K(:,iP,iT)) * Zi * fOut.F(iP,iT) / fOut.Fd(iP,iT);
            Sinfty = (Im - fOut.Kd(:,iP,iT) * Zi);
          else
            S0ti = zeros(obj.m, obj.m);
            Sinfty = Im;
          end
          
          % Compute M^(0) and the complicated part of L^(0)
          Mzero = Mzero + SinftyProd * S0ti' * obj.build_M0ti(y, fOut, iT, iP);
          Lzerosum = Lzerosum + SinftyProd * S0ti' * SstarProd(:,:,iP+1);
          SinftyProd = SinftyProd * Sinfty';
        end
        Minfty(:,:,iT) = MinftyTilde(:,:,iT) + Mzero;
        Lzero(:,:,iT) = obj.T(:,:,obj.tau.T(iT+1)) * Lzerosum';
      end
      
      % A^y
      Ay = zeros(obj.p, obj.p, fOut.dt);
      for iT = 1:obj.n
        AyTilde = zeros(obj.p, obj.p);

        % Even for the diffuse case, we're going to call everything L
        for jP = 1:obj.p-1
          % Looping over columns of Ay
          if isnan(y(jP,iT))
            continue
          end
          
          % We're building AyTilde element-by-element here. The i's index the
          % row (which v_ti is affected), the j's index the column (which y_tj is used).
          Lprod = Im;
          for iP = jP+1:obj.p
            % Looping over rows of Ay
            Zi = obj.Z(iP, :, obj.tau.Z(iT));
            
            if iT <= fOut.dt && fOut.Fd(jP,iT) ~= 0
              AyTilde(iP, jP) = Zi * Lprod * fOut.Kd(:,jP,iT);
            else
              AyTilde(iP, jP) = Zi * Lprod * fOut.K(:,jP,iT);
            end
            
            if iT <= fOut.dt && fOut.Fd(iP,iT) ~= 0
              Sk = (Im - fOut.Kd(:,iP,iT) * Zi);
            else
              Sk = (Im - fOut.K(:,iP,iT) * Zi);
            end
            Lprod = Sk * Lprod;
          end
          
        end
        
        Ay(:,:,iT) = Ip - AyTilde;        
      end
      
      components = struct('Ay', Ay, 'Aa', Aa, 'Aainfty', Aainfty,...
        'Ldagger', Ldagger, 'Mdagger', Mdagger, ...
        'Linfty', Linfty, 'Minfty', Minfty, 'Lzero', Lzero);
    end
    
    function M0ti = build_M0ti(obj, y, fOut, iT, iP)
      % Build the M_{t,i}^{(0)} matrix for use in the diffuse smoother recursion
      %
      % See "A Practitioner's Guide and Matlab Toolbox for Mixed Frequency State Space 
      % Models", Appendix C.
      
      M0ti = zeros(obj.m, obj.p);
      Im = eye(obj.m);
      SstarProdJ = Im;
      for jP = iP+1:obj.p
        if isnan(y(jP,iT))
          continue
        end
        
        Zj = obj.Z(jP, :, obj.tau.Z(iT));
        if fOut.Fd(jP,iT) ~= 0
          Sstar = (Im - fOut.Kd(:,jP,iT) * Zj);
          Mstarti = zeros(obj.m, 1);
        else
          Sstar = (Im - fOut.K(:,jP,iT) * Zj);
          Mstarti = Zj' / fOut.F(jP,iT);
        end
        
        M0ti(:,jP) = SstarProdJ * Mstarti;
        SstarProdJ = SstarProdJ * Sstar';
      end
    end
    
    function Ldagger = build_Ldagger(obj, y, fOut)
      % Build L^\dagger needed for the error variances.
      %
      % See "A Practitioner's Guide and Matlab Toolbox for Mixed Frequency State Space 
      % Models", Appendix C.
      
      Im = eye(obj.m);
      Ldagger = zeros(obj.m, obj.m, obj.n); 
      
      for iT = 1:fOut.dt
        SstarProd = Im;

        for jP = 1:obj.p
          if isnan(y(jP,iT))
            continue
          end
          Z = obj.Z(jP,:,obj.tau.Z(iT));
          if fOut.Fd(jP,iT) ~= 0
            Sinfty = (Im - fOut.Kd(:,jP,iT) * Z);
            Sstar = Sinfty;
          else
            Sstar = (Im - fOut.K(:,jP,iT) * Z);
          end
          SstarProd = Sstar * SstarProd;
        end
        Ldagger(:,:,iT) = obj.T(:,:,obj.tau.T(iT+1)) * SstarProd;
      end
      
      for iT = fOut.dt+1:obj.n
        Lprod = Im;
        for jP = 1:obj.p
          if isnan(y(jP,iT))
            continue
          end
          Z = obj.Z(jP,:,obj.tau.Z(iT));
          K = fOut.K(:,jP,iT);
          Lprod = (Im - K * Z) * Lprod;
        end
        Ldagger(:,:,iT) = obj.T(:,:,obj.tau.T(iT+1)) * Lprod;
      end
    end
    
    %% Simulation smoother helpers
    function [yPlus, alphaPlus] = generateSimulationData(obj)
      % Used in simulation smoother. Returns generated data yPlus and alphaPlus.
      
      % Note that from Jarocinski (2015), we need to draw a0 from N(0, P0). %
      %
      % I believe that we don't want to do anything like x+ because we have no generative
      % model for it. We only have a model for E(y | alpha, x). 
      % 
      % We also ignore the effects of c, d, beta*x and gamma*w since we're using 
      % Jarocinski's simpler simulation smoother. 
      
      % Generate a0tilde
      if isempty(obj.a0) || isempty(obj.Q0)
        obj = obj.setDefaultInitial();
      end
      
      Hsqrt = zeros(size(obj.H));
      for iH = 1:size(Hsqrt,3)
        nonZero = diag(obj.H(:,:,iH)) ~= 0;
        Hsqrt(nonZero,nonZero,iH) = chol(obj.H(nonZero,nonZero,iH));
      end
      
      Qsqrt = zeros(size(obj.Q));
      for iQ = 1:size(Qsqrt,3)
        nonZero = diag(obj.Q(:,:,iH)) ~= 0;
        Qsqrt(nonZero,nonZero,iQ) = chol(obj.Q(nonZero,nonZero,iQ));
      end
      
      % alpha0 ~ N(0, P0). If any elements use the diffuse initialization, it doesn't
      % matter how we initialize them, so just replace those elements of P0 with 0 here.
      P0_temp = obj.P0;
      P0_temp(isinf(P0_temp)) = 0;
      nonZero = diag(P0_temp) ~= 0;
      P0_temp_chol = zeros(size(P0_temp));
      P0_temp_chol(nonZero,nonZero) = chol(P0_temp(nonZero,nonZero));
      a0Draw = P0_temp_chol * randn(obj.m, 1);
      
      % Generate state 
      eta = nan(obj.g, obj.n);
      rawEta = randn(obj.g, obj.n);

      alphaPlus = nan(obj.m, obj.n);
      alphaPlus(:,1) = obj.T(:,:,obj.tau.T(1)) * a0Draw;
      for iT = 2:obj.n
        eta(:,iT) = Qsqrt(:,:,obj.tau.Q(iT)) * rawEta(:, iT);
        alphaPlus(:,iT) = obj.T(:,:,obj.tau.T(iT)) * alphaPlus(:,iT-1) + ...
          obj.R(:,:,obj.tau.R(iT)) * eta(:,iT);
      end
      
      % Generate observed data
      epsilon = nan(obj.p, obj.n);
      rawEpsilon = randn(obj.p, obj.n);

      yPlus = nan(obj.p, obj.n);
      for iT = 1:obj.n
        epsilon(:,iT) = Hsqrt(:,:,obj.tau.H(iT)) * rawEpsilon(:,iT);
        yPlus(:,iT) = obj.Z(:,:,obj.tau.Z(iT)) * alphaPlus(:,iT) + epsilon(:,iT);
      end
    end
    
  end
  
  methods (Static)
    %% General Utility Methods
    function setSS = setAllParameters(ss, value)
      % Create a StateSpace with all paramters set to value
      
      % Set all parameter values equal to the scalar provided
      for iP = 1:length(ss.systemParam)
        ss.(ss.systemParam{iP}) = value * ones(size(ss.(ss.systemParam{iP})));
      end
      
      % Needs to be a StateSpace since we don't want a ThetaMap
      setSS = StateSpace(ss.Z(:,:,1), ss.H(:,:,1), ss.T(:,:,1), ss.Q(:,:,1), ...
        'd', ss.d(:,1), 'beta', ss.beta(:,:,1), ...
        'c', ss.c(:,1), 'gamma', ss.gamma(:,:,1), 'R', ss.R(:,:,1));
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
      setSS.a0 = a0value;
      
      % Do I want to account for diffuse states?
      % A: No, we're assuming A0 and R0 don't count here.
      if ~isempty(ss.P0)
        if value == Inf || value == -Inf
          % We don't allow the size of A0 and R0 to change so put value in Q0
          setSS.P0 = ss.P0;
          setSS.Q0 = repmat(value, size(ss.Q0));
        else
          diffuseP0 = ss.A0 * ss.A0';
          diffuseP0(diffuseP0 ~= 0) = Inf;
          
          P0value = ss.R0 * repmat(value, size(ss.Q0)) * ss.R0' + diffuseP0;
          setSS.P0 = P0value;
        end
      end
    end
  end
end
