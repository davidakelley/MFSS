classdef StateSpace < AbstractStateSpace
  % State estimation of models with known parameters 
  %
  % See documentation in the function reference. 
 
  % David Kelley, 2016-2017
 
  % Todo:
  %     - Add filter/smoother weight decompositions
  %     - Add IRF/historical decompositions
  
  properties (Hidden)
    % Indicator for use of analytic gradient
    useAnalyticGrad = true;
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
    end
    
    function P0 = get.P0(obj)
      P0 = obj.P0Private;
    end
    
    function obj = set.P0(obj, newP0)
      obj.P0Private = newP0;
    end
  end
  
  methods
    %% Constructor
    function obj = StateSpace(Z, d, H, T, c, R, Q)
      % StateSpace constructor
      %
      % Args:
      %     Z (matrix): Observation loadings
      %     d (matrix): Observation constants
      %     H (matrix): Observation error covariances
      %     T (matrix): State transition coefficients
      %     c (matrix): State constants
      %     R (matrix): Error selection 
      %     Q (matrix): State error covariances
      % 
      % Returns: 
      %     obj (StateSpace): a StateSpace object
      
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
    end
  end
  
  methods
    %% State estimation methods 
    function [a, logli, filterOut] = filter(obj, y)
      % Estimate the filtered state
      %
      % Arguments:
      %     y (double): observed data (p x T)
      %
      % Returns:
      %     a (double) : filtered state (m x [T+1])
      %     logli (double) : log-likelihood
      %     filterOut (struct) : structure of additional intermeidate quantites
      %
      % See Also:
      %     smooth
      
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
    
    function [logli, gradient, fOut] = gradient(obj, y, tm, theta)
      % Returns the likelihood and the change in the likelihood given the
      % change in any system parameters that are currently set to nans.

      % Handle inputs
      assert(isa(tm, 'ThetaMap'), 'tm must be a ThetaMap.');
      if nargin < 4
        theta = tm.system2theta(obj);
      end
      assert(all(size(theta) == [tm.nTheta 1]), ...
        'theta must be a nTheta x 1 vector.');

      if obj.useAnalyticGrad
        % Transform parameters to allow for univariate treatment
        ssMulti = obj;
        [obj, yUni, factorC, oldTau] = obj.prepareFilter(y);

        % Generate parameter gradient structure
        GMulti = tm.parameterGradients(theta);
        [GMulti.a1, GMulti.P1] = tm.initialValuesGradients(obj, GMulti, theta); 

        % Transform to univariate filter gradients
        [GUni, GY] = obj.factorGradient(y, GMulti, ssMulti, factorC, oldTau);

        % Run filter with extra outputs, comput gradient
        [~, logli, fOut, ftiOut] = obj.filter_detail_m(yUni);      
        gradient = obj.gradient_filter_m(y, GUni, GY, fOut, ftiOut);
      else
        if obj.useParallel
          [logli, gradient, fOut] = obj.gradientFiniteDifferences_parallel(y, tm, theta);
        else
          [logli, gradient, fOut] = obj.gradientFiniteDifferences(y, tm, theta);
        end
      end
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
    function obj = setDefaultInitial(obj)
      % Set default a0 and P0.
      % Run before filter/smoother after a0 & P0 inputs have been processed
      
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
  
  methods
    %% Filter/smoother Helper Methods
    function [obj, y, factorC, oldTau] = prepareFilter(obj, y)
      % Make sure data matches observation dimensions
      obj.validateKFilter();
      obj = obj.checkSample(y);
      
      % Set initial values
      obj = obj.setDefaultInitial();

      % Handle multivariate series
      [obj, y, factorC, oldTau] = obj.factorMultivariate(y);
    end
    
    function validateKFilter(obj)
      % Validate parameters
      
      obj.validateStateSpace();
      
      % Make sure all of the parameters are known (non-nan)
      assert(~any(cellfun(@(x) any(any(any(isnan(x)))), obj.parameters)), ...
        ['All parameter values must be known. To estimate unknown '...
        'parameters, see StateSpaceEstimation']);
    end
    
    function [ssUni, yUni, factorC, oldTau] = factorMultivariate(obj, y)
      % Compute new Z and H matricies so the univariate treatment can be applied
      %
      % Args:
      %     y : observed data
      %
      % Returns: 
      %     ssUni : StateSpace model with diagonal measurement error
      %     yUni : data modified for ssUni 
      %     factorC : diagonalizing matrix(s) for original H matrix
      %     oldTau : structure of mapping from original tau to new tau
      %
      % Todo: What happens when H and Z don't match periods? 
      
      %{
      % If H is already diagonal, do nothing
      if arrayfun(@(x) isdiag(obj.H(:,:,x)), 1:size(obj.H,3))
        ssUni = obj;
        yUni = y;
        factorC = eye(obj.p);
        
        oldTau = struct('H', 1:max(obj.tau.H), 'Z', 1:max(obj.tau.Z), ...
          'd', 1:max(obj.tau.d), ...
          'correspondingNewHOldZ', correspondingNewHOldZ, ...
          'correspondingNewHOldd', correspondingNewHOldd, ...        
          'obsPattern', obsPatternH);
        return
      end
      %}
      
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
      
      [~, ~, ~, T, c, R, Q] = obj.getInputParameters();
      
      ssUni = StateSpace(newZ, newd, newH, T, c, R, Q);
      
      % Set initial values
      ssUni.a0 = obj.a0;
      ssUni.P0 = obj.P0;
      
      oldTau = struct('H', oldTauH, 'Z', oldTauZ, 'd', oldTaud, ...
        'correspondingNewHOldZ', correspondingNewHOldZ, ...
        'correspondingNewHOldd', correspondingNewHOldd, ...        
        'obsPattern', obsPatternH);
    end
    
    function [Guni, GY] = factorGradient(obj, y, GMulti, ssMulti, factorC, oldTau)
      % factorGradient transforms the gradient of the multivariate model to the
      % univarite model given the univariate parameters and the factorzation
      % matrix. 
      
      if size(GMulti.H, 3) ~= 1
        warning('TVP gradient not yet tested.');
      end

      Guni = GMulti;
      
      nHslices = size(obj.H, 3);
      factorCinv = zeros(obj.p, obj.p, nHslices);
      kronCinv = zeros(obj.p^2, obj.p^2, nHslices);
      Guni.H = zeros([size(GMulti.H, 1), size(GMulti.H, 2), nHslices]);
      GC = zeros([size(GMulti.H) nHslices]);
      for iH = 1:nHslices
        ind = logical(oldTau.obsPattern(iH,:));
        kronInd = logical(kron(ind,ind));
        
        % Get the indexes of the observed GC and GHstar
        indexMatObsH = reshape(1:sum(ind)^2, [sum(ind), sum(ind)]);
        diagObsCols = diag(indexMatObsH);
        lowerObsTril = tril(indexMatObsH, -1);
        lowerDiagObsCols = lowerObsTril(lowerObsTril ~= 0);
                
        if ~any(ind)
          continue
        end

        % Find GC and GHstar
        Kp = AbstractSystem.genCommutation(sum(ind));
        Np = eye(sum(ind)^2) + Kp;
        Cmult = kron(obj.H(ind,ind,iH) * factorC(ind,ind,iH)', eye(sum(ind))) * Np;
        Hstarmult = kron(factorC(ind,ind,iH)', factorC(ind,ind,iH)');

        % Get the indexes corresponding to the full GC and GHstar
        indexMatH = reshape(1:obj.p^2, [obj.p obj.p]);
        indexMatH(~ind, :) = 0; indexMatH(:, ~ind) = 0;
        diagCols = diag(indexMatH);
        diagCols(diagCols == 0) = [];

        lowerTril = tril(indexMatH, -1);
        lowerDiagCols = lowerTril(lowerTril ~= 0);

        Wtilde = [Cmult(lowerDiagObsCols,:); Hstarmult(diagObsCols,:)];
        Gtilde = GMulti.H(:,indexMatH(indexMatH ~= 0),oldTau.H(iH)) * pinv(Wtilde);

        GC(:, lowerDiagCols, iH) = Gtilde(:,1:length(lowerDiagCols));
        Guni.H(:,diagCols,iH) = Gtilde(:,length(lowerDiagCols)+1:end);

        % Check
        reconstructed = GC(:,kronInd,iH) * Cmult + Guni.H(:,kronInd,iH) * Hstarmult;
        indH = indexMatH(indexMatH ~= 0);
        original = GMulti.H(:,indH,oldTau.H(iH));
        assert(max(max(abs(reconstructed - original))) < 1e-10, 'Development error');

        % Get objects needed for Z, d and Y.
        factorCinv(ind,ind,iH) = inv(factorC(ind,ind,iH));
        kronCinv(kronInd,kronInd,iH) = kron(factorCinv(ind,ind,iH), factorCinv(ind,ind,iH)');
      end
      
      nZslices = size(obj.Z, 3);
      Guni.Z = zeros([size(GMulti.Z, 1) size(GMulti.Z, 2) nZslices]);
      for iZ = 1:nZslices
        indexMatZ = reshape(1:obj.p*obj.m, [obj.p, obj.m]);
        indexMatH = reshape(1:obj.p^2, [obj.p, obj.p]);
        iH = oldTau.correspondingNewHOldZ(iZ);
        iOldZ = oldTau.Z(iZ);
        ind = logical(oldTau.obsPattern(iH,:));
        indexMatZ(~ind, :) = 0;
        indexMatH(~ind, :) = 0; indexMatH(:, ~ind) = 0;
        indZ = indexMatZ(indexMatZ ~= 0);
        indH = indexMatH(indexMatH ~= 0);
        
        Guni.Z(:,indZ,iZ) = -GC(:,indH,iH) * kronCinv(indH,indH,iH) ...
          * kron(ssMulti.Z(ind,:,iOldZ), eye(sum(ind))) ...
          + GMulti.Z(:,indZ,iOldZ) * kron(eye(obj.m), factorCinv(ind,ind,iH)');
      end
      
      ndslices = size(obj.d, 2);
      Guni.d = zeros([size(GMulti.d, 1) size(GMulti.d, 2) ndslices]);
      for id = 1:ndslices
        indexMatd = (1:obj.p)';
        indexMatH = reshape(1:obj.p^2, [obj.p, obj.p]);
        iH = oldTau.correspondingNewHOldZ(iZ);
        iOldd = oldTau.d(id);
        ind = logical(oldTau.obsPattern(iH,:));
        indexMatd(~ind) = 0;
        indexMatH(~ind, :) = 0; indexMatH(:, ~ind) = 0;
        indd = indexMatd(indexMatd ~= 0);
        indH = indexMatH(indexMatH ~= 0);
        
        Guni.d(:,indd,id) = -GC(:,indH,iH) * kronCinv(indH,indH,iH) ...
          * kron(ssMulti.d(ind,iOldd), eye(sum(ind))) ...
          + GMulti.d(:,indd,iOldd) / factorC(ind,ind,iH)';
      end
      
      GY = zeros(size(GMulti.H, 1), obj.n, obj.p);
      for iT = 1:obj.n
        iH = obj.tau.H(iT);
        ind = ~isnan(y(:,iT));
        kronInd = logical(kron(ind,ind));
        GY(:,iT,ind) = -GC(:,kronInd,iH) * kronCinv(kronInd,kronInd,iH) * kron(y(ind,iT), eye(sum(ind))); 
      end
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
  
    function [ll, grad, fOut] = gradientFiniteDifferences(obj, y, tm, theta)
      % Compute numeric gradient using central differences
      [~, ll, fOut] = obj.filter(y);
      
      nTheta = tm.nTheta;
      grad = nan(nTheta, 1);
      
      stepSize = 0.5 * obj.delta;
      for iTheta = 1:nTheta
        try
          thetaDown = theta - [zeros(iTheta-1,1); stepSize; zeros(nTheta-iTheta,1)];
          ssDown = tm.theta2system(thetaDown);
          [~, llDown] = ssDown.filter(y);
          
          thetaUp = theta + [zeros(iTheta-1,1); stepSize; zeros(nTheta-iTheta,1)];
          ssUp = tm.theta2system(thetaUp);
          [~, llUp] = ssUp.filter(y);
          
          if obj.numericGradPrec == 1
            grad(iTheta) = (llUp - llDown) ./ (2 * stepSize);
          else
            thetaDown2 = theta - [zeros(iTheta-1,1); 2 * stepSize; zeros(nTheta-iTheta,1)];
            ssDown2 = tm.theta2system(thetaDown2);
            [~, llDown2] = ssDown2.filter(y);
            
            thetaUp2 = theta + [zeros(iTheta-1,1); 2 * stepSize; zeros(nTheta-iTheta,1)];
            ssUp2 = tm.theta2system(thetaUp2);
            [~, llUp2] = ssUp2.filter(y);
            
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
    
    function [ll, grad, fOut] = gradientFiniteDifferences_parallel(obj, y, tm, theta)
      % Compute numeric gradient using central differences
      [~, ll, fOut] = obj.filter(y);
      
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
        ssDown = tm.theta2system(thetaDown);
        [~, llDown] = ssDown.filter(y);
        
        thetaUp = theta + [zeros(iTheta-1,1); stepSize; zeros(nTheta-iTheta,1)];
        ssUp = tm.theta2system(thetaUp);
        [~, llUp] = ssUp.filter(y);
        
        grad(iTheta) = (llUp - llDown) ./ (2 * stepSize);
      end
      
      % Handle bad evaluations
      grad(imag(grad) ~= 0 | isnan(grad)) = -Inf;
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
  
  methods (Hidden)
    %% Filter/smoother/gradient/decomposition mathematical methods
    function [a, logli, filterOut] = filter_m(obj, y)
      % Filter using exact initial conditions
      %
      % Note that the quantities v, F and K are those that come from the
      % univariate filter and cannot be transfomed via the Cholesky
      % factorization to get the quanties from the multivariate filter. 
      %
      % See "Fast Filtering and Smoothing for Multivariate State Space Models",
      % Koopman & Durbin (2000) and Durbin & Koopman, sec. 7.2.5.
              
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
          v(iP,iT) = y(iP, iT) - Zjj * ati - obj.d(iP,obj.tau.d(iT));
          
          Fd(iP,iT) = Zjj * Pdti * Zjj';
          Fstar(iP,iT) = Zjj * Pstarti * Zjj' + obj.H(iP,iP,obj.tau.H(iT));
          
          Kd(:,iP,iT) = Pdti * Zjj' ./ Fd(iP,iT);
          Kstar(:,iP,iT) = Pstarti * Zjj' ./ Fstar(iP,iT);
          
          if Fd(iP,iT) ~= 0
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
        a(:,iT+1) = Tii * ati + obj.c(:,obj.tau.c(iT+1));
        
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
          
          v(iP,iT) = y(iP,iT) - Zjj * ati - obj.d(iP,obj.tau.d(iT));
          
          F(iP,iT) = Zjj * Pti * Zjj' + obj.H(iP,iP,obj.tau.H(iT));
          K(:,iP,iT) = Pti * Zjj' / F(iP,iT);
          
          LogL(iP,iT) = (log(F(iP,iT)) + (v(iP,iT)^2) / F(iP,iT));
          
          ati = ati + K(:,iP,iT) * v(iP,iT);
          Pti = Pti - K(:,iP,iT) * F(iP,iT) * K(:,iP,iT)';
        end
        
        Tii = obj.T(:,:,obj.tau.T(iT+1));
        
        a(:,iT+1) = Tii * ati + obj.c(:,obj.tau.c(iT+1));
        P(:,:,iT+1) = Tii * Pti * Tii' + ...
          obj.R(:,:,obj.tau.R(iT+1)) * obj.Q(:,:,obj.tau.Q(iT+1)) * obj.R(:,:,obj.tau.R(iT+1))';
      end
      
      % Consider changing to 
      %   sum(sum(F(:,d+1:end)~=0)) + sum(sum(Fstar(:,1:d)~=0))
      logli = -(0.5 * sum(sum(isfinite(y)))) * log(2 * pi) - 0.5 * sum(sum(LogL));
      
      filterOut = struct('a', a, 'P', P, 'Pd', Pd, 'v', v, 'F', F, 'Fd', Fd, ...
        'K', K, 'Kd', Kd, 'dt', dt);
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
          if isnan(y(iP,iT))
            ati(:,iT,iP+1) = ati(:,iT,iP);
            Pdti(:,:,iT,iP+1) = Pdti(:,:,iT,iP);
            Pstarti(:,:,iT,iP+1) = Pstarti(:,:,iT,iP);
            continue
          end
          
          Zjj = obj.Z(iP,:,obj.tau.Z(iT));
          v(iP,iT) = y(iP, iT) - Zjj * ati(:,iT,iP) - obj.d(iP,obj.tau.d(iT));
          
          Fd(iP,iT) = Zjj * Pdti(:,:,iT,iP) * Zjj';
          Fstar(iP,iT) = Zjj * Pstarti(:,:,iT,iP) * Zjj' + obj.H(iP,iP,obj.tau.H(iT));
          
          Kd(:,iP,iT) = Pdti(:,:,iT,iP) * Zjj' / Fd(iP,iT);
          Kstar(:,iP,iT) = Pstarti(:,:,iT,iP) * Zjj' / Fstar(iP,iT);
          
          if Fd(iP,iT) < 0 && abs(Fd(iP,iT)) > 1e-14
            error('Negative forecast variance.');
          elseif Fd(iP,iT) < 0 
            Fd(iP,iT) = 0;
          end
          
          if Fd(iP,iT) ~= 0
            % F diffuse nonsingular
            ati(:,iT,iP+1) = ati(:,iT,iP) + Kd(:,iP,iT) * v(iP,iT);
            
            Pstarti(:,:,iT,iP+1) = Pstarti(:,:,iT,iP) - ...
              (Kstar(:,iP,iT) * Kd(:,iP,iT)' + Kd(:,iP,iT) * Kstar(:,iP,iT)' ...
              - Kd(:,iP,iT) * Kd(:,iP,iT)') .* Fstar(iP,iT);
            
            Pdti(:,:,iT,iP+1) = Pdti(:,:,iT,iP) - Kd(:,iP,iT) .* Kd(:,iP,iT)' .* Fd(iP,iT);
            
            LogL(iP,iT) = log(Fd(iP,iT));
          else
            % F diffuse = 0
            ati(:,iT,iP+1) = ati(:,iT,iP) + Kstar(:,iP,iT) * v(iP,iT);
            
            Pstarti(:,:,iT,iP+1) = Pstarti(:,:,iT,iP) - ...
              Kstar(:,iP,iT) * Kstar(:,iP,iT)' .* Fstar(iP,iT);
            
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
        ati(:,iT,1) = a(:,iT);
        Pti(:,:,iT,1) = P(:,:,iT);
        for iP = 1:obj.p
          if isnan(y(iP,iT))
            ati(:,iT,iP+1) = ati(:,iT,iP);
            Pti(:,:,iT,iP+1) = Pti(:,:,iT,iP);
            continue
          end
          Zjj = obj.Z(iP,:,obj.tau.Z(iT));
          
          v(iP,iT) = y(iP,iT) - Zjj * ati(:,iT,iP) - obj.d(iP,obj.tau.d(iT));
          
          F(iP,iT) = Zjj * Pti(:,:,iT,iP) * Zjj' + obj.H(iP,iP,obj.tau.H(iT));
          K(:,iP,iT) = Pti(:,:,iT,iP) * Zjj' / F(iP,iT);
          
          LogL(iP,iT) = (log(F(iP,iT)) + (v(iP,iT)^2) / F(iP,iT));
          
          ati(:,iT,iP+1) = ati(:,iT,iP) + K(:,iP,iT) * v(iP,iT);
          Pti(:,:,iT,iP+1) = Pti(:,:,iT,iP) - K(:,iP,iT) * F(iP,iT) * K(:,iP,iT)';
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
      
      filterOut = struct('a', a, 'P', P, 'Pd', Pd, 'v', v, 'F', F, 'Fd', Fd, ...
        'K', K, 'Kd', Kd, 'dt', dt);
      detailOut = struct('ati', ati, 'Pti', Pti, 'Pdti', Pdti);
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
      filterOut = struct('a', a, 'P', P, 'Pd', Pd, 'v', v, 'F', F, 'Fd', Fd, ...
        'K', K, 'Kd', Kd, 'dt', dt);
    end
    
    function [alpha, smootherOut] = smoother_m(obj, y, fOut)
      % Univariate smoother
      
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
        Nti = obj.T(:,:,obj.tau.T(iT))' * Nti * obj.T(:,:,obj.tau.T(iT));
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
            % NOTE: minus sign!
            L0ti = (fOut.Kd(:,iP,iT) - fOut.K(:,iP,iT)) * Zti * fOut.F(iP,iT) ./ fOut.Fd(iP,iT);
            
            r0ti = Ldti' * r0ti;
            % NOTE: plus sign!
            r1ti = Zti' / fOut.Fd(iP,iT) * fOut.v(iP,iT) + L0ti' * r0ti + Ldti' * r1ti; 
            
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
       
        % What here needs tau_{iT+1}?
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
        'N', N, 'a0tilde', a0tilde);
    end
    
    function [alpha, smootherOut] = smoother_mex(obj, y, fOut)
      % Smoother mex mathematical function
      
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
      
      [alpha, eta, r, N, V, a0tilde] = mfss_mex.smoother_uni(y, ssStruct, fOut);
      smootherOut = struct('alpha', alpha, 'V', V, 'eta', eta, 'r', r, ...
        'N', N, 'a0tilde', a0tilde);
    end
    
    function gradient = gradient_filter_m(obj, y, G, GY, fOut, ftiOut)
      % Gradient mathematical function
      
      nTheta = size(G.T, 1);
      
      Nm = (eye(obj.m^2) + obj.genCommutation(obj.m));
      
      Gati = G.a1;
      GPstarti = G.P1;
      GPdti = zeros(size(G.P1, 1), obj.m^2);
      
      grad = zeros(nTheta, obj.n, obj.p);
      
      % Exact inial recursion
      for iT = 1:fOut.dt
        % warning('StateSpace:diffuse_grad', ...
        %  'Exact initial gradient probably doesn''t work yet.');
        for iP = find(~isnan(y(:,iT)))'
          % Fetch commonly used quantities
          Zti = obj.Z(iP,:,obj.tau.Z(iT));
          GZti = G.Z(:, iP:obj.p:(obj.p*obj.m), obj.tau.Z(iT));
          GHind = 1 + (obj.p+1)*(iP-1); 
          GHti = G.H(:, GHind, obj.tau.H(iT)); % iP-th diagonal element of H
          
          % Non-recursive quantities we'll need in every iteration
          Gvti = GY(:,iT,iP) - GZti * ftiOut.ati(:,iT,iP) ...
            - Gati * Zti' - G.d(:,iP,obj.tau.d(iT));
          GFstarti = 2 * GZti * ftiOut.Pti(:,:,iT,iP) * Zti' ...
            + GPstarti * kron(Zti', Zti') + GHti;
          GKstarti = GPstarti * kron(Zti' ./ fOut.F(iP,iT), eye(obj.m)) ...
            + GZti * ftiOut.Pti(:,:,iT,iP) ./ fOut.F(iP,iT) ...
            - (GFstarti .* fOut.F(iP,iT)^(-2) * Zti * ftiOut.Pti(:,:,iT,iP));
          
          if fOut.Fd(iP,iT) == 0
            % No diffuse to worry about, looks like standard recursion
            grad(:,iT,iP) = 0.5 * GFstarti ...
              * (1./fOut.F(iP,iT) - (fOut.v(iP,iT)^2 ./ fOut.F(iP,iT)^2)) ...
              + Gvti * fOut.v(iP,iT) ./ fOut.F(iP,iT);
            
            % Transition from i to i+1
            Gati = Gati + GKstarti * fOut.v(iP,iT) + Gvti * fOut.K(:,iP,iT)';
            GPstarti = GPstarti ...
              - GKstarti * kron(fOut.F(iP,iT) * fOut.K(:,iP,iT)', eye(obj.m)) * Nm ...
              - GFstarti * kron(fOut.K(:,iP,iT)', fOut.K(:,iP,iT)');
            % Note that GPdt(i+1) = GPdti
          else
            % Have to think about diffuse states
            GFdti = 2 * GZti * ftiOut.Pdti(:,:,iT,iP) * Zti' ...
              + GPdti * kron(Zti', Zti');
            GKdti = GPdti * kron(Zti' ./ fOut.Fd(iP,iT), eye(obj.m)) ...
              + GZti * ftiOut.Pdti(:,:,iT,iP) ./ fOut.Fd(iP,iT) ...
              - (GFdti .* fOut.Fd(iP,iT)^(-2) * Zti * ftiOut.Pdti(:,:,iT,iP));
            
            grad(:,iT,iP) = 0.5 * GFdti ./ fOut.Fd(iP,iT);
            
            % Transition from i to i+1
            Gati = Gati + GKdti * fOut.v(iP,iT) + Gvti * fOut.Kd(:,iP,iT)';
            Kchain = (fOut.K(:,iP,iT) * fOut.Kd(:,iP,iT)' ...
              + fOut.Kd(:,iP,iT) * fOut.K(:,iP,iT)' ...
              - fOut.Kd(:,iP,iT) * fOut.Kd(:,iP,iT)');
            GPstarti = GPstarti ...
              - GFdti * reshape(Kchain, [], 1)' ...
              - GKstarti * kron(fOut.Kd(:,iP,iT)', eye(obj.m)) * Nm * fOut.F(iP,iT) ...
              - GKstarti * kron(fOut.K(:,iP,iT)' - fOut.Kd(:,iP,iT)', eye(obj.m)) * Nm * fOut.F(iP,iT);
            GPdti = GPdti ...
              - GKdti * kron(fOut.Fd(iP,iT) * fOut.Kd(:,iP,iT)', eye(obj.m)) * Nm ...
              - GFdti * kron(fOut.Kd(:,iP,iT)', fOut.Kd(:,iP,iT)');
          end
        end
        
        % Transition from time t to t+1
        Tt = obj.T(:,:,obj.tau.T(iT+1));
        Rt = obj.R(:,:,obj.tau.R(iT+1));
        Qt = obj.Q(:,:,obj.tau.Q(iT+1));
                
        Gati = G.T(:,:,obj.tau.T(iT+1)) * kron(ftiOut.ati(:,iT,obj.p+1), eye(obj.m)) ...
          + Gati * Tt' + G.c(:,:,obj.tau.c(iT+1));
        GTPstarT = G.T(:,:,obj.tau.T(iT+1)) ...
          * kron(ftiOut.Pti(:,:,iT,obj.p+1) * Tt', eye(obj.m)) * Nm ...
          + GPstarti * kron(Tt', Tt');
        GRQR = G.R(:,:,obj.tau.R(iT+1)) * kron(Qt * Rt', eye(obj.m)) * Nm ...
          + G.Q(:,:,obj.tau.Q(iT+1)) * kron(Rt', Rt');
        GPstarti = GTPstarT + GRQR;
      end
      
      GPti = GPstarti;
      
      % Standard recursion
      for iT = fOut.dt+1:obj.n
        for iP = 1:obj.p
          % There's probably a simplification we can make if the y_t,i is
          % missing but I think its safer for now to not have anything about it.
          if isnan(y(iP,iT)); continue; end

          % Fetch commonly used quantities
          Zti = obj.Z(iP,:,obj.tau.Z(iT));
          GZti = G.Z(:, iP:obj.p:(obj.p*obj.m), obj.tau.Z(iT));
          GHind = 1 + (obj.p+1)*(iP-1); 
          GHti = G.H(:, GHind, obj.tau.H(iT)); % iP-th diagonal element of H
          
          % Compute non-recursive gradient quantities
          Gvti = GY(:,iT,iP) - GZti * ftiOut.ati(:,iT,iP) ...
            - Gati * Zti' - G.d(:,iP,obj.tau.d(iT));
          GFti = 2 * GZti * ftiOut.Pti(:,:,iT,iP) * Zti' ...
            + GPti * kron(Zti', Zti') + GHti;
          GKti = GPti * kron(Zti' ./ fOut.F(iP,iT), eye(obj.m)) ...
            + GZti * ftiOut.Pti(:,:,iT,iP) ./ fOut.F(iP,iT) ...
            - (GFti .* fOut.F(iP,iT)^(-2) * Zti * ftiOut.Pti(:,:,iT,iP));
          
          % Comptue the period contribution to the gradient
          grad(:,iT,iP) = 0.5 * GFti ...
            * (1./fOut.F(iP,iT) - (fOut.v(iP,iT)^2 ./ fOut.F(iP,iT)^2)) ...
            + Gvti * fOut.v(iP,iT) ./ fOut.F(iP,iT);
          
          % Transition from i to i+1
          Gati = Gati + GKti * fOut.v(iP,iT) + Gvti * fOut.K(:,iP,iT)';
          GPti = GPti - GKti * kron(fOut.F(iP,iT) * fOut.K(:,iP,iT)', eye(obj.m)) * Nm ...
            - GFti * kron(fOut.K(:,iP,iT)', fOut.K(:,iP,iT)');
        end
        
        % Transition from time t to t+1
        Tt = obj.T(:,:,obj.tau.T(iT+1));
        Rt = obj.R(:,:,obj.tau.R(iT+1));
        Qt = obj.Q(:,:,obj.tau.Q(iT+1));
        
        Gati = G.T(:,:,obj.tau.T(iT+1)) * kron(ftiOut.ati(:,iT,obj.p+1), eye(obj.m)) ...
          + Gati * Tt' + G.c(:,:,obj.tau.c(iT+1));
        GTPT = G.T(:,:,obj.tau.T(iT+1)) ...
          * kron(ftiOut.Pti(:,:,iT,obj.p+1) * Tt', eye(obj.m)) * Nm ...
          + GPti * kron(Tt', Tt');
        GRQR = G.R(:,:,obj.tau.R(iT+1)) * kron(Qt * Rt', eye(obj.m)) * Nm ...
          + G.Q(:,:,obj.tau.Q(iT+1)) * kron(Rt', Rt');
        GPti = GTPT + GRQR;
      end
      
      gradient = -sum(sum(grad, 3), 2);
    end
    
    function [alphaTweights, alphaTconstant] = smoother_weights(obj, y, fOut, sOut, decompPeriods)
      % Generate weights (effectively via multivariate smoother) 
      %
      % Weights ordered (state, observation, data period, state effect period)
      % Constant contributions are ordered (state, effect period)
      
      % pre-allocate W(j,t) where a(:,t) = \sum_{j=1}^{t-1} W(j,t)*y(:,j)
      alphaTweights = zeros(obj.m, obj.p, obj.n, length(decompPeriods)); 
      alphaTconstant = zeros(obj.m, length(decompPeriods)); 
      
      % Add N_{t+1} so the recursion works later
      sOut.N(:,:,end+1) = 0;
      
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
  end
  
  methods (Static)
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
