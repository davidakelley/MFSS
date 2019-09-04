classdef MFDFM < AbstractModel
  % Mixed-frequency dynamic factor model estimated via maximum likelihood
  
  % David Kelley, 2019

  properties
    nFactors

    % TODO: Make constant optional
    constant = true;
  end
  
  properties (SetAccess = protected)
    % Number of lags in VAR
    nLags
  end
  
  properties (Access = protected)
    % Index of which 
    GammaIndex 
  end

  methods
    function obj = MFDFM(data, nFactors, nLags, accumulator)
      % MFDFM Constructor
      % 
      % Arguments: 
      %     data (double): data for VAR (T x p)
      %     nFactors (double): number of factors 
      %     nLags (double): number of lags in factor equation
      %     accumulator (Accumulator): timing specification of data
      % Returns: 
      %     obj (MFVAR): estimation object
      
      obj.Y = data;      
      obj.nFactors = nFactors;
      obj.nLags = nLags;
      obj.p = size(data,2);
      obj.modelName = 'Dynamic Factor Model';
      if nargin > 3 
        obj.accumulator = accumulator;
      else 
        obj.accumulator = Accumulator([], [], []);
      end
      
      % Generate index of which states are used for each observable
      params = struct('Gamma', ones(obj.p, obj.nFactors), ... 
        'obsCons', ones(obj.p,1), ...
        'obsSigma', eye(obj.p), ...
        'phi', ones(obj.nFactors, obj.nFactors * obj.nLags), ...
        'stateCons', ones(obj.nFactors, 1), ...
        'stateSigma', eye(obj.nFactors));
      ssML = obj.params2system(params);
      inxCell = arrayfun(@(x) find(ssML.Z(x,:)), 1:obj.p, 'Uniform', false);
      
      obj.GammaIndex = cat(1, inxCell{:});
    end    
  end
  
  methods
    %% EM algorithm main functions
    function params = estimateParameters(obj, alpha, V, J)
      % Estimate parameters given an estimated state
      
      % Unrestricted parameter estimates
      paramsState = obj.estimateVAR_OLS(alpha, V, J);
      paramsObs = obj.estimateMeas_OLS(obj.Y, alpha, V);
        
      % Scale restriction: shock sign == 1
      paramsState.sigma = eye(obj.nFactors);
      % Sign restriction: positive loadings along the diagonal of Gamma
      signFlips = sign(paramsObs.Gamma(1:obj.p+1:end));
      paramsObs.Gamma = paramsObs.Gamma .* repmat(signFlips, [obj.p 1]);
      
      params = struct('Gamma', paramsObs.Gamma, ...
        'obsCons', paramsObs.cons, ...
        'obsSigma', paramsObs.sigma, ...
        'phi', paramsState.phi, ...
        'stateCons', paramsState.cons, ...
        'stateSigma', paramsState.sigma);
    end
    
    function ssA = params2system(obj, params)
      % Convert VAR parameters into state space object.
      
      Z = [params.Gamma zeros(obj.p, obj.nFactors * (obj.nLags - 1))];
      d = params.obsCons;
      H = params.obsSigma;
      T = [params.phi; ...
        [eye(obj.nFactors * (obj.nLags - 1)) zeros(obj.nFactors * (obj.nLags - 1), obj.nFactors)]];
      c = [params.stateCons; zeros(obj.nFactors * (obj.nLags - 1), 1)];
      R = [eye(obj.nFactors); zeros(obj.nFactors * (obj.nLags - 1), obj.nFactors)];
      Q = params.stateSigma;
      
      ss = StateSpace(Z, H, T, Q, 'd', d, 'c', c, 'R', R);
      
      if ~isempty(obj.accumulator.index)
        ssA = obj.accumulator.augmentStateSpace(ss);
      else
        ssA = ss;
      end
    end
    
    function params = initializeParameters(obj)
      % Initialize parameters from a static factor model
      [alphaBase, Gamma, Sigma] = StockWatsonMF(obj.Y, ...
        'factors', obj.nFactors, 'accum', obj.accumulator, 'verbose', obj.verbose);
      alpha = lagmatrix(alphaBase, 0:obj.nLags-1);
      alpha(isnan(alpha)) = 0;
      
      nStates = obj.nFactors * obj.nLags;
      zeroMats = zeros([nStates, nStates, size(alphaBase,1)]);
      V = zeroMats;
      J = zeroMats;
      paramsState = estimateVAR_OLS(obj, alpha, V, J);
      
      params = struct('Gamma', Gamma, ...
        'obsCons', zeros(obj.p, 1), ...
        'obsSigma', Sigma, ...
        'phi', paramsState.phi, ...
        'stateCons', paramsState.cons, ...
        'stateSigma', paramsState.sigma);
    end    
    
    %% Helper functions
      
    function params = estimateVAR_OLS(obj, alpha, V, J)
      % Estimate the OLS VAR taking the uncertainty of the state into account. 
      %
      % Everything here is in the companion VAR(1) form from the smoother. 
      % We just need to worry about estimating the VAR(1) case then. 
      
      lagInx = 1:size(V,3)-1;
      coincInx = 2:size(V,3);
      
      constants = ones(1, length(lagInx))';
      
      xInd = 1:obj.nFactors*obj.nLags;
      yInd = 1:obj.nFactors;
      
      % OLS regression with uncertainty for state
      xvals = [alpha(lagInx, xInd), constants];
      yvals = alpha(coincInx, yInd);
      addVx = sum(V(xInd, xInd, lagInx), 3);
      addJ  = sum(J(xInd, yInd, lagInx), 3)';
      addVy = sum(V(yInd, yInd, coincInx),3);
      
      xxT =  blkdiag(addVx,0) + xvals' * xvals;
      yxT = [addJ zeros(obj.nFactors, 1)] + yvals' * xvals;
      yyT = addVy + yvals' * yvals;
      
      OLS_state = yxT/xxT;
      SigmaRaw = (yyT-OLS_state*yxT') ./ (size(alpha,1) - 1);
      Sigma = (SigmaRaw + SigmaRaw') ./ 2;
     
      params = struct('phi', OLS_state(:, 1:obj.nFactors*obj.nLags), ...
        'cons', OLS_state(:,end), ...
        'sigma', Sigma);
    end
    
    function params = estimateMeas_OLS(obj, y, alpha, V)
      % Estimate OLS regression of the observations on the state (with state uncertainty)
      %
      % Compute it all together for simplicity, even though it's faster by-equation.
      
      coefs = nan(obj.p, obj.nFactors + 1);
      sigma = zeros(obj.p);
      for iS = 1:obj.p
        xInd = obj.GammaIndex(iS,:);
        obsInd = ~isnan(obj.Y(:,iS));
        constants = ones(sum(obsInd), 1);
        
        yvals = y(obsInd,iS);
        xvals = [alpha(obsInd,xInd) constants];
        addV = sum(V(xInd,xInd,obsInd), 3);
        
        xxT = blkdiag(addV, 0) + xvals' * xvals;
        yxT = yvals' * xvals;
        yyT = yvals' * yvals;
        
        coefs(iS,:) = yxT / xxT;
        sigma(iS,iS) = (yyT-coefs(iS,:)*yxT') ./ sum(obsInd);
      end
      
      params = struct('Gamma', coefs(:,1:end-1), ...
        'cons', coefs(:,end), ...
        'sigma', sigma);
    end
  end
end
