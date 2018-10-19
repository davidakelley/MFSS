classdef MFVAR
  % MFVAR Mixed-Frequency VAR
  %
  % Mixed-frequency VAR estimated via maximum likelihood and Bayesian methods
  
  % David Kelley, 2018
  
  properties
    Y
    presample
    accumulator
    
    nLags
    constant = true;
    verbose = true;
    
    tol = 1e-7;
    maxIter = 20000;
    
    diagnosticPlot = true;
  end
  
  properties (Hidden)
    p  % Number of series
  end
  
  methods
    
    function obj = MFVAR(data, lags, accumulator)
      
      obj.Y = data;
      
      obj.nLags = lags;
      if nargin > 2 
        obj.accumulator = accumulator;
      else 
        obj.accumulator = Accumulator([], [], []);
      end
      
      obj.p = size(obj.Y, 2);
    end
    
    function ssML = estimate(obj)
      % Estimate maximum likelihood parameters via EM algorithm
      
      if obj.verbose
        algoTitle = 'Mixed-Frequency VAR EM Estimation';
        line = @(char) repmat(char, [1 46]);
        fprintf('\n%s\n', algoTitle);
        fprintf('%s\n  Iteration  |  Log-likelihood |  Improvement\n%s\n', ...
          line('='), line('-'));
        tolLvl = num2str(abs(floor(log10(obj.tol)))+1);
        screenOutFormat = ['%11.0d | %16.8f | %12.' tolLvl 'f\n'];
      end

      % Initialize with simple interpolation
      interpY = obj.interpolateData(obj.Y, obj.accumulator);
      alpha0 = lagmatrix(interpY, 0:obj.nLags-1);
      alpha0(any(isnan(alpha0),2),:) = [];
      zeroMats = zeros([obj.p*obj.nLags, obj.p*obj.nLags, size(alpha0,1)]);

      alpha = alpha0;
      V = zeroMats;
      J = zeroMats;
      a0 = zeros(size(alpha0, 2) + length(obj.accumulator.index), 1);
      P0 = 1000 * eye(size(alpha0, 2) + length(obj.accumulator.index));
      
      params = obj.estimateOLS_VJ(alpha, V, J);
     
      % Generate ThetaMap for progress window
      Z = [eye(obj.p) zeros(obj.p, obj.p * (obj.nLags - 1))];
      H = zeros(obj.p);
      T = [nan(size(params.phi)); [eye(obj.p * (obj.nLags - 1)) zeros(obj.p * (obj.nLags - 1), obj.p)]];
      c = [nan(size(params.cons)); zeros(obj.p * (obj.nLags - 1), 1)];
      R = [eye(obj.p); zeros(obj.p * (obj.nLags - 1), obj.p)];
      Q = nan(size(params.sigma));
      ssE = StateSpaceEstimation(Z, H, T, Q, 'c', c, 'R', R);
      if ~isempty(obj.accumulator.index)
        ssEA = obj.accumulator.augmentStateSpaceEstimation(ssE);
        tm = ssEA.ThetaMapping;
      else
        tm = ssE.ThetaMapping;
      end
      
      % Set up progress window
      [ssVAR, theta] = obj.params2system(params, tm);
      progress = EstimationProgress(theta, obj.diagnosticPlot, size(alpha0,2), ssVAR);
      stop = false;
      errorIndicator = '';
      
      % EM algorithm
      iter = 0;
      logli0 = -Inf;
      improvement = -Inf;
      while ~stop && abs(improvement) > obj.tol && iter < obj.maxIter
        % M-step: Get parameters conditional on state
        params = obj.estimateOLS_VJ(alpha, V, J);
       
        % E-step: Get state conditional on parameters
        [alpha, logli, V, J, a0, ~, ssVAR, theta] = obj.stateEstimate(params, a0, P0, tm);
        
        % Put filtered state in figure for plotting
        progress.alpha = alpha';  
        progress.ss = ssVAR;
        if iter < 2
          % Initialization has low likelihood - makes plot uninformative
          oVals.fval = nan;
        else
          oVals.fval = -logli;
        end
        progress.totalEvaluations = progress.totalEvaluations + 1;
        stop = progress.update(theta, oVals);

        % Compute improvement
        improvement = logli - logli0;
        logli0 = logli;
        iter = iter + 1;

        if ~isfinite(logli)
          errorIndicator = 'nanlogli';
          stop = true;
        end
        if improvement < 0
          errorIndicator = 'backup';
          stop = true;
        end
        
        if obj.verbose
          if iter <=2 || improvement < 0 || ~isempty(errorIndicator)
            bspace = [];
          else
            bspace = repmat('\b', [1 length(screenOut)]);
          end
          screenOut = sprintf(screenOutFormat, iter, logli, improvement);
          fprintf([bspace screenOut]);
        end
      end
      
      ssML = obj.params2system(params);
      ssML.a0 = a0;
      ssML.P0 = P0;
      if obj.verbose
        fprintf('%s\n', line('-'));
      end
      
      progress.nextSolver();

      switch errorIndicator
        case 'nanlogli'
          warning('Error in evaluation of log-likelihood.');
        case 'backup'
          warning('EM algorithm did not improve likelihood.');
        case ''
        otherwise
          error('Unknown error.');
      end
    end
  end
  
  %% EM algorithm
  methods (Hidden)
    function [state, logli, V, J, a0tilde, V0, ssVAR, theta] = stateEstimate(obj, params, a0, P0, tm)
      [ssVAR, theta] = obj.params2system(params, tm);
      ssVAR.a0 = a0;
      ssVAR.P0 = P0;
      
      [state, sOut, fOut] = ssVAR.smooth(obj.Y);
      logli = sOut.logli;
      
      a1tilde = state(1,:)'; 
      V1 = sOut.V(:,:,1); 
      
      % No observed data in period 0, L_0 = T_1.
      if isempty(ssVAR.tau)
        L0 = ssVAR.T;
      else
        L0 = ssVAR.T(:,:,ssVAR.tau.T(1));
      end
      r0 = L0' * sOut.r(:,1);
      a0tilde = ssVAR.a0 + ssVAR.P0 * r0;
%       N0 = L0' * sOut.N(:,:,1) * L0;
%       V0 = AbstractSystem.enforceSymmetric(ssVAR.P0 - ssVAR.P0 * N0 * ssVAR.P0);
      
%       a0tilde = a1tilde; 
      V0 = V1;
      
      if nargout > 2
        ssVAR = ssVAR.setDefaultInitial();
        ssVAR = ssVAR.prepareFilter(obj.Y, [], []);
        sOut.N = cat(3, sOut.N, zeros(size(sOut.N, 1)));
        [V, J] = ssVAR.getErrorVariances(obj.Y', fOut, sOut);
      end
    end
    
    function [ssA, theta] = params2system(obj, params, tm)
      % Convert VAR parameters into state space object.
      
      Z = [eye(obj.p) zeros(obj.p, obj.p * (obj.nLags - 1))];
      H = zeros(obj.p);
      T = [params.phi; [eye(obj.p * (obj.nLags - 1)) zeros(obj.p * (obj.nLags - 1), obj.p)]];
      c = [params.cons; zeros(obj.p * (obj.nLags - 1), 1)];
      R = [eye(obj.p); zeros(obj.p * (obj.nLags - 1), obj.p)];
      Q = params.sigma;
      
      ss = StateSpace(Z, H, T, Q, 'c', c, 'R', R);
      
      if ~isempty(obj.accumulator.index)
        ssA = obj.accumulator.augmentStateSpace(ss);
      else
        ssA = ss;
      end
      if nargout > 1
        theta = tm.system2theta(ssA);
      end
    end
    
    function params = estimateOLS_VJ(obj, alpha, V, J)
      % Estimate an OLS VAR taking the uncertainty of the state into account. 
      %
      % Everything here is in the companion VAR(1) form from the smoother. 
      % We just need to worry about estimating the VAR(1) case then. 
      
      lagInx = 1:size(V,3)-1;
      coincInx = 2:size(V,3);
      
      constants = ones(1, length(lagInx))';
      
      xInd = 1:obj.p*obj.nLags;
      yInd = 1:obj.p;
      
      xvals = [alpha(lagInx, xInd), constants];
      yvals = alpha(coincInx, yInd);
      addVx = sum(V(xInd, xInd, lagInx), 3);
      addJ  = sum(J(xInd, yInd, lagInx), 3)';
      addVy = sum(V(yInd, yInd, coincInx),3);
      
      % Restricted OLS Regression of errors
      xxT =  blkdiag(addVx,0) + xvals' * xvals;
      yxT = [addJ zeros(obj.p, 1)] + yvals' * xvals;
      yyT = addVy + yvals' * yvals;
      
      OLS = yxT/xxT;
      Sigma = (yyT-OLS*yxT') ./ (size(alpha,1) - 1); % FIXME: -1? obj.nLags?
      Sigma = (Sigma + Sigma') ./ 2;
      
      params = struct('phi', OLS(:, 1:obj.p*obj.nLags), 'cons', OLS(:,end), ...
        'sigma', Sigma);
    end
  end
  
  %% Utility methods
  methods (Static, Hidden)
    function interpY = interpolateData(Y, accum)
      % Interpolate any low-frequency data in Y. 
      interpY = Y;
      for iS = accum.index
        interpY(:,iS) = interp1(find(~isnan(Y(:,iS))), Y(~isnan(Y(:,iS)), iS), 1:size(Y,1));
      end
    end
  end
end

