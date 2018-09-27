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
    maxIter = 10000;
  end
  
  properties (Hidden)
    p  % Number of series
  end
  
  methods
    
    function obj = MFVAR(data, lags, accumulator)
      
      obj.Y = data;
      
      obj.nLags = lags;
      obj.accumulator = accumulator;
      
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
      alpha0 = lagmatrix(interpY, 1:obj.nLags);
      alpha0(any(isnan(alpha0),2),:) = [];
      zeroMats = zeros([obj.p*obj.nLags, obj.p*obj.nLags, size(alpha0,1)]);
      params = obj.estimateOLS_VJ(alpha0, zeroMats, zeroMats);
      
      % EM algorithm
      iter = 0;
      logli0 = -Inf;
      improvement = -Inf;
      while abs(improvement) > obj.tol && iter < obj.maxIter
        % E-step: Get state conditional on parameters
        [alpha, logli, V, J] = obj.stateEstimate(params);
        
        % M-step: Get parameters conditional on state
        params = obj.estimateOLS_VJ(alpha, V, J);
        
        % Compute improvement
        improvement = logli - logli0;
        logli0 = logli;
        iter = iter + 1;

        if improvement < 0
          warning('EM algorithm did not improve likelihood.');
          break
        end
        
        if obj.verbose
          if iter <=2 || improvement < 0
            bspace = [];
          else
            bspace = repmat('\b', [1 length(screenOut)]);
          end
          screenOut = sprintf(screenOutFormat, iter, logli, improvement);
          fprintf([bspace screenOut]);
        end
      end
      
      ssML = obj.params2system(params);
      if obj.verbose
        fprintf('%s\n', line('-'));
      end
    end
  end
  
  %% EM algorithm
  methods (Hidden)
    function [state, logli, V, J] = stateEstimate(obj, params)
      ssVAR = obj.params2system(params);
      
      [state, sOut, fOut] = ssVAR.smooth(obj.Y);
      logli = sOut.logli;
      
      if nargout > 1
        ssVAR = ssVAR.setDefaultInitial();
        ssVAR = ssVAR.prepareFilter(obj.Y, []);
        sOut.N = cat(3, sOut.N, zeros(size(sOut.N, 1)));
        [V, J] = ssVAR.getErrorVariances(obj.Y', fOut, sOut);
      end
    end
    
    function ssA = params2system(obj, params)
      % Convert VAR parameters into state space object.
      
      Z = [eye(obj.p) zeros(obj.p, obj.p * (obj.nLags - 1))];
      H = zeros(obj.p);
      T = [params.phi; [eye(obj.p * (obj.nLags - 1)) zeros(obj.p * (obj.nLags - 1), obj.p)]];
      c = [params.cons; zeros(obj.p * (obj.nLags - 1), 1)];
      R = [eye(obj.p); zeros(obj.p * (obj.nLags - 1), obj.p)];
      Q = params.sigma;
      
      ss = StateSpace(Z, H, T, Q, 'c', c, 'R', R);
      
      ssA = obj.accumulator.augmentStateSpace(ss);
      
      ssA.a0 = zeros(size(ssA.c, 1), 1);
      ssA.P0 = 10 * eye(size(ssA.T, 1));
    end
    
    function params = estimateOLS_VJ(obj, alpha, V, J)
      % Estimate an OLS VAR taking the uncertainty of the state into account. 
      %
      % Everything here is in the companion VAR(1) form from the smoother. We just need to
      % worry about estimating the VAR(1) case then. 
      
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

