classdef MFVAR < AbstractModel
  % Mixed-frequency VAR estimated via maximum likelihood
  
  % David Kelley, 2018
  
  properties
    constant = true;
    stationaryTol = 1.0001;
  end
  
  properties (SetAccess = protected)
    % Number of lags in VAR
    nLags
  end

  methods
    function obj = MFVAR(data, lags, accumulator)
      % MFVAR Constructor
      % 
      % Arguments: 
      %     data (double): data for VAR (T x p)
      %     lags (double): number of lags to include in VAR
      %     accumulator (Accumulator): timing specification of data
      % Returns: 
      %     obj (MFVAR): estimation object
      
      obj.Y = data;      
      obj.nLags = lags;
      if nargin > 2 
        obj.accumulator = accumulator;
      else 
        obj.accumulator = Accumulator([], [], []);
      end
      obj.modelName = 'VAR';
    end
  end
  
  %% EM algorithm
  methods (Hidden)
    function ssA = params2system(obj, params)
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
    end
    
    function params = estimateParameters(obj, alpha, V, J)
      % Estimate the OLS VAR taking the uncertainty of the state into account. 
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
      addVx = blkdiag(sum(V(xInd, xInd, lagInx), 3), 0);
      addJ  = [sum(J(xInd, yInd, lagInx), 3)' zeros(obj.p, 1)];
      addVy = sum(V(yInd, yInd, coincInx),3);
      
      % Restricted OLS Regression of errors
      xxT = addVx + xvals' * xvals;
      yxT = addJ + yvals' * xvals;
      yyT = addVy + yvals' * yvals;
      
      OLS = yxT/xxT;
      
      % What is the right thing to divide by to get Sigma? 
      % T - 1?
      % T? I don't think this is right because the DFM case uses T-1
      SigmaRaw = (yyT-OLS*yxT') ./ (size(alpha,1) - 1);
      
      % Ensure Sigma is symmetric
      Sigma = (SigmaRaw + SigmaRaw') ./ 2;
      
      params = struct('phi', OLS(:, 1:obj.p*obj.nLags), 'cons', OLS(:,end), ...
        'sigma', Sigma);
    end
    
    function params = initializeParameters(obj)
      % Initialize parameters from a static factor model
      [~, ~, ~, Yhat] = StockWatsonMF(obj.Y, ...
        'factors', 1, 'accum', obj.accumulator, 'verbose', obj.verbose);
      alpha = lagmatrix(Yhat, 0:obj.nLags-1);
      alpha(isnan(alpha)) = 0;
      
      zeroMats = zeros([obj.p * obj.nLags, obj.p * obj.nLags, obj.n]);
      params = obj.estimateParameters(alpha, zeroMats, zeroMats);
    end    
  end
  
  %% Gibbs sampler
  methods (Hidden)
    function [paramsDraw, logML] = sampleParameters(obj, alphaDraw)
      
      % Generate Lags and Regressor matrix X
      Ydraw = alphaDraw(2:end, 1:obj.p);
      constVec = ones(size(alphaDraw, 1)-1,1);
      Xdraw = [alphaDraw(1:end-1, :) constVec];

      % TODO: Add dumm data priors
%       if isempty(YDum)==false
%         Ydraw=[YDum; Ydraw];
%         Xdraw=[XDum; Xdraw];
%       end
      
      [vl,d,vr] = svd(Xdraw,0);
      di = 1 ./ diag(d);
      Btemp = vl' * Ydraw;
      B = (vr .* repmat(di', obj.p * obj.nLags + obj.constant, 1)) * Btemp;
      PcholTranspose = vr .* repmat(di', obj.p * obj.nLags + obj.constant, 1);

      % Matrix of residuals
      u = Ydraw - Xdraw * B;
      S = u' * u;
      degf = size(Ydraw,1) - size(Xdraw,2);
      Sinv = S \ eye(obj.p);
      
      % Storage Matrix for the draws
      maxAttempt = 1000;
      flagContinue = 0;
      countIter = 0;
      while  flagContinue < 1
        [Btemp, sigmaDraw] = AbstractModel.drawMNIW(B, PcholTranspose', Sinv, degf);
        T = [Btemp(1:end-obj.constant,:)'; ...
          eye(obj.p*(obj.nLags-1)) zeros(obj.p*(obj.nLags-1),obj.p)];
        maxEig = max(abs(eig(T))); 
        flagContinue = maxEig < obj.stationaryTol;
        
        countIter = countIter + 1;
        assert(countIter < maxAttempt, 'Maximum number of iterations inside MCMC');
      end
      
      phiDraw = Btemp(1:obj.p*obj.nLags,:)';
      consDraw = Btemp(end,:)';
      
      paramsDraw = struct('phi', phiDraw, 'cons', consDraw, 'sigma', sigmaDraw);
      
      SChol = chol(S);
      SDetLog = 2 * sum(log(diag(SChol)));
      XXInv = PcholTranspose * PcholTranspose';
      XXInvChol = chol(XXInv);
      XXInvDetLog = 2 * sum(log(diag(XXInvChol)));
      k = size(XXInv,1);
      nS = size(S,1);

      logML = -0.5 * (obj.n - k) * nS * log(pi) ...
        - 0.5 * (obj.n - k) * SDetLog ...
        + 0.5 * nS * XXInvDetLog + obj.mvgamma(nS,(obj.n - k)/2); 
    end
    
    function [alphaTilde, ssLogli] = sampleState(obj, params)
      % Take a draw of the state
      % 
      
      % TODO: add compact 
      ss = obj.params2system(params);
      [alphaTilde, ssLogli] = ss.smoothSample(obj.Y);
    end
    
    %{
    function alpha = smooth_compact(obj, params, data)
      % Compact smoother for VAR 
      
    end
    
    function [ssFull, ssCompact] = generateCompact(phi, cons, sigma)
      
      % number of y variables, and number of obs
      nyvars = size(Yt, 1);
      % number of x variables, and number of obs
      [nxvars,n] = size(Xt);
      % Number of coefficients for each variable
      p  = size(phi,2);
      
      % Take Cholesky and Construct C Submatrices
      C = chol(sigma, 'lower');
      Cqq = C(1:nyvars, 1:nyvars);
      Cmq = C(nyvars+1:end, 1:nyvars);
      Cmm = C(nyvars+1:end, nyvars+1:end);
      invCqq = eye(nyvars) / Cqq;
      
      % Build Lag Xt Data Matrix
      X0_lag =  [X0 Xt];
      
      XtLags = zeros(nxvars*obj.nLags,n+1);
      % Loop over through number of lags to input "lagged" data from Xt
      for ii=1:obj.nLags
        XtLags((ii-1)*nxvars+1:ii*nxvars,:) = X0_lag(:,end-n+1-ii:end-ii+1);
      end
      
      % Construct coly,colx Indices
      coefColumns = reshape(1:p,nyvars+nxvars,obj.nLags);
      colY = reshape(coefColumns(1:nyvars,:),1,[]);
      colX = reshape(coefColumns(nyvars+1:end,:),1,[]);
      
      % Rip Out Sub Beta/Constant Matrices
      betaYY = phi(1:nyvars,colY);
      betaYX = phi(1:nyvars,colX);
      betaXX = phi(nyvars+1:end,colX);
      betaXY = phi(nyvars+1:end,colY);
      
      constantY = cons(1:nyvars);
      constantX = cons(nyvars+1:end);
      
      % Create Compact System Matrices
      Z = [eye(nyvars) zeros(nyvars,nyvars*obj.nLags);
        Cmq*invCqq betaXY-Cmq*invCqq*betaYY];
      d = [zeros(nyvars,n);
        repmat(constantX,1,n)-Cmq*invCqq*repmat(constantY,1,n)+...
        (betaXX-Cmq*invCqq*betaYX)*XtLags(:,1:n)];
      H = [zeros(nyvars,nyvars+nxvars); zeros(nxvars,nyvars) Cmm*Cmm'];
      
      T = [betaYY zeros(nyvars,nyvars); ...
        eye(nyvars*obj.nLags) zeros(nyvars*obj.nLags,nyvars)];
      c = [repmat(constantY,1,n+1)+betaYX*XtLags;
        zeros(nyvars*obj.nLags,n+1)];
      R = [eye(nyvars);
        zeros(nyvars*obj.nLags,nyvars)];
      Q =  Cqq*Cqq';
      
      ssCompact = StateSpace(Z, H, T, Q, 'd', d, 'c', c, 'R', R);
      
      % Augment Original "Full" State Space System Matrices w/ Additional Lag
      nVar = nxvars + nyvars;
      numQ = nyvars;
      
      Zfull = [matStru.Z zeros(nVar,numQ)];
      dfull =  matStru.d;
      Hfull =  matStru.H;
      
      numCoefs = size(matStru.T,1);
      numLags = numCoefs / nVar;
      
      Tfull = [matStru.T zeros(numCoefs,numQ); zeros(numQ,nVar*(numLags-1)) eye(numQ) zeros(numQ,nVar)];
      cfull = [matStru.c; zeros(numQ,1)];
      Rfull = [matStru.R; zeros(numQ,nVar)];
      Qfull =  matStru.Q;
      
      fullSS = struct('Z', Zfull, 'd', dfull, 'H', Hfull, ...
        'T', Tfull, 'c', cfull, 'R', Rfull, 'Q', Qfull);
      ssFull = StateSpace(Z, H, T, Q);
    end
    %}    
  end  
end

