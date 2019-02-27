classdef MFVAR
  % Mixed-frequency VAR estimated via maximum likelihood
  
  % David Kelley, 2018
  
  properties
    Y
    accumulator
    
    constant = true;
    verbose = false;
    
    tol = 1e-7;
    maxIter = 20000;
    stationaryTol = 1.0001;
    
    diagnosticPlot = true;
  end
  
  properties (Access = protected)
    % Number of lags in VAR
    nLags
  end

  properties (Dependent, Hidden)
    % Number of series in VAR
    p
    % Sample length
    n
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
    end
    
    function p = get.p(obj)
      % Getter for p
      p = size(obj.Y, 2);
    end
    
    function n = get.n(obj)
      % Getter for n
      n = size(obj.Y, 1);
    end
    
    function ssML = estimate(obj)
      % Estimate maximum likelihood parameters via EM algorithm
      % 
      % Arguments:
      %     [none]
      % Returns: 
      %     ssML: StateSpace of estimated MF-VAR
      
      if obj.verbose
        algoTitle = 'Mixed-Frequency VAR EM Estimation';
        line = @(char) repmat(char, [1 46]);
        fprintf('\n%s\n', algoTitle);
        fprintf('%s\n  Iteration  |  Log-likelihood |  Improvement\n%s\n', ...
          line('='), line('-'));
        tolLvl = num2str(abs(floor(log10(obj.tol)))+1);
        screenOutFormat = ['%11.0d | %16.8f | %12.' tolLvl 'f\n'];
      end
      
      tm = generateTM(obj);

      alpha = obj.initializeState();
      alpha0 = alpha;

      zeroMats = zeros([obj.p*obj.nLags, obj.p*obj.nLags, size(alpha0,1)]);
      V = zeroMats;
      J = zeroMats;
      a0 = zeros(size(alpha0, 2) + length(obj.accumulator.index), 1);
      P0 = 1000 * eye(size(alpha0, 2) + length(obj.accumulator.index));
      
      params = obj.estimateOLS_VJ(alpha, V, J);
     
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
        [alpha, logli, V, J, a0, ssVAR, theta] = obj.stateEstimate(params, a0, P0, tm);
        
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
    
    function [sampleStates, paramSamples, ssMedian] = sample(obj, nBurn, nKeep)
      % Take samples of the parameters and states
      %
      % Arguments: 
      %   nBurn (integer): samples to discard in warmup
      %   nKeep (integer): samples to keep
      % 
      % Returns: 
      %   sampleStates (float, 3D): stacked samples of alphaHat
      %   ssMedian (StateSpace): median parameters of the sampled state spaces

      if nargin < 3
        nKeep = 500;
      end
      if nargin < 2
        nBurn = 500;
      end
      
      nTotal = nBurn + nKeep;
      iSamp = 1;
      phiSample = nan(obj.p, obj.p*obj.nLags, nKeep);
      consSample = nan(obj.p, nKeep);
      sigmaSample = nan(obj.p, obj.p, nKeep);
      sampleStates = nan(obj.n, obj.p, nKeep);
      
      % TODO: Add a few iterations of the EM to get near HPD region
      tempMdl = obj;
      tempMdl.maxIter = 50;
      ssML = tempMdl.estimate(); 
      alphaFull0 = ssML.smooth(obj.Y);
      
      alpha0 = alphaFull0(:,1:obj.p*obj.nLags);
      paramSample = obj.sampleParameters(alpha0);
      
      % Set up progress window
      theta = [0 0]';
      progress = EstimationProgress(theta, obj.diagnosticPlot, ...
        obj.p*obj.nLags, obj.params2system(paramSample));
      stop = false;
      
      while iSamp < nTotal+1 && ~stop
        
        [alphaDraw, ssLogli] = obj.sampleState(paramSample);
        stateSample = alphaDraw(:,1:obj.p);
        
        [paramSample, paramLogML] = obj.sampleParameters(alphaDraw);
        
        % Update progress window
        progress.alpha = alphaDraw';  
        progress.ss = obj.params2system(paramSample);
        oVals.fval = -(ssLogli + paramLogML);
        stop = progress.update(theta, oVals);

        if iSamp > nBurn
          sampleStates(:,:,iSamp-nBurn) = stateSample;
          phiSample(:,:,iSamp-nBurn) = paramSample.phi;
          consSample(:,iSamp-nBurn) = paramSample.cons;
          sigmaSample(:,:,iSamp-nBurn) = paramSample.sigma;
        end
        if ~all(all(isnan(paramSample.phi)))
          iSamp = iSamp + 1;
        end
      end
      
      paramSamples = struct('phi', phiSample, 'cons', consSample, 'sigma', sigmaSample);
      
      phiMedian = median(phiSample, 3);
      consMedian = median(consSample, 2);
      sigmaMedian = median(sigmaSample, 3);
      ssMedian = obj.params2system(struct('phi', phiMedian', ...
        'cons', consMedian, 'sigma', sigmaMedian));
    end
  end
  
  %% EM algorithm
  methods (Hidden)
    function [state, logli, V, J, a0tilde, ssVAR, theta] = stateEstimate(obj, params, a0, P0, tm)
      % Estimate latent state and variances
      
      [ssVAR, theta] = obj.params2system(params, tm);
      ssVAR.a0 = a0;
      ssVAR.P0 = P0;
      
      [state, sOut, fOut] = ssVAR.smooth(obj.Y);
      logli = sOut.logli;
            
      % No observed data in period 0, L_0 = T_1.
      if isempty(ssVAR.tau)
        L0 = ssVAR.T;
      else
        L0 = ssVAR.T(:,:,ssVAR.tau.T(1));
      end
      r0 = L0' * sOut.r(:,1);
      a0tilde = ssVAR.a0 + ssVAR.P0 * r0;
      
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
      addVx = sum(V(xInd, xInd, lagInx), 3);
      addJ  = sum(J(xInd, yInd, lagInx), 3)';
      addVy = sum(V(yInd, yInd, coincInx),3);
      
      % Restricted OLS Regression of errors
      xxT =  blkdiag(addVx,0) + xvals' * xvals;
      yxT = [addJ zeros(obj.p, 1)] + yvals' * xvals;
      yyT = addVy + yvals' * yvals;
      
      OLS = yxT/xxT;
      Sigma = (yyT-OLS*yxT') ./ (size(alpha,1) - 1);
      Sigma = (Sigma + Sigma') ./ 2;
      
      params = struct('phi', OLS(:, 1:obj.p*obj.nLags), 'cons', OLS(:,end), ...
        'sigma', Sigma);
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
        [Btemp, sigmaDraw] = obj.MNIWDraw(B, PcholTranspose', Sinv, degf);
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
    
  %% Utility methods
  methods (Hidden)
    function alpha = initializeState(obj)
      % Initialize with simple interpolation
      interpY = obj.interpolateData(obj.Y, obj.accumulator);
      alpha = lagmatrix(interpY, 0:obj.nLags-1);
      alpha(isnan(alpha)) = 0;
    end
    
    function tm = generateTM(obj)
      % Generate ThetaMap for VAR
      Z = [eye(obj.p) zeros(obj.p, obj.p * (obj.nLags - 1))];
      H = zeros(obj.p);
      T = [nan(obj.p, obj.p*obj.nLags); ...
        [eye(obj.p * (obj.nLags - 1)) zeros(obj.p * (obj.nLags - 1), obj.p)]];
      c = [nan(obj.p,1); zeros(obj.p * (obj.nLags - 1), 1)];
      R = [eye(obj.p); zeros(obj.p * (obj.nLags - 1), obj.p)];
      Q = nan(obj.p);
      ssE = StateSpaceEstimation(Z, H, T, Q, 'c', c, 'R', R);
      if ~isempty(obj.accumulator.index)
        ssEA = obj.accumulator.augmentStateSpaceEstimation(ssE);
        tm = ssEA.ThetaMapping;
      else
        tm = ssE.ThetaMapping;
      end
    end
    
  end
  
  methods (Static, Hidden)
    function interpY = interpolateData(Y, accum)
      % Interpolate any low-frequency data in Y. 
      interpY = Y;
      for iS = accum.index
        interpY(:,iS) = interp1(find(~isnan(Y(:,iS))), Y(~isnan(Y(:,iS)), iS), ...
          1:size(Y,1), 'linear', 'extrap');
      end
    end
    
    function [X,W,WInv,WChol,WLogDet] = MNIWDraw(muMat, PChol, SInv, v)
      % Generates a draw of (X,W)~MNIW(muMat,P,S,v) such that
      %   X|W ~ MN( muMat, W   kron P )
      %   W ~ IW( v    , S          )
      %
      % *Notes*
      % 1. *Pchol=chol(P)*
      % 2. *Sinv=inv(S)*
      %
      % Input
      % muMat: [p,q] matrix with Mean
      % PCol:  [p,p] matrix with *PChol=Chol(P)*, i.e. P=PChol'*PChol
      % SInv:  [q,q] INVERSE matrix for IW, Sinv=Inv(S)
      % v:     (scalar) degrees of Freedom for IW
      %
      % Output
      %
      % X  [p,q] draw from  X|W ~ MN( muMat, W   kron P )
      % W  [q,q] draw from  IW( v , S )   S=inv(SInv)
      % WInv  [q,q] inverse of W through SVD if Nargout > 2
      % WChol [q,q] chol    of W s.t. W=WChol'*WChol
      %             NOTE: this is A Cholesky factor, but not the
      %                   upper triangular cholesky factor obtained by calling
      %                   Chol
      % WLogDet           Log determinant
      % Alejandro Justiniano February 4 2014 (C)
      
      [Nr,Nc] = size(muMat);
      
      % 1. Obtain draw of W ~ IW(v,S)
      drMat = mvnrnd(zeros(1,Nc), SInv, v);
      Wtemp = (drMat' * drMat) \ eye(Nc);
      % This is more robust but probably slower than inv(W) for n small
      W = 0.5*(Wtemp + Wtemp');
      
      % 2. Obtain chol(W) and inv(W) using the SVD      
      
      % 2.a SVD
      % PP*DD*PPinv'=W  Notice the transpose
      % PPinv=inv(PP)'
      % PPinv'=inv(PP);
      [WChol,flagNotPD]=chol(W);
      if flagNotPD~=0
        [PP,DD,PPinv]=svd(W);
        % 2.b Truncate small singular values
        tolZero=eps;
        firstZero = find(diag(DD) < tolZero, 'first');
        if isempty(firstZero)==false
          PP=PP(:,1:firstZero-1);
          PPinv=PPinv(:,1:firstZero-1);
          DD=DD(1:firstZero-1,1:firstZero-1);
        end
        WChol=sqrt(DD)*PPinv';
      end
      
      % 3. Inverse and Cholesky
      if nargout > 2
        WInv =PP*(DD\eye(Nc))*PPinv';
        WLogDet=sum(log(diag(DD)));
      end
      
      % 4. Draw from MN( mu, W kron P )
      X=(PChol')*(randn(Nr,Nc))*WChol+muMat;
    end
    
    function logGamma = mvgamma(n,degf)
      % =====================================
      % mvgamma
      %
      % function logGamma=mvgamma(n,degf)
      %
      % Multivariate Gamma Function of dimension *n* with *degf*
      % degrees of freedom.
      %
      % Output is log(gamma^n(degf)) *including the constant*
      %
      % Alejandro Justiniano February 13 2014
      if degf <= (n-1)/2
        disp('logGamma is infinite!')
      end
      vecArg=(degf+.5*(0:-1:1-n));
      logGamma=sum(gammaln(vecArg))+0.25*n*(n-1)*log(pi);
    end    
  end
end

