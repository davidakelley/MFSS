classdef (Abstract) AbstractModel 
  % Generic model class with EM algorithm implementation
  
  properties
    Y
    verbose = false;
    diagnosticPlot = true;

    tol = 1e-6;
    maxIter = 20000;    
  end
  
  properties (SetAccess = protected)
    % Accumulator object
    accumulator
 
    % Number of series 
    p
    
    % Model name
    modelName
  end
  
  properties (Dependent, Hidden)
    % Sample length
    n
  end
  
  %% EM Algorithm
  methods
    function [ssML, params] = estimate(obj, params0)
      % Estimate maximum likelihood parameters via EM algorithm
      % 
      % Arguments:
      %     params0 (optional): structure of system parameters to initialize from
      % Returns: 
      %     ssML: StateSpace of estimated model
      %     params: structure of model parameters
      
      % Initialize the parameters
      if nargin < 2
        params0 = obj.initializeParameters();
      end
      
      ss = obj.params2system(params0);

      % Initial state estimate from the initialized values
      alpha = ss.smooth(obj.Y);
      a0 = alpha(1,:)';
      P0 = 1000 * eye(size(alpha, 2));
      
      zeroMats = zeros([size(alpha,2), size(alpha,2), size(alpha,1)]);
      V = zeroMats;
      J = zeroMats;
      
      % Set up progress window
      progress = EstimationProgress(obj.modelName, obj.diagnosticPlot, size(alpha,2), ss);
      stop = false;
      errorIndicator = '';
      
      % EM algorithm
      if obj.verbose
        algoTitle = 'Mixed-Frequency DFM EM Estimation';
        line = @(char) repmat(char, [1 46]);
        fprintf('\n%s\n', algoTitle);
        fprintf('%s\n  Iteration  |  Log-likelihood |  Improvement\n%s\n', ...
          line('='), line('-'));
        tolLvl = num2str(abs(floor(log10(obj.tol)))+1);
        screenOutFormat = ['%11.0d | %16.8f | %12.' tolLvl 'f\n'];
      end
      
      iter = 0;
      logli0 = -Inf;
      improvement = -Inf;
      while ~stop && abs(improvement) > obj.tol && iter < obj.maxIter
        % M-step: Get parameters conditional on state
        params = obj.estimateParameters(alpha, V, J);
        
        % E-step: Get state conditional on parameters
        [alpha, logli, V, J, a0, ss] = obj.estimateState(params, a0, P0);
        
        % Put filtered state in figure for plotting
        progress.alpha = alpha';  
        progress.ss = ss;
        if iter < 2
          % Initialization has low likelihood - makes plot uninformative
          oVals.fval = nan;
        else
          oVals.fval = -logli;
        end
        progress.totalEvaluations = progress.totalEvaluations + 1;
        stop = progress.update([], oVals);

        % Compute improvement
        improvement = logli - logli0;
        logli0 = logli;
        iter = iter + 1;

        if ~isfinite(logli)
          errorIndicator = 'nanlogli';
          stop = true;
        end
        if improvement < 0 
          % If we get a small negative change in the likelihood, call it good enough and
          % stop. If we get a large change in the likelihood, throw the warning.
          if abs(improvement) > 10 * obj.tol
            errorIndicator = 'backup';
          else
            errorIndicator = '';
          end
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
        case ''
        case 'nanlogli'
          warning('Error in evaluation of log-likelihood.');
        case 'backup'
          warning('EM algorithm decreased likelihood by %3.2g.', abs(improvement));
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
  
  %% EM implementation functions
  methods (Abstract)
    % Functions that must be implemented in each model to run the EM algorithm.
    params = estimateParameters(obj, alpha, V, J)
      
    ssA = params2system(obj, params)
     
    params = initializeParameters(obj)
  end
  
  methods
    % Common EM algorithm functions across all models 
    function [state, logli, V, J, a0tilde, ssVAR] = estimateState(obj, params, a0, P0)
      % Estimate latent state and variances
      
      [ssVAR] = obj.params2system(params);
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
        ssVAR_filter = ssVAR.setDefaultInitial();
        ssVAR_filter = ssVAR_filter.prepareFilter(obj.Y, [], []);
        sOut.N = cat(3, sOut.N, zeros(size(sOut.N, 1)));
        [V, J] = ssVAR_filter.getErrorVariances(obj.Y', fOut, sOut);
      end
    end
  end
  
  %% Gibbs sampler implementation
  
  %% Utility Methods
  methods
    function n = get.n(obj)
      % Getter for n
      n = size(obj.Y, 1);
    end
        
    function p = get.p(obj)
      % Getter for p
      p = size(obj.Y, 2);
    end
  end
  
  methods (Static)
    function [X, W] = drawMNIW(muMat, PChol, SInv, v)
      % Generates a draw of (X,W)~MNIW(muMat,P,S,v) such that
      %   X|W ~ MN(muMat, W kron P)
      %   W ~ IW(v, S)
      %
      % Input
      %   muMat: [p,q] matrix with Mean
      %   PCol:  [p,p] matrix with *PChol=Chol(P)*, i.e. P=PChol'*PChol
      %   SInv:  [q,q] INVERSE matrix for IW, Sinv=Inv(S)
      %   v:     (scalar) degrees of Freedom for IW
      %
      % Output
      %   X:  [p,q] draw from  X|W ~ MN(muMat, W kron P)
      %   W:  [q,q] draw from  IW(v, S);   S=inv(SInv)
      %
      % Alejandro Justiniano, February 2014
      
      [Nr,Nc] = size(muMat);
      
      % Obtain draw of W ~ IW(v,S)
      % This is more robust but probably slower than inv(W) for n small
      drMat = mvnrnd(zeros(1,Nc), SInv, v);
      Wtemp = (drMat' * drMat) \ eye(Nc);
      W = 0.5 * (Wtemp + Wtemp');
      
      % Obtain chol(W) and inv(W) using the SVD      
      % PP*DD*PPinv'=W  Notice the transpose
      % PPinv=inv(PP)'
      % PPinv'=inv(PP);
      [WChol, flagNotPD] = chol(W);
      if flagNotPD ~= 0
        [~, DD, PPinv] = svd(W);
        
        % Truncate small singular values
        firstZero = find(diag(DD) < eps, 'first');
        if ~isempty(firstZero)
          PPinv = PPinv(:,1:firstZero-1);
          DD = DD(1:firstZero-1,1:firstZero-1);
        end
        
        WChol = sqrt(DD) * PPinv';
      end
      
      % Draw from MN(mu, W kron P)
      X = PChol' * randn(Nr,Nc) * WChol + muMat;
    end
    
    function logGamma = mvgamma(n, degf)
      % Multivariate Gamma Function of dimension *n* with *degf* degrees of freedom.
      %
      % Output is log(gamma^n(degf)) *including the constant*
      %
      % Alejandro Justiniano, February 2014
      
      if degf <= (n-1)/2
        warning('logGamma is infinite!')
      end
      
      vecArg = degf + 0.5 * (0:-1:1-n);
      logGamma = sum(gammaln(vecArg)) + 0.25 * n * (n-1) * log(pi);
    end    
  end
  
  
end
