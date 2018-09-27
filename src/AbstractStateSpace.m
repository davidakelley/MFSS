classdef (Abstract) AbstractStateSpace < AbstractSystem
  % Abstract state space for accumulators and utilities 
  %
  % Provides the structural parts of the parameter manipulation and some
  % utility functions.
  %
  % Two current subclasses: StateSpace (for systems with known parameters) and
  % StateSpaceEstimation (to estimate unknown parameters of a system).
  
  % David Kelley, 2016-2017

  properties
    % Observation equation parameters
    Z
    d
    beta 
    H
    
    % State equation parameters
    T
    c
    gamma
    R
    Q
    
    % Structure of time-varrying parameter indexes
    tau
  end
  
  properties (Hidden)
    % Numeric gradient precision order (either 1 or 2)
    numericGradPrec = 1;
    % Numeric gradient step size
    delta = 1e-8;    
  end
  
  properties (Abstract, Dependent)
    % Initial state variance
    a0
    P0
  end
  
  properties (Dependent, SetAccess = protected, Hidden)
    P0Private
  end
  
  properties (Dependent, Hidden)
    % Q0 is a primative but we need a dependent property for initialization 
    % order purposes 
    Q0
  end
  
  properties (SetAccess = protected, Hidden)
    % Initial value selection matricies - these are the primatives
    a0Private
    A0
    R0
    Q0Private
    
    % Lists of variables used across methods
    systemParam = {'Z', 'd', 'beta', 'H', 'T', 'c', 'gamma', 'R', 'Q'};
    symmetricParams = {'H', 'Q', 'Q0', 'P'};    % Include P0? P?
    % FIXME: Why P in symmetricParams?
  end
  
  methods
    %% Property getters & setters
    function a0Private = get.a0Private(obj)
      a0Private = obj.a0Private;
    end
    
    function obj = set.a0Private(obj, newa0)
      % Check if obj.m is empty - if it is, we're loading a saved object so ignore the
      % assertion. 
      assert(isempty(obj.m) || isempty(newa0) || size(newa0, 1) == obj.m, ...
        'a0 should be a m X 1 vector');
      obj.a0Private = newa0;
    end
    
    function P0Private = get.P0Private(obj)
      diffuseP0 = obj.A0 * obj.A0';
      diffuseP0(diffuseP0 ~= 0) = Inf;
      P0Private = diffuseP0 + obj.R0 * obj.Q0 * obj.R0';
    end
    
    function obj = set.P0Private(obj, newP0)
      
      if isempty(newP0)
        % Unset initial value
        obj.A0 = [];
        obj.R0 = [];
        obj.Q0Private = [];
        return
      end
      
     % Handle input options for P0
      if size(newP0, 1) == 1
        % Scalar value passed for kappa
        % (Note that this is ok evn if m == 1)
        newP0 = eye(obj.m) * newP0;
      elseif size(newP0, 2) == 1
        % Vector value passed for kappa
        assert(size(newP0, 1) == obj.m, 'kappa vector must be length m');
        newP0 = diag(newP0);
      else
        % Regular P0 value passed
        assert(all(size(newP0) == [obj.m obj.m]), 'P0 should be a m X m matrix');
      end
      
      % Find diffuse elements
      diffuse = any(isinf(newP0), 2);
      nondiffuse = all(~isinf(newP0), 2);
      
      % Set up exact initial parameters
      select = eye(obj.m);
      obj.A0 = select(:, diffuse);
      obj.R0 = select(:, nondiffuse);
      obj.Q0Private = newP0(nondiffuse, nondiffuse);
    end
    
    function Q0 = get.Q0(obj)
      Q0 = obj.Q0Private;
    end
    
    function obj = set.Q0(obj, newQ0)
      assert(~isempty(obj.P0), 'Cannot set Q0 without first setting P0.');      
      obj.Q0Private = newQ0;
    end
  end
 
  methods
    %% Constructor
    function obj = AbstractStateSpace(Z, d, beta, H, T, c, gamma, R, Q)
      % Constructor
      % Inputs: Parameters matricies or a structure of parameters
      if nargin == 1
        % Structure of parameter values or AbstractStateSpace object passed
        parameters = Z;
      elseif nargin == 9
        parameters = struct('Z', Z, 'd', d, 'beta', beta, 'H', H, ...
          'T', T, 'c', c, 'gamma', gamma, 'R', R, 'Q', Q);
      elseif nargin == 0
        return;
      else
        error('MFSS:inputError', 'Input error.');
      end
      
      obj = obj.setSystemParameters(parameters);
    end

    %% Utility methods
    function param = parameters(obj, index)
      % Getter for cell array of parameters
      param = {obj.Z, obj.d, obj.beta, obj.H, ...
        obj.T, obj.c, obj.gamma, obj.R, obj.Q, obj.a0, obj.Q0}; 
      if nargin > 1
        param = param{index};
      end
    end

    function [nLags, positions] = LagsInState(obj, varPos)
      % Finds the lags of a variable in the state space
      % Input:
      %   varPos    - linear index of vector position of variable of interest
      % Output:
      %   nLags     - number of lags in the state of variable of interest
      %   positions - linear index position of the lags
      %
      % Criteria for determining if a candidate variable is a "lag variable":
      %   - The variable of interest loads on to the candidate with a 1.
      %   - All other variables have a loading on the candidate of 0.
      %   - No shocks are selected for the candidate in R.
      
      assert(~isempty(obj.tau), 'tau must be set to find lags in the state.');
      
      eVec = @(size, elem) [false(elem-1, 1); true; false(size-elem, 1)];

      % FIXME: Shouldn't c and gamma show up here too? 
      criteriaFn = @(interest) all([(obj.T(:, interest, obj.tau.T(1)) == 1) ...
        all(obj.T(:,((1:size(obj.T,1)) ~= interest), obj.tau.T(1)) == 0,2) ...
        all(obj.R(:, :, obj.tau.R(1)) == 0, 2) ~eVec(obj.m, interest)], 2);
      
      nLags = 0;
      positions = [];
      while any(criteriaFn(varPos))
        nLags = nLags + 1;
        varPos = find(criteriaFn(varPos));
        positions = [positions; varPos]; %#ok<AGROW>
      end
    end
    
    function [obj, y, x, w] = checkSample(obj, y, x, w)
      % Check the data timing against the time-varrying parameters
      % 
      % Args: 
      %     y (double) : observed data
      %     x (double) : exogenous series in measurement equation
      %     w (double) : exogenous series in state equation
      % 
      % TODO: 
      %     - check that we're only observing accumulated series when we should
      
      if size(y, 1) ~= obj.p && size(y,2) == obj.p
        y = y';
      end
      if nargin > 2 && ~isempty(x) && size(x, 1) ~= obj.k && size(x,2) == obj.k
        x = x';
      elseif nargin < 3 || isempty(x)
        if ~isempty(obj.beta)
          error('MFSS:noX', 'Model specified with beta and no X data provided.')
        end
        x = zeros(obj.k, size(y,2));
      end
      if nargin > 3 && ~isempty(w) && size(w, 1) ~= obj.l && size(w,2) == obj.l
        w = w';
      elseif nargin < 4 || isempty(w)
        if ~isempty(obj.gamma)
          error('MFSS:noW', 'Model specified with gamma and no W data provided.')
        end
        w = zeros(obj.l, size(y,2)+1);
      end
      
      assert(isnumeric(y), 'MFSS:nonnumericY', 'Input data y must be numeric.');
      assert(size(y, 1) == obj.p, ...
        'Number of series in y does not match observation equation.');
      assert(isnumeric(x), 'MFSS:nonnumericX', 'Input data x must be numeric.');
      assert(size(x, 1) == obj.k, ...
        'Number of exogenous series in x does not match observation equation.');
      assert(~any(any(isnan(x))), 'MFSS:nanX', 'Input data x must be not nan.');
      assert(isnumeric(w), 'MFSS:nonnumericW', 'Input data w must be numeric.');
      assert(size(w, 1) == obj.l, ...
        'Number of exogenous series in w does not match state equation.');
      assert(~any(any(isnan(w))), 'MFSS:nanW', 'Input data w must be not nan.');
      
      if ~obj.timeInvariant
        % System with TVP, make sure length of taus matches data.
        assert(size(y, 2) == obj.n);
        if ~isempty(x) 
          assert(size(x, 2) == obj.n);
        end
        if ~isempty(w)
          assert(size(w, 2) == obj.n+1);
        end
      else
        % No TVP, set n then set tau as ones vectors of that length.
        obj.n = size(y, 2);
        if ~isempty(x) 
          assert(size(x, 2) == obj.n);
        end
        if ~isempty(w) 
          assert(size(w, 2) == obj.n+1);
        end

        obj = obj.setInvariantTau();
      end
    end    
    
    function validateStateSpace(obj)
      % Check dimensions of inputs to Kalman filter.
      if obj.timeInvariant
        maxTaus = ones([9 1]);
      else
        maxTaus = cellfun(@(name) max(obj.tau.(name)), obj.systemParam);
      end
      
      validate = @(x, sz, name) validateattributes(x, {'numeric', 'sym'}, ...
        {'size', sz}, 'StateSpace', name);
      
      % Measurement equation
      validate(obj.Z, [obj.p obj.m maxTaus(1)], 'Z');
      validate(obj.d, [obj.p maxTaus(2)], 'd');
      validate(obj.beta, [obj.p obj.k maxTaus(3)], 'beta');
      validate(obj.H, [obj.p obj.p maxTaus(4)], 'H');
      
      % State equation
      validate(obj.T, [obj.m obj.m maxTaus(5)], 'T');
      validate(obj.c, [obj.m maxTaus(6)], 'c');
      validate(obj.gamma, [obj.m obj.l maxTaus(7)], 'c');
      validate(obj.R, [obj.m obj.g maxTaus(8)], 'R');
      validate(obj.Q, [obj.g obj.g maxTaus(9)], 'Q');
    end
    
  end
  
  methods (Hidden = true)
    %% Constructor helper methods
    function obj = setSystemParameters(obj, parameters)
      % Obtain system matrices from structure
      % By passing both the different types of system matrices and the calendar
      % vector corresponding to the particular type used at any particular time
      % we allow for an arbitrary number of structural "breaks" in the state
      % space model. Furthermore, by allowing each system matrix/vector to have
      % its own calendar, no redundant matrices are saved in the workspace.
      
      if isa(parameters, 'StateSpace')
        [pZ, pd, pbeta, pH, pT, pc, pR, pQ] = parameters.getInputParameters();
        parameters = struct('Z', pZ, 'd', pd, 'beta', pbeta, 'H', pH, ...
          'T', pT, 'c', pc, 'gamma', pgamma, 'R', pR, 'Q', pQ);
      end
      structParams = structfun(@isstruct, parameters);
      
      if ~any(structParams)
        obj.timeInvariant = true;
      else
        obj.timeInvariant = false;
      
        tauLens = zeros(9,1);
        tauLens(structParams) = cellfun(@(x) ...
          length(parameters.(x).(['tau' x])), obj.systemParam(structParams));
        subtract = structParams & [0 0 0 0 1 1 1 1 1]';
        nCandidates = tauLens - subtract;
        assert(all(nCandidates(structParams) == max(nCandidates)), ['Bad tau specification. ' ... 
          'tau vectors for Z, d, beta & H should have n elements. ' ...
          'tau vectors for T, c, gamma, R & Q should have n+1 elements.']);
        obj.n = max(nCandidates);
      end
      
      % Set dimensions:
      %   p - number of observables
      %   m - number of state variables
      %   g - number of shocks
      %   k - number of exogenous measurement variables
      %   l - number of exogenous state variables 
      
      % Z system matrix
      if isstruct(parameters.Z)
        obj = obj.setTimeVarrying(length(parameters.Z.tauZ));
        assert(numel(parameters.Z.tauZ) == obj.n);
        obj.tau.Z = parameters.Z.tauZ;
        obj.Z = parameters.Z.Zt;
      else
        obj.Z = parameters.Z;
        if ~obj.timeInvariant
          obj.tau.Z = ones(obj.n, 1);
        end
      end
      obj.p = size(obj.Z, 1);
      obj.m = size(obj.Z, 2);

      % d system matrix
      if isempty(parameters.d)
        obj.d = zeros(obj.p, 1);
        if ~obj.timeInvariant
          obj.tau.d = ones(obj.n, 1);
        end
      elseif isstruct(parameters.d)
        obj = obj.setTimeVarrying(length(parameters.d.taud));
        assert(numel(parameters.d.taud) == obj.n);
        obj.tau.d = parameters.d.taud;
        obj.d = parameters.d.dt;
      elseif size(parameters.d, 2) > 1
        obj = obj.setTimeVarrying(size(parameters.d, 2));
        obj.tau.d = 1:obj.n;
        obj.d = parameters.d;
      else
        obj.d = parameters.d;
        if ~obj.timeInvariant
          obj.tau.d = ones(obj.n, 1);
        end
      end
      
      % beta system matrix
      if isempty(parameters.beta)
        obj.beta = zeros(obj.p, 0);
        if ~obj.timeInvariant
          obj.tau.beta = ones(obj.n, 1);
        end
      elseif isstruct(parameters.beta)
        obj = obj.setTimeVarrying(length(parameters.beta.taubeta));
        assert(numel(parameters.beta.taubeta) == obj.n);
        obj.tau.beta = parameters.beta.taubeta;
        obj.beta = parameters.beta.betat;
      elseif size(parameters.beta, 3) > 1
        obj = obj.setTimeVarrying(size(parameters.beta, 3));
        obj.tau.beta = 1:obj.n;
        obj.beta = parameters.beta;
      else
        obj.beta = parameters.beta;
        if ~obj.timeInvariant
          obj.tau.beta = ones(obj.n, 1);
        end
      end
      obj.k = size(obj.beta, 2);
      
      % H system matrix
      if isstruct(parameters.H)
        obj = obj.setTimeVarrying(length(parameters.H.tauH));
        assert(numel(parameters.H.tauH) == obj.n);
        obj.tau.H = parameters.H.tauH;
        obj.H = parameters.H.Ht;
      else
        obj.H = parameters.H;
        if ~obj.timeInvariant
          obj.tau.H = ones(obj.n, 1);
        end
      end
      
      % T system matrix
      if isstruct(parameters.T)
        obj = obj.setTimeVarrying(length(parameters.T.tauT) - 1);
        assert(numel(parameters.T.tauT) == obj.n+1);
        obj.tau.T = reshape(parameters.T.tauT, obj.n+1, 1);
        obj.T = parameters.T.Tt;
      else
        obj.T = parameters.T;
        if ~obj.timeInvariant
          obj.tau.T = ones(obj.n+1, 1);
        end
      end
      
      % c system matrix
      if isempty(parameters.c)
        obj.c = zeros(obj.m, 1);
        if ~obj.timeInvariant
          obj.tau.c = ones(obj.n+1, 1);
        end
      elseif isstruct(parameters.c)
        obj = obj.setTimeVarrying(length(parameters.c.tauc) - 1);
        assert(numel(parameters.c.tauc) == obj.n+1);
        obj.tau.c = parameters.c.tauc;
        obj.c = parameters.c.ct;
      elseif size(parameters.c, 2) > 1
        obj = obj.setTimeVarrying(size(parameters.c, 2) - 1);
        obj.tau.c = [(1:obj.n) obj.n];
        obj.c = parameters.c;
      else
        obj.c = parameters.c;
        if ~obj.timeInvariant
          obj.tau.c = ones(obj.n+1, 1);
        end
      end
      
      % gamma system matrix
      if isempty(parameters.gamma)
        obj.gamma = zeros(obj.m, 0);
        if ~obj.timeInvariant
          obj.tau.gamma = ones(obj.n+1, 1);
        end
      elseif isstruct(parameters.gamma)
        obj = obj.setTimeVarrying(length(parameters.gamma.taugamma) - 1);
        assert(numel(parameters.gamma.taugamma) == obj.n+1);
        obj.tau.gamma = parameters.gamma.taugamma;
        obj.gamma = parameters.gamma.gammat;
      elseif size(parameters.gamma, 3) > 1
        obj = obj.setTimeVarrying(size(parameters.gamma, 3) - 1);
        obj.tau.gamma = [(1:obj.n) obj.n];
        obj.gamma = parameters.gamma;
      else
        obj.gamma = parameters.gamma;
        if ~obj.timeInvariant
          obj.tau.gamma = ones(obj.n+1, 1);
        end
      end
      obj.l = size(obj.gamma, 2);

      % Q system matrix
      if isstruct(parameters.Q)
        obj = obj.setTimeVarrying(length(parameters.Q.tauQ) - 1);
        assert(numel(parameters.Q.tauQ) == obj.n+1);
        obj.tau.Q = parameters.Q.tauQ;
        obj.Q = parameters.Q.Qt;
      else
        obj.Q = parameters.Q;
        if ~obj.timeInvariant
          obj.tau.Q = ones(obj.n+1, 1);
        end
      end
      obj.g = size(obj.Q, 1);
      
      % R system matrix
      if isempty(parameters.R)
        assert(obj.m == obj.g, ...
          'Shock dimension does not match state dimension with no R matrix specified.');
        obj.R = eye(obj.m);
        if ~obj.timeInvariant
          obj.tau.R = ones(obj.n+1, 1);
        end
      elseif isstruct(parameters.R)
        obj = obj.setTimeVarrying(length(parameters.R.tauR) - 1);
        assert(numel(parameters.R.tauR) == obj.n+1);
        obj.tau.R = parameters.R.tauR;
        obj.R = parameters.R.Rt;
      else
        obj.R = parameters.R;
        if ~obj.timeInvariant
          obj.tau.R = ones(obj.n+1, 1);
        end
      end
      
      if ~obj.timeInvariant
        tauDims = [length(obj.tau.Z) length(obj.tau.d) ...
          length(obj.tau.beta) length(obj.tau.H) ...
          length(obj.tau.T)-1 length(obj.tau.c)-1 length(obj.tau.gamma)-1 ...
          length(obj.tau.R)-1 length(obj.tau.Q)-1];
        assert(all(tauDims == obj.n));
      end
    end
    
    function obj = setTimeVarrying(obj, n)
      if obj.timeInvariant
        obj.timeInvariant = false;
        obj.n = n;
      else
        if obj.n ~= n
          error('MFSS:TVPmismatch', 'TVP calendar length mismatch.');
        end
      end
    end
    
    function obj = setInvariantTau(obj)
      % Set the tau structure for TVP.
      taus = [repmat({ones([obj.n 1])}, [4 1]); ...
        repmat({ones([obj.n+1 1])}, [5 1])];
      
      if ~isempty(obj.tau)
        cellTau = struct2cell(obj.tau);
        assert(max(cat(1, cellTau{:})) == 1, 'Existing tau not invariant.');
      end
      obj.tau = cell2struct(taus, obj.systemParam);
    end
  end
end
