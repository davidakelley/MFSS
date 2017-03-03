classdef (Abstract) AbstractStateSpace < AbstractSystem
  % Abstract state space for accumulators and utilities 
  % Provides the structural parts of the parameter manipulation and some
  % utility functions.
  %
  % Two current subclasses: StateSpace (for systems with known parameters) and
  % StateSpaceEstimation (to estimate unknown parameters of a system).
  
  % David Kelley, 2016-2017

  properties
    Z, d, H           % Observation equation parameters
    T, c, R, Q        % State equation parameters
    tau               % Structure of time-varrying parameter indexes
    
    filterUni         % Use univarite filter if appropriate (H is diagonal)
    
    collapse          % Boolean option to collapse observation vector
  end
  
  properties (SetAccess = protected)
    % Initial values parameter protected. Use setInitial. 
    a0                % Initial state parameter
    Q0                % Exact initial value parameters
  end
  
  properties (SetAccess = protected, Hidden)
    A0, R0            % Initial value selection matricies
    
    % Indicators for if initial values have been specified 
    % 2/3: Why do I need these? Analytic gradient?
    usingDefaulta0 = true;
    usingDefaultP0 = true;

    % Lists of variables used across methods
    systemParam = {'Z', 'd', 'H', 'T', 'c', 'R', 'Q'};
    symmetricParams = {'H', 'Q', 'Q0', 'P'};    % Include P0? P?
  end
  
  methods
    %% Constructor
    function obj = AbstractStateSpace(Z, d, H, T, c, R, Q)
      % Constructor
      % Inputs: Parameters matricies or a structure of parameters
      if nargin == 1
        % Structure of parameter values was passed, contains all parameters
        parameters = Z;
      elseif nargin == 7
        parameters = obj.compileStruct(Z, d, H, T, c, R, Q);
      else
        error('Input error.');
      end
      
      obj = obj.setSystemParameters(parameters);
      
      % Check if we can use the univariate filter
      slicesH = num2cell(obj.H, [1 2]);
      obj.filterUni = ~any(~cellfun(@isdiag, slicesH));
      obj.collapse = false; %obj.p > obj.m;
    end
    
    %% Initialization
    function obj = setInitial(obj, a0, P0)
      % Setter for a0 and P0 (or kappa)
      % Takes user input for a0 and P0. Constructs A0 and R0 matricies based on
      % if elements of P0 are finite or not (nan or Inf both work to denote an
      % initialization as diffuse). 
      
      % Set a0 - still put diffuse elements in a0 since they'll be ignored in 
      % the exact initial filter and are needed in the approximate filter. 
      if ~isempty(a0)
        obj.usingDefaulta0 = false;

        assert(size(a0, 1) == obj.m, 'a0 should be a m X 1 vector');
        obj.a0 = a0;
      end
      
      % Set A0, R0 and Q0.
      if nargin > 2 && ~isempty(P0)
        obj.usingDefaultP0 = false;
        % Handle input options for P0
        if size(P0, 1) == 1 
          % Scalar value passed for kappa
          % (Note that this is ok evn if m == 1)
          P0 = eye(obj.m) * P0;
        elseif size(P0, 2) == 1
          % Vector value passed for kappa
          assert(size(P0, 1) == obj.m, 'kappa vector must be length m');
          P0 = diag(P0);
        else
          % Regular P0 value passed
          assert(all(size(P0) == [obj.m obj.m]), 'P0 should be a m X m matrix');
        end
        
        % Find diffuse elements
        diffuse = any(isinf(P0), 2);
        nondiffuse = all(~isinf(P0), 2);
        
        % Set up exact initial parameters
        select = eye(obj.m);
        obj.A0 = select(:, diffuse);
        obj.R0 = select(:, nondiffuse);
        obj.Q0 = P0(nondiffuse, nondiffuse);
      end
    end
    
    %% Utility methods
    function param = parameters(obj, index)
      % Getter for cell array of parameters
      param = {obj.Z, obj.d, obj.H, obj.T, obj.c, obj.R, obj.Q, obj.a0, obj.Q0}; 
      if nargin > 1
        param = param{index};
      end
    end
    
    function vec = vectorizedParameters(obj)
      param = obj.parameters;
      if obj.usingDefaulta0
        param{8} = [];
      end
      if obj.usingDefaultP0
        param{9} = [];
      end
      
      vectors = cellfun(@(x) x(:), param, 'Uniform', false);
      vec = vertcat(vectors{:});
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
      
      criteriaFn = @(interest) all([(obj.T(:, interest, obj.tau.T(1)) == 1) ...
        all(obj.T(:,((1:size(obj.T,1)) ~= interest), obj.tau.T(1)) == 0,2) ...
        all(obj.R(:, :, obj.tau.R(1)) == 0, 2)], 2);
      
      nLags = 0;
      positions = [];
      while any(criteriaFn(varPos))
        nLags = nLags + 1;
        varPos = find(criteriaFn(varPos));
        positions = [positions; varPos]; %#ok<AGROW>
      end
    end
    
  end
  
  methods (Hidden = true)
    %% Internal helpers    
    function obj = setQ0(obj, Q0)
      % Utility function for setting Q0 without altering A0, R0.
      % Useful for setting bounds matricies of ThetaMap
      obj.Q0 = Q0;
      obj.usingDefaultP0 = false;
    end
    
    %% Constructor helper methods
    function obj = setSystemParameters(obj, parameters)
      % Obtain system matrices from structure
      % By passing both the different types of system matrices and the calendar
      % vector corresponding to the particular type used at any particular time
      % we allow for an arbitrary number of structural "breaks" in the state
      % space model. Furthermore, by allowing each system matrix/vector to have
      % its own calendar, no redundant matrices are saved in the workspace.
      
      structParams = structfun(@isstruct, parameters);
      
      if ~any(structParams)
        obj.timeInvariant = true;
      else
        obj.timeInvariant = false;
      
        tauLens = cellfun(@(x) length(parameters.(x).(['tau' x])), ...
        obj.systemParam(structParams));
      
        subtract = cellfun(@(x) any(strcmpi(x, {'T', 'c', 'R', 'Q'})), ...
          obj.systemParam(structParams));
        nCandidates = tauLens - subtract;
        assert(isscalar(unique(nCandidates)), ['Bad tau specification. ' ... 
          'tau vectors for Z, d & H should have n elements. ' ...
          'tau vectors for T, c, R & Q should have n+1 elements.']);
        obj.n = unique(nCandidates);
      end
      
      % Z system matrix
      if isstruct(parameters.Z)
        obj = obj.setTimeVarrying(length(parameters.Z.tauZ));
        assert(numel(parameters.Z.tauZ) == obj.n);
        obj.tau.Z = reshape(parameters.Z.tauZ, obj.n, 1);
        obj.Z = parameters.Z.Zt;
      else
        obj.Z = parameters.Z;
        if ~obj.timeInvariant
          obj.tau.Z = ones(obj.n, 1);
        end
      end
      
      % d system matrix
      if isstruct(parameters.d)
        obj = obj.setTimeVarrying(length(parameters.d.taud));
        assert(numel(parameters.d.taud) == obj.n);
        obj.tau.d = reshape(parameters.d.taud, obj.n, 1);
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
      
      % H system matrix
      if isstruct(parameters.H)
        obj = obj.setTimeVarrying(length(parameters.H.tauH));
        assert(numel(parameters.H.tauH) == obj.n);
        obj.tau.H = reshape(parameters.H.tauH, obj.n, 1);
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
      if isstruct(parameters.c)
        obj = obj.setTimeVarrying(length(parameters.c.tauc) - 1);
        assert(numel(parameters.c.tauc) == obj.n+1);
        obj.tau.c = reshape(parameters.c.tauc, obj.n+1, 1);
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
      
      % R system matrix
      if isstruct(parameters.R)
        obj = obj.setTimeVarrying(length(parameters.R.tauR) - 1);
        assert(numel(parameters.R.tauR) == obj.n+1);
        obj.tau.R = reshape(parameters.R.tauR, obj.n+1, 1);
        obj.R = parameters.R.Rt;
      else
        obj.R = parameters.R;
        if ~obj.timeInvariant
          obj.tau.R = ones(obj.n+1, 1);
        end
      end
      
      % Q system matrix
      if isstruct(parameters.Q)
        obj = obj.setTimeVarrying(length(parameters.Q.tauQ) - 1);
        assert(numel(parameters.Q.tauQ) == obj.n+1);
        obj.tau.Q = reshape(parameters.Q.tauQ, obj.n+1, 1);
        obj.Q = parameters.Q.Qt;
      else
        obj.Q = parameters.Q;
        if ~obj.timeInvariant
          obj.tau.Q = ones(obj.n+1, 1);
        end
      end
      
      if ~obj.timeInvariant
        tauDims = [length(obj.tau.Z) length(obj.tau.d) length(obj.tau.H) ...
          length(obj.tau.T)-1 length(obj.tau.c)-1 ...
          length(obj.tau.R)-1 length(obj.tau.Q)-1];
        assert(all(tauDims == obj.n));
      end
      
      % Set dimensions:
      %   p - number of observables
      %   m - number of state variables
      %   g - number of shocks
      obj.p = size(obj.Z(:,:,end), 1);
      [obj.m, obj.g] = size(obj.R(:,:,end));
    end
    
    function obj = setTimeVarrying(obj, n)
      if obj.timeInvariant
        obj.timeInvariant = false;
        obj.n = n;
      else
        assert(obj.n == n, 'TVP calendar length mismatch.');
      end
    end
    
    function obj = setInvariantTau(obj)
      % Set the tau structure for TVP.
      taus = [repmat({ones([obj.n 1])}, [3 1]); ...
        repmat({ones([obj.n+1 1])}, [4 1])];
      
      if ~isempty(obj.tau)
        cellTau = struct2cell(obj.tau);
        assert(max(cat(1, cellTau{:})) == 1, 'Existing tau not invariant.');
      end
      obj.tau = cell2struct(taus, obj.systemParam);
    end
  end
end
