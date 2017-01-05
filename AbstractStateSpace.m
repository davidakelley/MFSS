classdef (Abstract) AbstractStateSpace < AbstractSystem
  % Abstract state space for accumulators and utilities 
  % Provides the structural parts of the parameter manipulation and some
  % utility functions.
  %
  % Two current subclasses: StateSpace (for systems with known parameters) and
  % StateSpaceEstimation (to estimate unknown parameters of a system).
  
  % David Kelley, 2016
  % 
  % TODO (12/9/16)
  % ---------------
  %   - Which tau should be used for a0 and P0?
  
  properties
    Z, d, H           % Observation equation parameters
    T, c, R, Q        % State equation parameters
    tau               % Structure of time-varrying parameter indexes
    
    a0, P0            % Initial value parameters
    kappa = 1e6;      % Diffuse initialization constant
    
    % Indicators for if initial values have been specified:
    usingDefaulta0 = true;
    usingDefaultP0 = true;
  end
  
  properties (SetAccess = protected, Hidden)
    % Lists of variables used across methods
    systemParam = {'Z', 'd', 'H', 'T', 'c', 'R', 'Q', 'a0', 'P0'};
    symmetricParams = {'H', 'Q', 'P0', 'P'};
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
    end
    
    %% Initialization
    function obj = setInitial(obj, a0, P0)
      % Setter for a0 and P0 (or kappa)
      if ~isempty(a0)
        assert(size(a0, 1) == obj.m, 'a0 should be a m X 1 vector');
        obj.usingDefaulta0 = false;
        obj.a0 = a0;
      end
      
      if nargin > 2 && ~isempty(P0)
        obj.usingDefaultP0 = false;
        if size(P0, 1) == 1 % (This is ok if m == 1 too)
          % Scalar value passed for kappa
          obj.P0 = eye(obj.m) * P0;
        elseif size(P0, 2) == 1
          % Vector value passed for kappa
          assert(size(P0, 1) == obj.m, 'kappa vector must be length m');
          obj.P0 = diag(P0);
        else
          % Regular P0 value passed
          assert(all(size(P0) == [obj.m obj.m]), 'P0 should be a m X m matrix');
          obj.P0 = P0;
        end
      end
    end
    
    %% Utility methods
    function param = parameters(obj, index)
      % Getter for cell array of parameters
      param = {obj.Z, obj.d, obj.H, obj.T, obj.c, obj.R, obj.Q, obj.a0, obj.P0}; 
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

    function setSS = setAllParameters(obj, value)
      % Set all parameter values equal to the scalar provided
      setSS = obj;
      for iP = 1:length(setSS.systemParam)
        setSS.(setSS.systemParam{iP})(:) = value;
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
    %% Constructor helper methods
    function obj = setSystemParameters(obj, parameters)
      % Obtain system matrices from structure
      % By passing both the different types of system matrices and the calendar
      % vector corresponding to the particular type used at any particular time
      % we allow for an arbitrary number of structural "breaks" in the state
      % space model. Furthermore, by allowing each system matrix/vector to have
      % its own calendar, no redundant matrices are saved in the workspace.
      
      % Assume system is time invariant, correct when we get a TVP matrix
      obj.timeInvariant = true;
      
      % Z system matrix
      if isstruct(parameters.Z)
        obj = obj.setTimeVarrying(length(parameters.Z.tauZ));
        assert(numel(parameters.Z.tauZ) == obj.n);
        obj.tau.Z = reshape(parameters.Z.tauZ, obj.n, 1);
        obj.Z = parameters.Z.Zt;
      else
        obj.Z = parameters.Z;
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
      end
      
      % H system matrix
      if isstruct(parameters.H)
        obj = obj.setTimeVarrying(length(parameters.H.tauH));
        assert(numel(parameters.H.tauH) == obj.n);
        obj.tau.H = reshape(parameters.H.tauH, obj.n, 1);
        obj.H = parameters.H.Ht;
      else
        obj.H = parameters.H;
      end
      
      % T system matrix
      if isstruct(parameters.T)
        obj = obj.setTimeVarrying(length(parameters.T.tauT) - 1);
        assert(numel(parameters.T.tauT) == obj.n+1);
        obj.tau.T = reshape(parameters.T.tauT, obj.n+1, 1);
        obj.T = parameters.T.Tt;
      else
        obj.T = parameters.T;
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
      end
      
      % R system matrix
      if isstruct(parameters.R)
        obj = obj.setTimeVarrying(length(parameters.R.tauR) - 1);
        assert(numel(parameters.R.tauR) == obj.n+1);
        obj.tau.R = reshape(parameters.R.tauR, obj.n+1, 1);
        obj.R = parameters.R.Rt;
      else
        obj.R = parameters.R;
      end
      
      % Q system matrix
      if isstruct(parameters.Q)
        obj = obj.setTimeVarrying(length(parameters.Q.tauQ) - 1);
        assert(numel(parameters.Q.tauQ) == obj.n+1);
        obj.tau.Q = reshape(parameters.Q.tauQ, obj.n+1, 1);
        obj.Q = parameters.Q.Qt;
      else
        obj.Q = parameters.Q;
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
        warning('tau already set.');
      end
      obj.tau = cell2struct(taus, obj.systemParam(1:7));
    end
  end
end
