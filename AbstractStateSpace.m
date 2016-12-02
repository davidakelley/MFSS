classdef (Abstract) AbstractStateSpace
  % Abstract state space for accumulators and utilities 
  % Provides the structural parts of the parameter manipulation and some
  % utility functions.
  %
  % Two current subclasses: StateSpace (for systems with known parameters) and
  % StateSpaceEstimation (to estimate unknown parameters of a system).
  
  % David Kelley, 2016
  % 
  % TODO (11/30/16)
  % ---------------
  %   - Create utiltiy methods for standard accumulator creation (descriptive
  %   specification as opposed to explicitly stating phi/Horizon values).

  properties
    Z, d, H           % Observation equation parameters
    T, c, R, Q        % State equation parameters
    accumulator       % Structure defining accumulated series
    tau               % Structure of time-varrying parameter indexes
    
    a0, P0            % Initial value parameters
    kappa = 1e6;      % Diffuse initialization constant
    
    % Indicators for if initial values have been specified:
    usingDefaulta0 = true;
    usingDefaultP0 = true;
  end
  
  properties(SetAccess = protected, Hidden)
    % Dimensions
    p                 % Number of observed series
    m                 % Number of states
    g                 % Number of shocks
    n                 % Time periods
    
    % General options
    systemParam = {'Z', 'd', 'H', 'T', 'c', 'R', 'Q', 'a0', 'P0'};
    symmetricParams = {'H', 'Q', 'P0', 'P'};
    timeInvariant     % Indicator for TVP models
  end
  
  methods
    %% Constructor
    function obj = AbstractStateSpace(Z, d, H, T, c, R, Q, accumulator)
      % Constructor
      % Inputs: Parameters matricies or a structure of parameters
      if nargin == 1
        % Structure of parameter values was passed, contains all parameters
        accumulator = Z.accumulator;
        parameters = rmfield(Z, 'accumulator');
      elseif nargin == 7
        parameters = obj.compileStruct(Z, d, H, T, c, R, Q);
        accumulator = [];
      elseif nargin >= 8
        parameters = obj.compileStruct(Z, d, H, T, c, R, Q);
      else
        error('Input error.');
      end
      
      obj = obj.setSystemParameters(parameters);
      obj = obj.addAccumulators(accumulator);
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
      vectors = cellfun(@(x) x(:), param, 'Uniform', false);
      vec = vertcat(vectors{:});
    end

    function returnFlag = checkConformingSystem(obj, ss)
      % Check if the dimensions of a state space match the current object
      
      assert(isa(ss, 'AbstractStateSpace') || isa(ss, 'ThetaMap'));
      assert(obj.p == ss.p, 'Observation dimension mismatch (p).');
      assert(obj.m == ss.m, 'State dimension mismatch (m).');
      assert(obj.g == ss.g, 'Shock dimension mismatch (g).');
      assert(obj.timeInvariant == ss.timeInvariant, ...
        'Mismatch in time varrying parameters.');
      if ~obj.timeInvariant
        assert(obj.n == obj.n, 'Time dimension mismatch (n).');
      end
      
      assert(obj.usingDefaulta0 == ss.usingDefaulta0, ...
        'Mismatch in handling of a0.');
      assert(obj.usingDefaultP0 == ss.usingDefaultP0, ...
        'Mismatch in handling of P0.');
      
      returnFlag = true;
    end
    
    function setSS = setAllParameters(obj, value)
      % Set all parameter values equal to the scalar provided
      setSS = obj;
      for iP = 1:length(setSS.systemParam)
        setSS.(setSS.systemParam{iP})(:) = value;
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
        obj.tau.Z = parameters.Z.tauZ;
        obj.Z = parameters.Z.Zt;
      else
        obj.Z = parameters.Z;
      end
      
      % d system matrix
      if isstruct(parameters.d)
        obj = obj.setTimeVarrying(length(parameters.d.taud));
        obj.tau.d = parameters.d.taud;
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
        obj.tau.H = parameters.H.tauH;
        obj.H = parameters.H.Ht;
      else
        obj.H = parameters.H;
      end
      
      % T system matrix
      if isstruct(obj.T)
        obj = obj.setTimeVarrying(length(parameters.T.tauT) - 1);
        obj.tau.T = parameters.T.tauT;
        obj.T = parameters.T.Tt;
      else
        obj.T = parameters.T;
      end
      
      % c system matrix
      if isstruct(parameters.c)
        obj = obj.setTimeVarrying(length(parameters.c.tauc) - 1);
        obj.tau.c = parameters.c.tauc;
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
        obj.tau.R = parameters.R.tauR;
        obj.R = parameters.R.Rt;
      else
        obj.R = parameters.R;
      end
      
      % Q system matrix
      if isstruct(parameters.Q)
        obj = obj.setTimeVarrying(length(parameters.Q.tauQ) - 1);
        obj.tau.Q = parameters.Q.tauQ;
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
      % FIXME: obj.tau shouldn't be set if it's already defined
      taus = [repmat({ones([1 obj.n])}, [3 1]); ...
        repmat({ones([1 obj.n+1])}, [4 1])];
      obj.tau = cell2struct(taus, obj.systemParam(1:7));
    end
    
    %% Accumulator helper methods
    function obj = addAccumulators(obj, accumulator)
      % Augment system parameters with Harvey accumulators
      %
      % Augment Standard State Space with Harvey Accumulator
      % $\xi$  - $\xi_{h}$ indice vector of each series accumulated
      % $A$    - selection matrix
      % $\psi$ - full history of accumulator indicator (for each series)
      % $\tau$ - number of unique accumulator types
      %        - Two Cases:'flow'    = 1 Standard Flow variable;
      %                    'average' = 0 Time-averaged stock variable
      
      % Based on work by Andrew Butters, 2013
      
      if isempty(accumulator)
        return
      else
        obj.accumulator = accumulator;
      end
      
      % Validate input
      if all(ismember(fieldnames(accumulator), {'xi', 'psi', 'Horizon'}))
        accum.index = accumulator.xi;
        accum.calendar = accumulator.psi;
        accum.horizon = accumulator.Horizon;
        accumulator = accum;
      end
      if islogical(accumulator.index)
        accumulator.index = find(accumulator.index);
      end
      assert(all(ismember(fieldnames(accumulator), {'index', 'calendar', 'horizon'})), ...
        'accumulator must be a structure with fields index, calendar, and horizon');
      assert(length(accumulator.index) == size(accumulator.calendar, 1));
      assert(length(accumulator.index) == size(accumulator.horizon, 1));
      
      accumIndex = accumulator.index;
      calendar = accumulator.calendar;
      horizon = accumulator.horizon;
      
      % Set tau
      if isempty(obj.tau)
        obj.n = size(calendar, 2)-1;
        obj = obj.setInvariantTau();
      end
      
      accumTypes = any((calendar==0),2);
      [accumObs, accumStates] = find(any(obj.Z(accumIndex,:,:) ~= 0, 3));
      
      [states, AddLags, ColAdd, RowPos, mNew] = obj.accumAugmentedStateDims(horizon, accumObs, accumStates);
      
      if sum(AddLags)==0
        newZ0 = obj.Z;
        newT0 = obj.T;
        newc0 = obj.c;
        newR0 = obj.R;
      else
        newZ0 = zeros(obj.p, obj.m + sum(AddLags), obj.n);
        newZ0(:, 1:obj.m,:) = obj.Z;
        newT0 = zeros(obj.m + sum(AddLags), obj.m + sum(AddLags), obj.n);
        newT0(1:obj.m, 1:obj.m,:) = Tt;
        newT0(sub2ind(size(newT0), (obj.m + 1:mNew)', reshape(ColAdd(ColAdd>0),[],1))) = 1;
        newc0 = zeros(obj.m + sum(AddLags), obj.n);
        newc0(1:obj.m,:) = ct;
        newR0 = zeros(obj.m + sum(AddLags), obj.g, obj.n);
        newR0(1:obj.m, :, :) = Rt;
        obj.m = mNew;
      end
      
      % Indentify unique elements of the state that need accumulation
      nonZeroZpos = zeros(length(accumIndex), obj.m);
      [APsi, ~, nonZeroZpos(sub2ind([length(accumIndex) obj.m], accumObs, accumStates))] ...
        = unique([accumStates accumTypes(accumObs) horizon(accumObs, :) calendar(accumObs, :)], 'rows');
      
      nTau  = size(APsi, 1);
      st = APsi(:, 1);
      accumTypes = APsi(:, 2);
      Hor = APsi(:, 3:size(horizon,2) + 2);
      Psi = APsi(:, size(horizon,2) + 3:end);
      
      mOld = obj.m;
      obj.m = obj.m + nTau;
      
      % Construct new T matrix
      Ttypes   = [obj.tau.T' Psi' Hor'];
      [uniqueTs,~, newtauT] = unique(Ttypes,'rows');
      NumT = size(uniqueTs, 1);
      newT = zeros(obj.m, obj.m, NumT);
      for jj = 1:NumT
        newT(1:mOld, :, jj) = [newT0(:,:,uniqueTs(jj,1)) zeros(mOld, nTau)];
        for h = 1:nTau
          if accumTypes(h) == 1
            newT(mOld + h, [1:mOld mOld + h],jj) = ...
              [newT0(APsi(h,1),:,uniqueTs(jj,1)) ...
              uniqueTs(jj,1+h)];
          else
            newT(mOld+h, [1:mOld mOld+h],jj) =...
              [(1/uniqueTs(jj,1+h)) * newT0(APsi(h,1),:,uniqueTs(jj,1))...
              (uniqueTs(jj,1+h)-1)/uniqueTs(jj,1+h)];
            if uniqueTs(jj, nTau+1+h) > 1
              cols = RowPos(st(h)==states, 1:uniqueTs(jj,nTau+1+h)-1);
              newT(mOld+h, cols, jj) = newT(mOld+h,cols,jj) + (1/uniqueTs(jj,1+h));
            end
          end
        end
      end
      obj.T = newT;
      obj.tau.T = newtauT;
      
      % Construct new c vector
      ctypes   = [obj.tau.c' Psi'];
      [uniquecs, ~, newtauc] = unique(ctypes,'rows');
      Numc = size(uniquecs, 1);
      newc = zeros(obj.m, Numc);
      for jj = 1:Numc
        newc(1:mOld, jj) = newc0(:, uniquecs(jj,1));
        for h = 1:nTau
          if accumTypes(h) == 1
            newc(mOld+h, jj) = newc0(APsi(h,1), uniquecs(jj,1));
          else
            newc(mOld+h, jj) = (1/uniquecs(jj,1+h))*...
              newc0(APsi(h,1), uniquecs(jj,1));
          end
        end
      end
      obj.c = newc;
      obj.tau.c = newtauc;
      
      % Construct new R matrix
      Rtypes   = [obj.tau.R' Psi'];
      [uniqueRs, ~, newtauR] = unique(Rtypes,'rows');
      NumR = size(uniqueRs, 1);
      newR = zeros(obj.m, obj.g, NumR);
      for jj = 1:NumR
        newR(1:mOld,1:obj.g,jj) = newR0(:, :, uniqueRs(jj,1));
        for h = 1:nTau
          if accumTypes(h) == 1
            newR(mOld+h,:,jj) = newR0(APsi(h,1), :, uniqueRs(jj,1));
          else
            newR(mOld+h,:,jj) = (1/uniqueRs(jj, 1+h))*...
              newR0(APsi(h,1), :, uniqueRs(jj,1));
          end
        end
      end
      obj.R = newR;
      obj.tau.R = newtauR;
      
      % Construct new Z matrix
      newZ = zeros(obj.p, obj.m, size(obj.Z,3));
      for jj = 1:size(obj.Z, 3)
        newZ(:,1:mOld,jj) = newZ0(:,:,jj);
        newZ(accumIndex,:,jj) = zeros(length(accumIndex), obj.m);
        for h = 1:length(accumIndex)
          cols = find(newZ0(accumIndex(h),:,jj));
          newZ(accumIndex(h), mOld + nonZeroZpos(h,cols),jj) = newZ0(accumIndex(h),cols,jj);
        end
      end
      obj.Z = newZ;
    end
    
    function [states, AddLags, ColAdd, RowPos, mNew] = ...
        accumAugmentedStateDims(obj, horizon, accumObs, accumStates)
      % Checking we have the correct number of lags of the states that need
      % accunulation to be compatible with Horizon
      states = unique(accumStates);
      maxHor = max(horizon, [], 2);
      AddLags = zeros(length(states), 1);
      ColAdd  = zeros(length(states), max(maxHor));
      RowPos  = zeros(length(states), max(maxHor));
      mNew = obj.m;
      for jj=1:length(states)
        mHor = max(maxHor(accumObs(states(jj) == accumStates)));
        [numlagsjj,lagposjj] = obj.LagsInState(states(jj));
        
        AddLags(jj) = max(mHor-1-numlagsjj-1,0);
        RowPos(jj,1:numlagsjj+1) = [states(jj) lagposjj'];
        if AddLags(jj) > 0
          ColAdd(jj,numlagsjj+2) = lagposjj(end);
          if AddLags(jj)>1
            ColAdd(jj,numlagsjj+3:numlagsjj+1+AddLags(jj)) = mNew:mNew+AddLags(jj)-1;
          end
        end
        RowPos(jj,numlagsjj+2:numlagsjj+1+AddLags(jj)) = mNew+1:mNew+AddLags(jj);
        mNew = mNew+AddLags(jj);
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
    
    %% General utility functions
    function sOut = compileStruct(obj, varargin) %#ok<INUSL>
      % Combines variables passed as arguments into a struct
      % struct = compileStruct(a, b, c) will place the variables a, b, & c
      % in the output variable using the variable names as the field names.
      sOut = struct;
      for iV = 1:nargin-1
        sOut.(inputname(iV+1)) = varargin{iV};
      end
    end
    
    function [Finv, logDetF] = pseudoinv(obj, F, tol) %#ok<INUSL>
      tempF = 0.5 * (F + F');
      
      [PSVD, DSDV, PSVDinv] = svd(tempF);
      
      % Truncate small singular values
      firstZero = find(diag(DSDV) < tol, 1, 'first');
      
      if ~isempty(firstZero)
        PSVD = PSVD(:, 1:firstZero - 1);
        PSVDinv = PSVDinv(:, 1:firstZero - 1);
        DSDV = DSDV(1:firstZero - 1, 1:firstZero - 1);
      end
      
      Finv = PSVD * (DSDV\eye(length(DSDV))) * PSVDinv';
      logDetF = sum(log(diag(DSDV)));
    end
    
  end
end
