classdef (Abstract) AbstractSystem
  % A class of general systems containing measurements and states.
  
  % David Kelley, 2016-2017
  
  properties (Dependent, Hidden)
    % Static properties
    % Indicator for use of compiled functions
    useMex
    % Indicator for use of parallel toolbox
    useParallel
  end
  
  properties (SetAccess = protected, Hidden)
    p % Number of observed series
    m % Number of states
    g % Number of shocks
    k % Number of exogenous measurement series
    l % Number of exogenous state series
    
    timeInvariant % Indicator for TVP models
  end
  
  properties (Hidden)
    % Observed time periods
    n
    % Logical vector for which states are stationary
    stationaryStates
  end
  
  methods
    function obj = AbstractSystem()
      % Constructor - empty, should do nothing
      if nargin == 0
        return
      end
    end
  end
  
  methods
    %% Getter/setter methods for static properties
    function use = get.useMex(obj)
      use = obj.getsetGlobalUseMex();
    end
    
    function obj = set.useMex(obj, use)
      obj.getsetGlobalUseMex(use);
    end
    
    function use = get.useParallel(obj)
      use = obj.getsetGlobalUseParallel();
    end
    
    function obj = set.useParallel(obj, use)
      obj.getsetGlobalUseParallel(use);
    end
  end
  
  methods (Static, Hidden)
    %% Methods to handle static properties
    function returnVal = getsetGlobalUseMex(newVal)
      % Static function to mimic a static class property of whether the mex
      % functions should be used (avoids overhead of checking for them every time)
      persistent useMex_persistent;
      
      % Setter
      if nargin > 0 && ~isempty(newVal)
        useMex_persistent = newVal;
      end
      
      % Default setter
      if isempty(useMex_persistent)
        % Check mex files exist
        mexMissing = any([...
          isempty(which('mfss_mex.filter_uni'));
          isempty(which('mfss_mex.smoother_uni'))]);
        if mexMissing
          useMex_persistent = false;
          warning('MEX files not found. See .\mex\make.m');
        else
          useMex_persistent = true;
        end
      end
      
      % Getter
      returnVal = useMex_persistent;
    end
    
    function returnVal = getsetGlobalUseParallel(newVal)
      % Static function to mimic a static class property of whether the mex
      % functions should be used (avoids overhead of checking for them every time)
      persistent useParallel_persistent;
      
      % Setter
      if nargin > 0 && ~isempty(newVal)
        useParallel_persistent = newVal;
      end
      
      % Default setter
      if isempty(useParallel_persistent)
        % Default to not using parallel since its often slower
        useParallel_persistent = false;
      end
      
      % Getter
      returnVal = useParallel_persistent;
    end
  end
  
  methods (Hidden)
    function returnFlag = checkConformingSystem(obj, sys)
      % Check if the dimensions of a system match the current object
      assert(isa(sys, 'AbstractSystem'));
      
      assert(obj.p == sys.p, 'Observation dimension mismatch (p).');
      assert(obj.m == sys.m, 'State dimension mismatch (m).');
      assert(obj.g == sys.g, 'Shock dimension mismatch (g).');
      if ~isempty(sys.k)
        assert(obj.k == sys.k, 'Exogenous measurement dimension mismatch (k).');
      end
      if ~isempty(sys.l)
        assert(obj.l == sys.l, 'Exogenous state dimension mismatch (l).');
      end
      
      assert(obj.timeInvariant == sys.timeInvariant, ...
        'Mismatch in time varrying parameters.');
      if ~obj.timeInvariant
        assert(obj.n == obj.n, 'Time dimension mismatch (n).');
      end
      
      returnFlag = true;
    end
  end
  
  methods (Static)
    %% General utility functions
    function mat = enforceSymmetric(mat)
      % Force a matrix to be symmetric. Corrects for small rounding errors in recursions.
      mat = 0.5 .* (mat + mat');
    end
  end
  
end