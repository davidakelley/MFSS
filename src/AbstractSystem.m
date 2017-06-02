classdef (Abstract) AbstractSystem
  % A class of general systems containing measurements and states.
  
  % David Kelley, 2016-2017
  
  properties (Dependent)
    % Static Properties
    
    % Indicator for use of compiled functions
    useMex    
    
    % Indicator for use of parallel toolbox
    useParallel
  end
  
  properties (SetAccess = protected, Hidden)
    p % Number of observed series
    m % Number of states
    g % Number of shocks
    
    timeInvariant % Indicator for TVP models
  end
  
  properties (Hidden)
    n % Observed time periods
    
    stationaryStates % Logical vector for which states are stationary
  end
  
  methods
    %% Constructor
    % Empty - should do nothing
    function obj = AbstractSystem()
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
      use = obj.getsetGlobalUseMex();
    end
    
    function obj = set.useParallel(obj, use)
      obj.getsetGlobalUseMex(use);
    end
  end
  
  methods (Static, Hidden)
    %% Method to handle static property
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
    function [Finv, logDetF] = pseudoinv(F, tol)
      % Returns the pseudo-inverse and log determinent of F
      %
      % [Finv, logDetF] = pseudoinv(F, tol) finds the inverse and log
      % determinent of F. Elements of the SVD of F less than tol are taken as 0.
      
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
    
    function K = genCommutation(m, n)
      % Generate commutation matrix
      %
      % K = genCommutation(m, n) returns a commutation matrix for an m X n
      % matrix A such that K * vec(A) = vec(A').
      
      % From Magnus & Neudecker (1979) (Definition 3.1): a commutation matrix is
      % "a suqare mn-dimensional matrix partitioned into mn sub-matricies of
      % order (n, m) such that the ij-th submatrix has a 1 in its ji-th position
      % and zeros elsewhere."
      if nargin == 1, n = m; end
      
      E = @(i, j) [zeros(i-1, n); ...
        zeros(1, j-1), 1, zeros(1, n-j); zeros(m-i, n)];
      K = zeros(m * n);
      for iComm = 1:m
        for jComm = 1:n
          K = K + kron(E(iComm, jComm), E(iComm, jComm)');
        end
      end
    end
    
  end
end