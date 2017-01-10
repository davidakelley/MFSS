classdef (Abstract) AbstractSystem
  % A class of general systems containing measurements and states. 
    
  % David Kelley, 2016
  
  properties (SetAccess = protected, Hidden)
    p               % Number of observed series
    m               % Number of states
    g               % Number of shocks
        
    timeInvariant   % Indicator for TVP models
  end
  
  properties (Hidden)
    n               % Observed time periods
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
    function sOut = compileStruct(varargin)
      % Combines variables passed as arguments into a struct
      % 
      % struct = compileStruct(a, b, c) will place the variables a, b, & c
      % in the output variable using the variable names as the field names.
      
      sOut = struct;
      for iV = 1:nargin
        sOut.(inputname(iV)) = varargin{iV};
      end
    end
    
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
    
    function K = genCommutation(m)
      % Generate commutation matrix
      % 
      % K = genCommutation(m) returns a commutation matrix for a
      % square matrix A of size m such that K * vec(A) = vec(A'). 
      
      % From Magnus & Neudecker (1979), Definition 3.1, a commutation matrix is 
      % "a suqare mn-dimensional matrix partitioned into mn sub-matricies of 
      % order (n, m) such that the ij-th submatrix has a 1 in its ji-th position 
      % and zeros elsewhere."
      
      E = @(i, j) [zeros(i-1, m); ...
        zeros(1, j-1), 1, zeros(1, m-j); zeros(m-i, m)];
      K = zeros(m^2);
      for iComm = 1:m
        for jComm = 1:m
          K = K + kron(E(iComm, jComm), E(iComm, jComm)');
        end
      end
    end
    
  end
end