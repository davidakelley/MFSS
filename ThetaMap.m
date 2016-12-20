classdef ThetaMap < AbstractSystem
  % ThetaMap - Mapping from a vector of parameters to a StateSpace 
  %
  % A vector theta will be used to construct a StateSpace. Each non-fixed scalar
  % parameter of the StateSpace is allowed to be a function of one element of
  % the theta vector. Multiple parameter values may be based off of the same
  % element of theta.
  %
  % There are two primary uses of a ThetaMap:
  %   theta2system: Creates a new StateSpace from a theta vector
  %   system2theta: Computes the theta vector that created the StateSpace
  % 
  % The state space parameters can also be restricted to lie between an upper
  % and lower bound set using ThetaMap.addRestrictions. 
  % 
  % When estimating unknown values in a StateSpace, a ThetaMap can be generated
  % where all free parameters will be determined by an independent element of
  % theta (except in variance matricies that must be symmetric) - see the
  % ThetaMapEstimation constructor.
  % 
  % In the internals of a ThetaMap, there are five primary components:
  %   - fixed: A StateSpace where each of the fixed values is entered into the
  %   corresponding parameter.
  %   - index: A StateSpace object of integer indexes. Fixed elements must be
  %   denoted with a zero. Elements that depend on theta can have an integer
  %   value between 1 and the length of theta, indicating the element of theta
  %   that will be used to construct that element.
  %   - transformations: A cell array of function handles. Each function should
  %   take a scalar input and return a scalar output.
  %   - derivatives: A cell array of derivatives for each transformation.
  %   - inverses: A cell array of inverses for each transformation.
  %   - transformationIndex: A StateSpace object of integer indexes. Fixed
  %   elements must be denoted with a zero. Elements that depend on theta can
  %   have an integer value between 1 and the total number of transformations
  %   provided, indicating which function will be used to transform the value of
  %   theta(i) before placing it in the appropriate parameter matrix.
  
  % David Kelley, 2016
  %
  % TODO (12/12/2016)
  % ---------------
  %   - Write checkThetaMap or write setters for most properties
  %   - Write a test class specifically for ThetaMap. It should verify: 
  %     - We can go system -> theta -> system and get the same thing both when
  %       a0 and P0 are defined and when they're not.
  %     - Check a generated system obeys bounds. 
  %     - Creating a system with accumulators is the same as generating a system
  %       without accumulators and adding the accumulators in afterward. 
  %     - checkThetaMap purpose: Can we eliminate an element from the map and
  %       get a reduced size theta?
  %   - Move initialValuesGradients to StateSpace?
  %   - Optional: Allow inverses to be passed empty and do the line-search 
  %     to find the numeric inverse. 
  %   - Write documentation on initial values derivation
  
  properties (SetAccess = protected)
    % Number of elements in theta vector
    nTheta            

    % StateSpace containing all fixed elements
    fixed
    
    % StateSpace of indexes of theta that determines parameter values
    index
    
    % Cell array of transformations of theta to get parameter values
    transformations
    % Derivatives of transformations
    derivatives
    % Inverses of transformations
    inverses
    % StateSpace of indexes of transformations applied for parameter values
    transformationIndex
    
    % Parameter bounds
    LowerBound
    UpperBound
  end
  
  properties (SetAccess = protected, Hidden)
    usingDefaulta0
    usingDefaultP0
  end
  
  methods
    %% Constructor
    function obj = ThetaMap(fixed, index, ...
        transformations, derivatives, inverses, transformationIndex)
      % Generate map from elements of theta to StateSpace parameters
      
      % Make sure dimensions match
      index.checkConformingSystem(transformationIndex);

      vectorizedTransformParams = cellfun(@(x) (x(:))', ...
        transformationIndex.parameters, 'Uniform', false);
      nTransform = max([vectorizedTransformParams{:}]);
      
      assert(length(transformations) >= nTransform, ...
        'Insufficient transformations provided for indexes given.');
      assert(length(transformations) == length(derivatives), ...
        'Derivatives must be provided for every transformation.');
      assert(length(inverses) == length(derivatives), ...
        'Inverses must be provided for every transformation.');
      
      % Set properties
      assert(isa(fixed, 'AbstractStateSpace'));
      obj.fixed = fixed;
      assert(isa(index, 'StateSpace'));
      obj.index = index;
      obj.transformations = transformations;
      obj.derivatives = derivatives;
      obj.inverses = inverses;
      assert(isa(transformationIndex, 'StateSpace'));
      obj.transformationIndex = transformationIndex;
      
      % Set dimensions
      vectorizedIndexParams = cellfun(@(x) (x(:))', ...
        index.parameters, 'Uniform', false);
      obj.nTheta = max([vectorizedIndexParams{:}]);
      
      obj.p = index.p;
      obj.m = index.m;
      obj.g = index.g;
      obj.n = index.n;
      obj.timeInvariant = index.timeInvariant;
      
      obj.usingDefaulta0 = index.usingDefaulta0;
      obj.usingDefaultP0 = index.usingDefaultP0;
      
      % Initialize bounds
      ssLB = fixed.setAllParameters(-Inf);
      ssUB = fixed.setAllParameters(Inf);
      obj.LowerBound = ssLB;
      obj.UpperBound = ssUB;      
      
      % Set variances to be positive
      ssLB.H(1:obj.p+1:end) = eps;
      ssLB.Q(1:obj.m+1:end) = eps;
      if ~obj.usingDefaultP0
        ssLB.P0(1:obj.m+1:end) = eps;
      end
      obj = obj.addRestrictions(ssLB, ssUB);
    end
  end
  
  methods (Static)
    %% Alternate constructors
    function tm = ThetaMapEstimation(ss)
      % Generate a ThetaMap where all parameters values to be estimated are
      % independent elements of theta
      % 
      % Inputs
      %   ss: StateSpace or StateSpaceEstimation
      % Outputs
      %   tm: ThetaMap

      assert(isa(ss, 'AbstractStateSpace'));
      
      % Generate default index and transformation systems
      index = ThetaMap.IndexStateSpace(ss);
      transformationIndex = ThetaMap.TransformationIndexStateSpace(ss);
      
      transformations = {@(x) x};
      derivatives = {@(x) 1};
      inverses = {@(x) x};
      
      if ss.usingDefaulta0 && ~isempty(ss.a0)
        warning('usingDefaulta0 but a0 already set!');
      end
      if ss.usingDefaultP0 && ~isempty(ss.P0)
        warning('usingDefaultP0 but P0 already set!');
      end
      
      % Create object
      tm = ThetaMap(ss, index, transformations, derivatives, ...
        inverses, transformationIndex);
    end
    
    function tm = ThetaMapAll(ss)
      % Generate a ThetaMap where every element of the system parameters is 
      % included in theta
      % 
      % Inputs
      %   ss: StateSpace or StateSpaceEstimation
      % Outputs
      %   tm: ThetaMap
      
      assert(isa(ss, 'AbstractStateSpace'));
      
      % Set all elements to missing, let ThetaMapEstimation do the work
      for iP = 1:length(ss.systemParam)
        ss.(ss.systemParam{iP})(:) = nan;
      end
      
      tm = ThetaMap.ThetaMapEstimation(ss);      
    end
  end
  
  methods
    %% Conversion functions
    function ss = theta2system(obj, theta)
      % Generate a StateSpace from a vector theta
      % 
      % Inputs
      %   theta: Vector of varried parameters
      % Output 
      %   ss:    A StateSpace
      
      % Handle inputs
      assert(all(size(theta) == [obj.nTheta 1]), ...
        'Size of theta does not match ThetaMap.');
      assert(all(isfinite(theta)), 'Theta must be non-nan');
      
      % Dimensions and preallocation
      nParameterMats = length(obj.fixed.systemParam);
      knownParams = cell(nParameterMats, 1);
      
      % Construct the new parameter matricies (including initial values)
      for iP = 1:nParameterMats
        iName = obj.fixed.systemParam{iP};
        
        % Get fixed values
        constructedParamMat = obj.fixed.(iName);
        
        % Fill non-fixed values with transformations of theta
        % We don't need to worry about symmetric matricies here since the index
        % matricies should be symmetric as well at this point.
        freeValues = find(logical(obj.index.(iName)));
        for jF = freeValues'
          jTrans = obj.transformations{obj.transformationIndex.(iName)(jF)};
          jTheta = theta(obj.index.(iName)(jF));
          constructedParamMat(jF) = jTrans(jTheta);
        end
        knownParams{iP} = constructedParamMat;
      end
      
      % Create StateSpace using parameters just constructed
      ss = ThetaMap.cellParams2ss(knownParams, obj); 
    end
    
    function theta = system2theta(obj, ss)
      % Get the theta vector that would determine a system
      % 
      % Inputs
      %   ss:    A StateSpace
      % Output 
      %   theta: Vector of varried parameters
            
      % Handle inputs
      obj.index.checkConformingSystem(ss);
      ssParamValues = ss.vectorizedParameters();
      assert(all(obj.LowerBound.vectorizedParameters < ssParamValues | ...
        isnan(ssParamValues) | ...
        obj.index.vectorizedParameters == 0), ...
        'System violates lower bound of ThetaMap.');
      assert(all(obj.UpperBound.vectorizedParameters > ssParamValues | ...
        isnan(ssParamValues) | ...
        obj.index.vectorizedParameters == 0), ...
        'System violates upper bound of ThetaMap.');
      
      vecIndex = obj.index.vectorizedParameters();
      vecTransIndexes = obj.transformationIndex.vectorizedParameters();
      
      % Loop over theta, identify elements determined by each theta element and
      % compute the inverse of the transformation to get the theta value
      theta = nan(obj.nTheta, 1);
      for iTheta = 1:obj.nTheta
        iIndexes = vecIndex == iTheta;
        nParam = sum(iIndexes);

        iParamValues = ssParamValues(iIndexes);
        iTransformIndexes = vecTransIndexes(iIndexes);
        
        % Get the optimal theta value for each parameter, make sure they match
        iInverse = obj.inverses{iTransformIndexes};
        thetaVals = arrayfun(@(x) iInverse(iParamValues(x)), 1:nParam);        
        assert(all(thetaVals - thetaVals(1) < 1e4 * eps | isnan(thetaVals)), ...
          'Transformation inverses result in differing values of theta.');
        theta(iTheta) = thetaVals(1);
      end
      
    end
    
    %% Gradient functions
    function G = parameterGradients(obj, theta)
      % Create the gradient of the parameter matricies at a given value of theta
      % 
      % Inputs
      %   theta: Vector of varried parameters
      % Outputs
      %   G:     Structure of gradients for each state space parameter. Each 
      %          gradient will be nTheta X (elements in a slice) X tau_x.
      
      % Since every element of the parameter matricies is a function of a 
      % single element of theta and we have derivatives of those functions, 
      % we can simply apply those derivatives to each point to get the full 
      % gradients.
      % Also, we don't need to worry about symmetric v. nonsymmetric
      % matricies since the index matricies are symmetric.

      % Handle input
      assert(isnumeric(theta) || isa(theta, 'StateSpace'), ...
        ['Invalid input: pass either a theta vector of length %d ' ...
        'or a conforming StateSpace object.'], obj.nTheta);
      if isa(theta, 'StateSpace')
        ss = theta;
        theta = obj.system2theta(ss);
      end
      
      % Construct structure of gradients
      nParameterMats = length(obj.fixed.systemParam);
      structConstructTemp = [obj.fixed.systemParam; cell(1, nParameterMats)];
      G = struct(structConstructTemp{:});

      % Construct the new parameter matricies
      for iP = 1:nParameterMats
        iName = obj.fixed.systemParam{iP};
        
        paramGrad = zeros(obj.nTheta, numel(obj.index.(iName)(:,:,1)), ...
          size(obj.index.(iName), 3));
        
        freeValues = reshape(find(logical(obj.index.(iName))), 1, []);
        for jF = freeValues
          freeIndex = obj.index.(iName)(jF);
          jTrans = obj.derivatives{obj.transformationIndex.(iName)(jF)};
          jTheta = theta(freeIndex);
          paramGrad(freeIndex, jF) = jTrans(jTheta);
        end
        
        G.(iName) = paramGrad;
      end  
      
      if obj.usingDefaulta0
        G.a0 = [];
      end
      if obj.usingDefaultP0
        G.P0 = [];
      end
    end
    
    function [Ga0, GP0] = initialValuesGradients(obj, ss, G)
      % Get the gradients of the initial values.
      % 
      % Inputs
      %   ss:  StateSpace where the gradient should be taken
      %   G:   Structure of gradients from ThetaMap.parameterGradients
      % Outputs
      %   Ga0: Gradient of a0
      %   GP0: Gradient of P0
      
      % If we're using the default a0 or P0, calculate those gradients as
      % functions of the parameters. For the diffuse case, the gradients will be
      % zeros. 
      
      % Handle inputs
      assert(isnumeric(ss) || isa(ss, 'StateSpace'), ...
        ['Invalid input: pass either a theta vector of length %d ' ...
        'or a conforming StateSpace object.'], obj.nTheta);
      if isnumeric(ss)
        theta = ss;
        ss = obj.theta2system(theta);
      end
       
      % Determine which case we have
      if obj.usingDefaulta0 || obj.usingDefaultP0
        T1 = ss.T(:,:,1);
        R1 = ss.R(:,:,1);
        Q1 = ss.Q(:,:,1);
        stationaryStateFlag = all(eig(T1) < 1);
      else
        stationaryStateFlag = false;
      end
      
      % Ga0
      if ~obj.usingDefaulta0
        Ga0 = [];
      elseif obj.usingDefaulta0 && ~stationaryStateFlag
        % Diffuse case - a0 is always the zero vector and so will not change
        % with any of the theta parameters
        Ga0 = zeros(obj.nTheta, obj.m);
      else
        % See documentation for calculation of the initial conditions gradients
        IminusTinv = inv(eye(obj.m) - T1);
        IminusTPrimeInv = inv((eye(obj.m) - T1)');
        Ga0 = G.T * kron(IminusTinv, IminusTPrimeInv) * ...
          kron(ss.c(:,:,1), eye(obj.m)) + ...
          G.c / (eye(obj.m) - T1);
      end
      
      % GP0
      if ~obj.usingDefaultP0 
        GP0 = [];
      elseif obj.usingDefaultP0 && ~stationaryStateFlag
        % Diffuse case - P0 is always a large diagonal matrix that doesn't
        % change with parameter values so its gradient is all zeros
        GP0 = zeros(obj.nTheta, obj.m^2);
      else
        % See documentation for calculation of the initial conditions gradients
        Nm = (eye(obj.m^2) + obj.genCommutation(obj.m));
        vec = @(M) reshape(M, [], 1);
        
        rawGTkronT = ThetaMap.GAkronA(T1);
        GTkronT = zeros(obj.nTheta, obj.m^4);
        usedT = logical(obj.index.T);
        GTkronT(obj.index.T(usedT), :) = rawGTkronT(vec(usedT), :);
        
        IminusTkronTInv = inv(eye(obj.m^2) - kron(T1, T1));
        IminusTkronTPrimeInv = inv(eye(obj.m^2) - kron(T1, T1)');
        GP0 = GTkronT * kron(IminusTkronTInv, IminusTkronTPrimeInv) * ...
          kron(vec(R1 * Q1 * R1'), eye(obj.m^2)) + ...
          G.R * (kron(Q1 * R1', eye(obj.m))) * Nm + ...
          G.Q * kron(R1', R1');
      end
    end

    %% Utility functions
    function obj = addRestrictions(obj, ssLB, ssUB)
      % Restrict the possible StateSpaces that can be created by altering the
      % transformations used
      % 
      % Inputs
      %   ssLB: Lower bound StateSpace
      %   ssUB: Upper bound StateSpace
      % Output
      %   obj:  Altered ThetaMap with added lower and upper bounds
      
      % Handle inputs
      if nargin < 3 || isempty(ssUB)
        ssUB = obj.UpperBound;
      end
      if nargin < 2 || isempty(ssLB)
        ssLB = obj.LowerBound;
      end
      
      % Check dimensions
      ssLB.checkConformingSystem(obj);
      ssUB.checkConformingSystem(obj);
      
      for iP = 1:length(obj.LowerBound.systemParam)
        iParam = obj.LowerBound.systemParam{iP};
        % Find the higher lower bound and lower upper bound
        oldLBmat = obj.LowerBound.(iParam);
        passedLBmat = ssLB.(iParam);
        newLBmat = max(oldLBmat, passedLBmat);
        
        oldUBmat = obj.UpperBound.(iParam);
        passedUBmat = ssUB.(iParam);
        newUBmat = min(oldUBmat, passedUBmat);
        
        % Alter any transformations neccessary
        for iElem = 1:numel(newLBmat)
          if newLBmat(iElem) ~= oldLBmat(iElem) || newUBmat(iElem) ~= oldUBmat(iElem);
            [trans, deriv, inver] = ...
              ThetaMap.boundedTransform(newLBmat(iElem), newUBmat(iElem));
            
            % Just add a new transformation for now, we'll delete the old ones
            % in checkThetaMap later
            obj.transformationIndex.(iParam)(iElem) = length(obj.transformations) + 1;
            
            obj.transformations = [obj.transformations {trans}];
            obj.derivatives = [obj.derivatives {deriv}];
            obj.inverses = [obj.inverses {inver}];
          end
        end
        
        % Save new lower and upper bounds
        obj.LowerBound.(iParam) = newLBmat;
        obj.UpperBound.(iParam) = newUBmat;
      end
    end
    
    function obj = checkThetaMap(obj)
      % Verify that the ThetaMap is valid after user modifications. 
      
      % Minimize the size of theta needed after edits have been made to index:
      % If the user changes an element to be a function of a different theta
      % value, remove the old theta value - effectively shift all indexes down 
      % by 1.
      
      % Remove unused transformations (and reset transformationIndexes too)
      
      % Reset nTheta if we've added/removed elements
      
      % Make sure the lower bound is actually below the upper bound
      
      % Other error checking?
      
    end
  end
  
  methods (Static, Hidden)
    %% Alternate constructor helper functions
    function index = IndexStateSpace(ss)
      % Set up index StateSpace for default case where all unknown elements of 
      % the parameters are to be estimated individually
      % 
      % Inputs 
      %   ss:         StateSpaceEstimation with nan values for elements to be 
      %               determined by a ThetaMap
      % Outputs
      %   transIndex: A StateSpace with indexes for each element determined by
      %               theta that indicates the element of thete to be used
      
      paramEstimIndexes = cell(length(ss.systemParam), 1);
      indexCounter = 1;
      
      ssZeros = ss.setAllParameters(0);
      for iP = 1:length(ss.systemParam)
        iParam = ss.(ss.systemParam{iP});
        
        estimInds = ssZeros.(ss.systemParam{iP});
        if ~any(strcmpi(ss.systemParam{iP}, ss.symmetricParams))
          % Unrestricted matricies - Z, d, T, c, R
          % We need an element of theta for every missing element
          nRequiredTheta = sum(sum(sum(isnan(iParam))));
          estimInds(isnan(iParam)) = indexCounter:indexCounter + nRequiredTheta - 1;
        else
          % Symmetric variance matricies - H & Q. 
          % We only need as many elements of theta as there are missing elements
          % in the lower diagonal of these matricies. 
          nRequiredTheta = sum(sum(sum(isnan(tril(iParam)))));
          estimInds(isnan(tril(iParam))) = indexCounter:indexCounter + nRequiredTheta - 1;
          estimInds = estimInds + estimInds' - diag(diag(estimInds));
        end
        
        paramEstimIndexes{iP} = estimInds;        
        indexCounter = indexCounter + nRequiredTheta;
      end
      
      index = ThetaMap.cellParams2ss(paramEstimIndexes, ss);
    end
    
    function transIndex = TransformationIndexStateSpace(ss)
      % Create the default transformationIndex - all parameters values are zeros
      % except where ss is nan, in which case they are ones. 
      % 
      % Inputs 
      %   ss:         StateSpaceEstimation with nan values for elements to be 
      %               determined by a ThetaMap
      % Outputs
      %   transIndex: A StateSpace with indexes for each element determined by
      %               theta that indicates the transformation to be applied
      
      transIndParams = cell(length(ss.systemParam), 1);

      % Create parameter matrix of zeros, put a 1 where ss parameters are 
      % missing since all transformation will start as the unit transformation
      ssZeros = ss.setAllParameters(0);
      for iP = 1:length(ss.systemParam)
        iParam = ss.(ss.systemParam{iP});

        indexes = ssZeros.(ss.systemParam{iP});
        indexes(isnan(iParam)) = 1;
        transIndParams{iP} = indexes;
      end
      
      % Create StateSpace with system parameters
      transIndex = ThetaMap.cellParams2ss(transIndParams, ss);
    end
  end
  
  methods (Static, Hidden)
    %% Helper functions
    function [trans, deriv, inver] = boundedTransform(lowerBound, upperBound)
      % Generate a restriction transformation from a lower and upper bound
      % Also returns the derivative and inverse of the transformation
      % 
      % Inputs
      %   lowerBound: Scalar lower bound
      %   upperBound: Scalar upper bound
      % Outputs
      %   trans: transformation mapping [-Inf, Inf] to the specified interval
      %   deriv: the derivative of trans
      %   inver: the inverse of trans      
      
      if isfinite(lowerBound) && isfinite(upperBound)
        % Logistic function
        trans = @(x) lowerBound + ((upperBound - lowerBound) ./ (1 + exp(-x)));
        deriv = @(x) (exp(x) * (upperBound - lowerBound)) ./ ((exp(x) + 1).^2);
        inver = @(x) -log(((upperBound - lowerBound) ./ (x - lowerBound)) - 1);
      elseif isfinite(lowerBound)
        % Exponential 
        trans = @(x) exp(x) + lowerBound;
        deriv = @(x) exp(x);
        inver = @(x) log(x - lowerBound);
      elseif isfinite(upperBound)
        % Negative exponential 
        trans = @(x) -exp(x) + upperBound;
        deriv = @(x) -exp(x);
        inver = @(x) log(upperBound - x);
      else
        % Unit transformation
        trans = @(x) x;
        deriv = @(x) 1;
        inver = @(x) x;
      end      
    end
    
    function ssNew = cellParams2ss(cellParams, ssOld)
      % Create StateSpace with system parameters passed in a cell array
      % 
      % Inputs
      %   cellParams: Cell array with 9 cells: Z, d, H, T, c, R, Q, a0 & P0
      %   ssOld:      StateSpace or ThetaMap with information on handling of
      %               initial values
      % Outputs
      %   ssNew:      StateSpace constructed with new parameters
      
      ssNew = StateSpace(cellParams{1:7});
      
      % Set initial values with initial values just constructed
      ssNew = ssNew.setInitial(cellParams{8}, cellParams{9});
      ssNew.usingDefaulta0 = ssOld.usingDefaulta0;
      ssNew.usingDefaultP0 = ssOld.usingDefaultP0;
    end
    
    function Gkron = GAkronA(A)
      % Return G_theta(kron(A,A)) for a matrix A with elements determined by a
      % theta vector with an element for each entry in A.
      % 
      % Inputs
      %   A:     A square matrix
      % Outputs
      %   Gkron: The gradient of kron(A, A) ordered column-wise
      
      % David Kelley & Bill Kluender, 2016
      % The gradient can be broken up into two main components: 
      %   - A set of progressing diagonals 
      %   - A set of rows that repeat on the primary block diagonal
       
      assert(size(A, 1) == size(A, 2), 'Input must be square.');
      m = size(A, 1);
      
      % Diagonals
      submatDiags = arrayfun(@(x) kron(eye(m), kron(A(:, x)', eye(m))), 1:m, ...
        'Uniform', false);      
      diagonals = horzcat(submatDiags{:});
      
      % Rows
      submatRows = arrayfun(@(x) kron(eye(m), A(:, x)'), 1:m, 'Uniform', false);
      rows = kron(eye(m), horzcat(submatRows{:}));

      Gkron = diagonals + rows;
    end
  end
end
