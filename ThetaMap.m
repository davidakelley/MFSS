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
  %
  % David Kelley, 2016-2017
  %
  % TODO (1/10/17)
  % ---------------
  %   - Write tests for ThetaMap. They should verify: 
  %     - Check a generated system obeys bounds. 
  %     - Creating a system with accumulators is the same as generating a system
  %       without accumulators and adding the accumulators in afterward. 
  %   - Function comparison of transformations: if a string match of the
  %     functions() function fields match and the workspaces match, the 
  %     transformations are the same and can be combined. 
  %   - Optional: Allow inverses to be passed empty and do the line-search 
  %     to find the numeric inverse. 
  %   - Write documentation on initial values derivation
  
  properties 
    % StateSpace containing all fixed elements. 
    % Elements of the parameters that will be determined by theta must be set to
    % zero.
    fixed
    
    % StateSpace of indexes of theta that determines parameter values
    index
    
    % StateSpace of indexes of transformations applied for parameter values
    transformationIndex
    % Cell array of transformations of theta to get parameter values
    transformations
    % Derivatives of transformations
    derivatives
    % Inverses of transformations
    inverses
  end
  
  properties (SetAccess = protected)
    nTheta                          % Number of elements in theta vector
  end
  
  properties (SetAccess = protected, Hidden)
    % Parameter bounds - set in addRestrictions
    LowerBound
    UpperBound
    
    % Initial state conditions
    usingDefaulta0
    usingDefaultP0
  end
  
  methods
    %% Constructor
    function obj = ThetaMap(fixed, index, transformationIndex, ...
        transformations, derivatives, inverses)
      % Generate map from elements of theta to StateSpace parameters
      
      ThetaMap.validateInputs(fixed, index, transformationIndex, ...
        transformations, derivatives, inverses);
      
      % Set properties
      obj.fixed = fixed;
      obj.index = index;
      obj.transformationIndex = transformationIndex;
      
      obj.transformations = transformations;
      obj.derivatives = derivatives;
      obj.inverses = inverses;
      
      obj.usingDefaulta0 = fixed.usingDefaulta0;
      obj.usingDefaultP0 = fixed.usingDefaultP0;
      
      % Set dimensions
      obj.nTheta = max(index.vectorizedParameters);
      
      obj.p = fixed.p;
      obj.m = fixed.m;
      obj.g = fixed.g;
      obj.n = fixed.n;
      obj.timeInvariant = fixed.timeInvariant;
      
      % Initialize bounds
      ssLB = fixed.setAllParameters(-Inf);
      ssUB = fixed.setAllParameters(Inf);
      obj.LowerBound = ssLB;
      obj.UpperBound = ssUB;      
      
      % Set diagonals of variance matricies to be positive
      ssLB.H(1:obj.p+1:end) = eps;
      ssLB.Q(1:obj.m+1:end) = eps;
      if ~obj.usingDefaultP0
        ssLB.P0(1:obj.m+1:end) = eps;
      end
      obj = obj.addRestrictions(ssLB, ssUB);
      
      % Validate & remove duplicate transformations
      obj = obj.validateThetaMap();
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
      
      % Get fixed elements as a StateSpace
      % Create a StateSpace with zeros (value doesn't matter) replacing nans.
      for iP = 1:length(ss.systemParam)
        ss.(ss.systemParam{iP})(isnan(ss.(ss.systemParam{iP}))) = 0;
      end
      
      fixed = StateSpace(ss.Z, ss.d, ss.H, ss.T, ss.c, ss.R, ss.Q);
      fixed.tau = ss.tau;
      
      fixed.a0 = ss.a0;
      fixed.usingDefaulta0 = ss.usingDefaulta0;
      fixed.P0 = ss.P0;
      fixed.usingDefaultP0 = ss.usingDefaultP0;
      fixed.kappa = ss.kappa;
      
      % Create object
      tm = ThetaMap(fixed, index, transformationIndex, ...
        transformations, derivatives, inverses);
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
      
      % FIXME: Account for tau
      
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
        if obj.fixed.timeInvariant || strcmpi(iName, 'a0') || strcmpi(iName, 'P0')
          knownParams{iP} = constructedParamMat;
        else
          paramStruct = struct;
          paramStruct.([iName 't']) = constructedParamMat;
          paramStruct.(['tau' iName]) = obj.fixed.tau.(iName);
          knownParams{iP} = paramStruct;
        end        
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
        iInverses = obj.inverses(iTransformIndexes);
        thetaVals = arrayfun(@(x) iInverses{x}(iParamValues(x)), 1:nParam);        
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
      
      % TODO: Does this function handle TVP correctly?
      
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
    
    function obj = validateThetaMap(obj)
      % Verify that the ThetaMap is valid after user modifications. 
      
      % Minimize the size of theta needed after edits have been made to index:
      % If the user changes an element to be a function of a different theta
      % value, remove the old theta value - effectively shift all indexes down 
      % by 1.
      obj.index = ThetaMap.eliminateUnusedIndexes(obj.index);

      % Reset nTheta if we've added/removed elements
      obj.nTheta = max(obj.index.vectorizedParameters());
            
      % Remove duplicate and unused transformations
      obj = obj.compressTransformations();      

      % Make sure the lower bound is actually below the upper bound and
      % other error checking?
      assert(all(obj.LowerBound.vectorizedParameters() <= ...
        obj.UpperBound.vectorizedParameters()), ...
        'Elements of LowerBound are greater than UpperBound.');
    end
    
    function obj = compressTransformations(obj)
      % Removes unused or duplicate transformations.
      % For duplicates, set their indexes to the lower-indexed version and 
      % delete the higher-indexed version.
      
      % Remove unused transformations:
      % this should be very similar to removing missing index elements but
      % we also need to delete the transformations
      [obj.transformationIndex, unusedTransforms] = ...
        ThetaMap.eliminateUnusedIndexes(obj.transformationIndex);
      obj.transformations(unusedTransforms) = [];
      obj.derivatives(unusedTransforms) = [];
      obj.inverses(unusedTransforms) = [];
      
      % Remove duplicate transformations: 
      % Progress through the list searching for other transformations that match
      % the current transformation. When one is found, fix all indexes that
      % match that transformation, then delete it.
      iTrans = 0;
      while iTrans < length(obj.transformations)-1
        iTrans = iTrans + 1;
        
        % Check all transformations after the current entry
        duplicateTrans = ThetaMap.isequalTransform(...
          obj.transformations{iTrans}, obj.transformations(iTrans+1:end));
        duplicatesForRemoval = find([zeros(1, iTrans), duplicateTrans]);
        if isempty(duplicatesForRemoval)
          continue
        end
        
        % Reset the indexes of any transformations found to be duplicate
        paramNames = obj.transformationIndex.systemParam;
        for iP = 1:length(paramNames)
          dupInds = arrayfun(@(x) any(x == duplicatesForRemoval), ...
            obj.transformationIndex.(paramNames{iP}));
          obj.transformationIndex.(paramNames{iP})(dupInds) = iTrans;
        end
        
        % Compress indexes
        [obj.transformationIndex, unusedTransforms] = ...
          ThetaMap.eliminateUnusedIndexes(obj.transformationIndex);
        assert(isempty(unusedTransforms) || ...
          all(unusedTransforms == duplicatesForRemoval));
        
        % Remove duplicates
        obj.transformations(duplicatesForRemoval) = [];
        obj.derivatives(duplicatesForRemoval) = [];
        obj.inverses(duplicatesForRemoval) = [];
      end
    end
  end
  
  methods (Static, Hidden)
    %% Constructor helper functions
    function validateInputs(fixed, index, transformationIndex, ...
        transformations, derivatives, inverses)
      % Validate inputs
      assert(isa(fixed, 'StateSpace'));
      assert(isa(index, 'StateSpace'));
      assert(isa(transformationIndex, 'StateSpace'));
      
      % Dimensions
      index.checkConformingSystem(transformationIndex);
      nTransform = max(transformationIndex.vectorizedParameters());
      
      assert(length(transformations) >= nTransform, ...
        'Insufficient transformations provided for indexes given.');
      assert(length(transformations) == length(derivatives), ...
        'Derivatives must be provided for every transformation.');
      assert(length(inverses) == length(derivatives), ...
        'Inverses must be provided for every transformation.');

      vecFixed = fixed.vectorizedParameters();
      assert(~any(isnan(vecFixed)), 'Nan not allowed in fixed.');
      vecIndex = index.vectorizedParameters();
      assert(~any(isnan(vecIndex)), 'Nan not allowed in index.'); 
      
      % Non-zero elements of fixed are zero in index and vice-versa.
      assert(all(~(vecFixed ~= 0 & vecIndex ~= 0)), ...
        'Parameters determined by theta must be set to 0 in fixed.');
      
    end
    
    function [indexSS, missingInx] = eliminateUnusedIndexes(indexSS)
      % Decrement index values of a StateSpace so that therer are no unused
      % integer values from 1 to the maximum value in the system.
      
      assert(isa(indexSS, 'StateSpace'));
      vecIndex = indexSS.vectorizedParameters();
      maxVal = max(vecIndex);
      missingInx = setdiff(1:maxVal, vecIndex);
      if isempty(missingInx)
        return
      end
      
      paramNames = indexSS.systemParam;

      if ~isempty(missingInx)
        for iP = 1:length(paramNames)
          % We need to collapse down through every element that's missing. To do
          % this, count how many "missing" indexes each index is greater than
          % and subtract that number from the existing indexes.
          indexSubtract = arrayfun(@(x) sum(x > missingInx), indexSS.(paramNames{iP}));
          indexSS.(paramNames{iP}) = indexSS.(paramNames{iP}) - indexSubtract;
        end
      end
      
      % The new maximum value will be the current maximum value minus the number
      % of missing integers from 1:maxInx;
      newMax = maxVal - length(missingInx);
      newVecIndex = indexSS.vectorizedParameters();
      
      assert(newMax == max(newVecIndex));
      assert(isempty(setdiff(1:newMax, newVecIndex)), ...
        'Development error. Index cannot skip elements of theta.');
    end
    
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
    
    function result = isequalTransform(fn1, fn2)
      % Determines if two function handles represent the same function
      % 
      % Also accepts cell arrays of function handles. If only one element is a
      % cell array, each is checked to see if they are equal to the non-cell
      % array input. If both elements are cell arrays, they must be the same
      % size and will be checked element-wise.
      
      % David Kelley, 2017

      nComp = max(length(fn1), length(fn2));
      if iscell(fn1) && iscell(fn2)
        assert(size(fn1) == size(fn2), 'Cell array inputs must be the same size.');
      end
      
      if iscell(fn1)
        fnInfo1 = cellfun(@functions, fn1);
        fn1Strs = cellfun(@(x) x.function, fnInfo1, 'Uniform', false);
        fn1Workspace = cellfun(@(x) x.workspace{1}, fnInfo1);
      else
        fnInfo1 = functions(fn1);
        fn1Strs = repmat({fnInfo1.function}, [1 nComp]);
        fn1Workspace = repmat({fnInfo1.workspace{1}}, [1 nComp]);
      end
      if iscell(fn2)
        fnInfo2 = cellfun(@functions, fn2, 'Uniform', false);
        fn2Strs = cellfun(@(x) x.function, fnInfo2, 'Uniform', false);
        fn2Workspace = cellfun(@(x) x.workspace{1}, fnInfo2, 'Uniform', false);
      else
        fnInfo2 = functions(fn2);
        fn2Strs = repmat({fnInfo2.function}, [1 nComp]);
        fn2Workspace = repmat({fnInfo2.workspace{1}}, [1 nComp]);
      end
      
      result = strcmp(fn1Strs, fn2Strs) & cellfun(@isequal, fn1Workspace, fn2Workspace);      
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
      % The only way we were able to figure this out was to write out examples
      % by hand and then run some numerical tests. 
       
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
