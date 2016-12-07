classdef ThetaMap
  % ThetaMap - Mapping from a vector of parameters to a StateSpace 
  %
  % A vector theta will be used to construct a StateSpace. Each non-fixed scalar
  % parameter of the StateSpace is allowed to be a function of one element of
  % the theta vector. Multiple parameter values may be based off of the same
  % element of theta.
  %
  % When estimating unknown values in a StateSpace, a ThetaMap can be generated
  % where all free parameters will be determined by an independent element of
  % theta (except in variance matricies that must be symmetric) - see the
  % IndependentEstimation function.
  %
  % Once
  %
  % There are five primary components of a ThetaMap:
  %   - fixed: A StateSpace where each of the fixed values is entered into the
  %   corresponding parameter.
  %   - index: A StateSpace object of integer indexes. Fixed elements must be
  %   denoted with a zero. Elements that depend on theta can have an integer
  %   value between 1 and the length of theta, indicating the element of theta
  %   that will be used to construct that element.
  %   - transformations: A cell array of function handles. Each function should
  %   take a scalar input and return a scalar output.
  %   - derivatives: A cell array of derivatives for each transformation.
  %   - transformationIndex: A StateSpace object of integer indexes. Fixed
  %   elements must be denoted with a zero. Elements that depend on theta can
  %   have an integer value between 1 and the total number of transformations
  %   provided, indicating which function will be used to transform the value of
  %   theta(i) before placing it in the appropriate parameter matrix.
  
  % David Kelley, 2016
  %
  % TODO (12/2/2016)
  % ---------------
  %   - Allow user-defined transformations so a parameter value can 
  %   effectively be a function of another parameter value.
  %   - Check that system2theta satisfies bound conditions
  %   - Write checkThetaMap 
  %   - Determine transformations for accumulators, decide where to put them.
  %   - Which tau should be used for a0 and P0?
  %   - Do I need to be able to remove a parameter restriction ever? 
  
  properties(SetAccess = protected)
    nTheta            % Number of elements in theta vector

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
  
  properties(SetAccess = protected, Hidden)
    % Dimensions
    p                 % Number of observed series
    m                 % Number of states
    g                 % Number of shocks
    n                 % Time periods

    timeInvariant     % Indicator for TVP models
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
  
  methods(Static)
    %% Alternate constructors
    function tm = ThetaMapEstimation(ss)
      % Generate a ThetaMap where each parameter value to be estimated is an
      % independent element of theta (still restricts variance matricies to be
      % symmetric).
      assert(isa(ss, 'AbstractStateSpace'));
      
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
      
      tm = ThetaMap(ss, index, transformations, derivatives, inverses, transformationIndex);
    end
    
    function tm = ThetaMapAll(ss)
      % Generate a ThetaMap where every element of every parameter is included
      % in theta (still with symmetric restriction, used primarily for 
      % the gradient function)
      assert(isa(ss, 'AbstractStateSpace'));
      
      for iP = 1:length(ss.systemParam)
        ss.(ss.systemParam{iP})(:) = nan;
      end
      
      tm = ThetaMap.ThetaMapEstimation(ss);      
    end
  end
  
  methods(Static, Hidden)
    %% Alternate constructor helper functions
    function index = IndexStateSpace(ss)
      % Set up index StateSpace for default case where all unknown elements of 
      % the parameters are to be estimated individually
      
      paramEstimIndexes = cell(length(ss.systemParam), 1);
      indexCounter = 1;
      
      ssZeros = ss.setAllParameters(0);
      for iP = 1:length(ss.systemParam)
        iParam = ss.(ss.systemParam{iP});
        
        estimatedIndexes = ssZeros.(ss.systemParam{iP});
        if ~any(strcmpi(ss.systemParam{iP}, ss.symmetricParams))
          % Unrestricted matricies
          nRequiredTheta = sum(sum(sum(isnan(iParam))));
          estimatedIndexes(isnan(iParam)) = indexCounter:indexCounter + nRequiredTheta - 1;
        else
          % Symmetric matricies
          nRequiredTheta = sum(sum(sum(isnan(tril(iParam)))));
          estimatedIndexes(isnan(tril(iParam))) = indexCounter:indexCounter + nRequiredTheta - 1;
          
          estimatedIndexes = estimatedIndexes + estimatedIndexes' - diag(diag(estimatedIndexes));
        end
        paramEstimIndexes{iP} = estimatedIndexes;
        
        indexCounter = indexCounter + nRequiredTheta;
      end
      
      index = StateSpace(paramEstimIndexes{1:7}, []);
      index = index.setInitial(paramEstimIndexes{8}, paramEstimIndexes{9});
      index.usingDefaulta0 = ss.usingDefaulta0;
      index.usingDefaultP0 = ss.usingDefaultP0;
    end
    
    function transformationIndex = TransformationIndexStateSpace(ss)
      % Create the default transformationIndex - all parameters values are zeros
      % except where ss is nan, in which case they are ones. 
      paramTransIndexes = cell(length(ss.systemParam), 1);

      ssZeros = ss.setAllParameters(0);
      for iP = 1:length(ss.systemParam)
        iParam = ss.(ss.systemParam{iP});

        transformationIndexes = ssZeros.(ss.systemParam{iP});
        transformationIndexes(isnan(iParam)) = 1;
        paramTransIndexes{iP} = transformationIndexes;
      end
      
      transformationIndex = StateSpace(paramTransIndexes{1:7}, []);
      transformationIndex = transformationIndex.setInitial(paramTransIndexes{8}, paramTransIndexes{9});
      transformationIndex.usingDefaulta0 = ss.usingDefaulta0;
      transformationIndex.usingDefaultP0 = ss.usingDefaultP0;
    end
  end
  
  methods
    %% Conversion functions
    function ss = theta2system(obj, theta)
      % Generate a StateSpace from a vector theta
      assert(all(size(theta) == [obj.nTheta 1]), ...
        'Size of theta does not match ThetaMap.');
      assert(all(isfinite(theta)), 'Theta must be non-nan');
      
      nParameterMats = length(obj.fixed.systemParam);
      knownParams = cell(nParameterMats, 1);
      
      % Construct the new parameter matricies
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
      
      % Create StateSpace
      ss = StateSpace(knownParams{1:7}, []);
      ss = ss.setInitial(knownParams{8}, knownParams{9});
    end
    
    function theta = system2theta(obj, ss)
      % Get the theta vector that would determine a system
      
      % Since we know that each elemenet of the parameters must be determined
      % from an individual element of theta we can do nTheta univariate
      % optimization problems to find the theta that would produce ss.
      % To make sure the StateSpace is feasible given the ThetaMap, we find the
      % value of theta that would produce each element then check to make sure
      % that they're the same.
      
      % TODO: Check that the resulting StateSpace satisfies bounds conditions
      obj.index.checkConformingSystem(ss);
      
      vecIndex = obj.index.vectorizedParameters();
      ssParamValues = ss.vectorizedParameters();
      vecTransIndexes = obj.transformationIndex.vectorizedParameters();
      
      theta = nan(obj.nTheta, 1);
      for iTheta = 1:obj.nTheta
        iIndexes = vecIndex == iTheta;
        nParam = sum(iIndexes);

        iParamValues = ssParamValues(iIndexes);
        iTransformIndexes = vecTransIndexes(iIndexes);
        
        % Get the optimal theta value for each parameter, make sure they match
        thetaVals = arrayfun(@(x) obj.inverses{iTransformIndexes(x)}(iParamValues(x)), 1:nParam);        
        assert(all(thetaVals - thetaVals(1) < 1e4 * eps), ...
          'Transformation inverses result in differing values of theta.');
        
        theta(iTheta) = thetaVals(1);
      end
      
    end
    
    %% Gradient functions
    function G = parameterGradients(obj, theta)
      % Create the gradient of the parameter matricies at a given value of theta
      %
      % Each gradient will be nTheta X (elements in a slice) X tau_x.
      
      ss = obj.theta2system(theta);
      
      nParameterMats = length(obj.fixed.systemParam);
      structConstructTemp = [obj.fixed.systemParam; cell(1, nParameterMats)];
      G = struct(structConstructTemp{:});
      
      % If we're using the default a0 or P0, calculate those gradients as
      % functions of T, c and RQR'.
      if obj.fixed.usingDefaulta0 || obj.fixed.usingDefaultP0
        stationaryState = all(eig(ss.T(:,:,1)) < 1);
      end
      
      % Construct the new parameter matricies
      for iP = 1:nParameterMats
        iName = obj.fixed.systemParam{iP};
        if strcmpi(iName, 'a0') && obj.usingDefaulta0 && ~stationaryState
          % Diffuse case
          paramGrad = zeros(obj.nTheta, obj.m);
        elseif strcmpi(iName, 'a0') && obj.usingDefaulta0 && stationaryState
          % See note in documentation on initial conditions gradients
          IminusTinv = inv(eye(obj.m) - ss.T(:,:,1));
          IminusTPrimeInv = inv((eye(obj.m) - ss.T(:,:,1))');
          paramGrad = G.T * kron(IminusTinv, IminusTPrimeInv) * ...
            kron(ss.c(:,:,1), eye(obj.m)) + ...
            G.c * IminusTinv;
        
        elseif strcmpi(iName, 'P0') && obj.usingDefaultP0 && ~stationaryState
          % Diffuse case 
          paramGrad = zeros(obj.nTheta, obj.m^2);
        elseif strcmpi(iName, 'P0') && obj.usingDefaultP0 && stationaryState
          % See note in documentation on initial conditions gradients
          Nm = (eye(obj.m^2) + AbstractStateSpace.genCommutation(obj.m));
          vec = @(M) reshape(M, [], 1);

          rawGTkronT = ThetaMap.GAkronA(ss.T(:,:,1)); 
          GTkronT = zeros(obj.nTheta, obj.m^4);
          
          usedT = logical(obj.index.T);
          GTkronT(obj.index.T(usedT), :) = rawGTkronT(vec(usedT), :);
          
          Im2minusTkronInv = inv(eye(obj.m^2) - kron(ss.T(:,:,1), ss.T(:,:,1)));
          Im2minusTkronPrimeInv = inv(eye(obj.m^2) - kron(ss.T(:,:,1), ss.T(:,:,1))');
          paramGrad = GTkronT * kron(Im2minusTkronInv, Im2minusTkronPrimeInv) * ...
              kron(vec(ss.R(:,:,1) * ss.Q(:,:,1) * ss.R(:,:,1)'), eye(obj.m^2)) + ... 
            G.R * (kron(ss.Q(:,:,1) * ss.R(:,:,1)', eye(obj.m))) * Nm + ...
            G.Q * kron(ss.R(:,:,1)', ss.R(:,:,1)');

        else
          % Normal parameter matricies: find elements determined by theta and
          % use the provided derivatives.
          paramGrad = zeros(obj.nTheta, ...
            numel(obj.index.(iName)(:,:,1)), size(obj.index.(iName), 3));
          
          % We don't need to worry about symmetric matricies since the index
          % matricies should be symmetric as well at this point.
          freeValues = find(logical(obj.index.(iName)));
          
          if size(freeValues, 1) ~= 1
            freeValues = freeValues';
          end
          
          for jF = freeValues
            freeIndex = obj.index.(iName)(jF);
            jTrans = obj.derivatives{obj.transformationIndex.(iName)(jF)};
            jTheta = theta(freeIndex);
            paramGrad(freeIndex, jF) = jTrans(jTheta);
          end
        end
        
        G.(iName) = paramGrad;
      end
    end

    %% Utility functions
    function obj = addRestrictions(obj, ssLB, ssUB)
      % Restrict the possible StateSpaces that can be created by altering the
      % transformations used
      
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
            [trans, deriv, inver] = ThetaMap.boudnedTransform(newLBmat(iElem), newUBmat(iElem));
            
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
      % Minimize the size of theta needed after edits have been made to index,
      % reset nTheta if we've added elements, remove unused transformations, 
      % general error checking
      
    end
  end
  
  methods(Static, Hidden)
    % Helper functions
    function [trans, deriv, inverse] = boudnedTransform(lowerBound, upperBound)
      % Generate a restriction transformation from a lower and upper bound
      % Also returns the derivative and inverse of the transformation
      
      if isfinite(lowerBound) && isfinite(upperBound)
        trans = @(x) lowerBound + ((upperBound - lowerBound) ./ (1 + exp(-x)));
        deriv = @(x) (exp(x) * (upperBound - lowerBound)) ./ ((exp(x) + 1).^2);
        inverse = @(x) -log(((upperBound - lowerBound) ./ (x - lowerBound)) - 1);
      elseif isfinite(lowerBound)
        trans = @(x) exp(x) + lowerBound;
        deriv = @(x) exp(x);
        inverse = @(x) log(x - lowerBound);
      elseif isfinite(upperBound)
        trans = @(x) -exp(x) + upperBound;
        deriv = @(x) -exp(x);
        inverse = @(x) log(upperBound - x);
      else
        trans = @(x) x;
        deriv = @(x) 1;
        inverse = @(x) x;
      end      
    end
    
    function Gkron = GAkronA(A)
      % Return G_theta(kron(A,A))
      % David Kelley & Bill Klunder, 2016
      % The gradient can be broken up into two main components: 
      %   - A set of progressing diagonals 
      %   - A set of rows that repeat on the primary block diagonal
      assert(size(A, 1) == size(A, 2), 'Input must be square.');
      m = size(A, 1);
      
      % Diagonals
      submatDiags = arrayfun(@(x) kron(eye(m), kron(A(:, x)', eye(m))), 1:m, 'Uniform', false);      
      diagonals = horzcat(submatDiags{:});
      
      % Rows
      submatRows = arrayfun(@(x) kron(eye(m), A(:, x)'), 1:m, 'Uniform', false);
      rows = kron(eye(m), horzcat(submatRows{:}));

      Gkron = diagonals + rows;
    end
  end
end
