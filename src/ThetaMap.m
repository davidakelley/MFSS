classdef ThetaMap < AbstractSystem
  % ThetaMap - Mapping from a vector of parameters to a StateSpace 
  %
  % A vector theta will be used to construct a StateSpace. A vector psi will be 
  % defined element-wise as a function of thetta. Each non-fixed scalar
  % parameter of the StateSpace is allowed to be a function of one element of
  % the psi vector. Multiple parameter values may be based off of the same 
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
  %   take a scalar input and return a scalar output. While not technically
  %   disallowed, all transformations should be monotonic. 
  %   - inverses: A cell array of inverses for each transformation.
  %   - transformationIndex: A StateSpace object of integer indexes. Fixed
  %   elements must be denoted with a zero. Elements that depend on theta can
  %   have an integer value between 1 and the total number of transformations
  %   provided, indicating which function will be used to transform the value of
  %   theta(i) before placing it in the appropriate parameter matrix.
  %
  % David Kelley, 2016-2018
  
  properties 
    % StateSpace containing all fixed elements. 
    % Elements of the parameters that will be determined by theta must be set to zero.
    fixed
    
    % StateSpace of indexes of psi vector that determines staste space parameter values
    index
    
    % StateSpace of indexes of transformations to psi to get state space parameters
    transformationIndex
    % Cell array of transformations of psi to get parameter values
    transformations
    % Inverses of transformations
    inverses
    
    % Psi functions
    PsiTransformation
    % Inverse of Psi
    PsiInverse
    % Indexes of theta that determine each element of Psi
    PsiIndexes
    
    thetaLowerBound
    thetaUpperBound
    
    % Cell array of names of parameters to be estimated
    thetaNames
  end
  
  properties (SetAccess = protected)
    % Parameter bounds - set in addRestrictions
    LowerBound
    UpperBound
    
    % Number of elements in theta vector
    nTheta
    
    % Number of elements in psi vector
    nPsi
  end
  
  properties (SetAccess = protected)
    % Initial state conditions
    usingDefaulta0 = true;
    usingDefaultP0 = true;
  end
  
  methods
    %% Constructor
    function obj = ThetaMap(fixed, index, transformationIndex, ...
        transformations, inverses, varargin)
      % Generate map from elements of theta to StateSpace parameters
      
      inP = inputParser();
      inP.addParameter('explicita0', false);
      inP.addParameter('explicitP0', false);
      inP.addParameter('PsiTransformation', []);
      inP.addParameter('PsiInverse', []);
      inP.addParameter('PsiIndexes', []);
      inP.addParameter('Names', {});
    
      inP.parse(varargin{:});
      opts = inP.Results;
      
      ThetaMap.validateInputs(fixed, index, transformationIndex, ...
        transformations, inverses, opts);
      
      % Set dimensions
      if ~isempty(opts.PsiIndexes)
        obj.nTheta = max(cellfun(@max, opts.PsiIndexes));
      else
        obj.nTheta = max(ThetaMap.vectorizeStateSpace(index, opts.explicita0, opts.explicitP0));
      end
      if ~isempty(opts.PsiTransformation)
        obj.nPsi = length(opts.PsiTransformation);
      else
        obj.nPsi = obj.nTheta;
      end
      
      if ~isempty(opts.Names)
        assert(length(opts.Names) == obj.nTheta);
        obj.thetaNames = opts.Names;
      else
        obj.thetaNames = arrayfun(@(iT) sprintf('theta_%d', iT), 1:obj.nTheta, 'Uniform', false)';
      end
      
      % Set properties
      obj.fixed = fixed;
      obj.index = index;
      obj.transformationIndex = transformationIndex;
      
      obj.transformations = transformations;
      obj.inverses = inverses;
      
      if isempty(opts.PsiTransformation)
        obj.PsiTransformation = repmat({@(theta) theta}, [obj.nTheta 1]);
      else
        obj.PsiTransformation = opts.PsiTransformation;
      end
      if isempty(opts.PsiInverse)
        obj.PsiInverse = repmat({@(psi, inx) psi(inx(1))}, [obj.nTheta 1]);
      else
        obj.PsiInverse = opts.PsiInverse;
      end
      if isempty(opts.PsiIndexes)
        obj.PsiIndexes = num2cell(1:obj.nTheta);
      else
        obj.PsiIndexes = opts.PsiIndexes;
      end

      obj.usingDefaulta0 = ~opts.explicita0;
      obj.usingDefaultP0 = ~opts.explicitP0;

      obj.p = fixed.p;
      obj.m = fixed.m;
      obj.g = fixed.g;
      obj.n = fixed.n;
      obj.timeInvariant = fixed.timeInvariant;
      
      % Initialize bounds
      obj.thetaLowerBound = -Inf(obj.nTheta, 1);
      obj.thetaUpperBound = Inf(obj.nTheta, 1);
      
      ssLB = StateSpace.setAllParameters(fixed, -Inf);
      ssUB = StateSpace.setAllParameters(fixed, Inf);
      obj.LowerBound = ssLB;
      obj.UpperBound = ssUB;      
      
      % Set diagonals of variance matricies to be positive
      ssLB.H(1:obj.p+1:end) = eps * 10;
      ssLB.Q(1:obj.g+1:end) = eps * 10;
      if ~obj.usingDefaultP0
        P0LB = -Inf(obj.m);
        P0LB(1:obj.m+1:end) = eps * 10;
        ssLB.P0 = P0LB;
        ssLB.P0(1:obj.m+1:end) = eps * 10;
      end
      obj = obj.addRestrictions(ssLB, ssUB);
      
      % Validate & remove duplicate transformations
      obj = obj.validateThetaMap();
    end
  end
  
  methods (Static)
    %% Alternate constructors
    function tm = ThetaMapEstimation(ssE)
      % Generate a ThetaMap where all parameters values to be estimated are
      % independent elements of theta
      % 
      % Inputs
      %   ss: StateSpace or StateSpaceEstimation
      % Outputs
      %   tm: ThetaMap

      assert(isa(ssE, 'AbstractStateSpace'));
      
      % Generate default index and transformation systems
      index = ThetaMap.IndexStateSpace(ssE);
      transformationIndex = ThetaMap.TransformationIndexStateSpace(ssE);
      
      paramVec = [ssE.Z(:); ssE.d(:); ssE.beta(:); ssE.H(:); ...
        ssE.T(:); ssE.c(:); ssE.gamma(:); ssE.R(:); ssE.Q(:)];
      
      if isa(paramVec, 'sym')
        % theta is ordered by symbolic variables, then nan variables
        symTheta = symvar(paramVec);
        % psi will be ordered by symbolic variables, then nan variables
        symPsi = unique(paramVec(has(paramVec, symTheta) & ~isnan(paramVec)));
      else
        symTheta = [];
        symPsi = [];
      end
      
      explicita0 = ~isempty(ssE.a0);
      explicitP0 = ~isempty(ssE.P0);
      
      nTheta = length(symTheta) + ...
        sum(cellfun(@(paramName) sum(isnan(ssE.(paramName)(:))), ...
        {'Z', 'd', 'beta', 'T', 'c', 'gamma', 'R'})) + ...
        sum(cellfun(@(paramName) sum(isnan(reshape(tril(ssE.(paramName)), [], 1))), ...
        {'H', 'Q'}));
      if explicita0
        nTheta = nTheta + sum(isnan(ssE.a0));
      end
      if explicitP0
        nTheta = nTheta + sum(isnan(ssE.P0));
      end
      
      if isa(paramVec, 'sym')
        symNames = cellfun(@char, sym2cell(symTheta), 'Uniform', false)';
      else
        symNames = {};
      end
      names = [symNames; ...
        arrayfun(@(iT) sprintf('theta_%d', iT), length(symTheta)+1:nTheta, 'Uniform', false)'];
      
      % Theta to Psi transformations
      % Cell of length nPsi of which theta elements determine element of Psi
      symPsiInx = cell(length(symPsi),1);
      for iPsi = 1:length(symPsi)
        symPsiInx{iPsi} = find(arrayfun(@(iTheta) has(symPsi(iPsi), iTheta), symTheta));
      end      
      PsiIndexes = [symPsiInx; num2cell(length(symTheta)+1:nTheta)'];
      % Going from theta to psi
      symPsiTrans = cell(length(symPsi),1);
      for iPsi = 1:length(symPsi)
        symPsiTrans{iPsi} = matlabFunction(symPsi(iPsi), ...
          'Vars', {symTheta(PsiIndexes{iPsi})});
      end
      PsiTransformations = [symPsiTrans; ...
        repmat({@(theta) theta}, [length(PsiIndexes)-length(symPsi) 1])]; 
      % Find theta given psi. Empty inverses will be done numerically.
      PsiInverses = [repmat({[]}, [length(symTheta) 1]); ...
        repmat({@(psi, inx) psi(inx(1))}, [nTheta-length(symTheta) 1]);];
      
      % Psi to state space parameter transformations
      transformations = {@(x) x};
      inverses = {@(x) x};
      
      % Get fixed elements as a StateSpace
      % Create a StateSpace with zeros (value doesn't matter) replacing nans.
      
      fixed = StateSpace(zeros(size(ssE.Z)), zeros(size(ssE.H)), ...
        zeros(size(ssE.T)), zeros(size(ssE.Q)), ...
        'd', zeros(size(ssE.d)), 'beta', zeros(size(ssE.beta)), ...
        'c', zeros(size(ssE.c)), 'gamma', zeros(size(ssE.gamma)), 'R', zeros(size(ssE.R)));
      
      for iP = 1:length(fixed.systemParam)
        if isa(ssE.(ssE.systemParam{iP}), 'sym')
          fixedElems = isfinite(ssE.(ssE.systemParam{iP})) & ...
            ~has(ssE.(ssE.systemParam{iP}), symTheta);
        else
          fixedElems = isfinite(ssE.(ssE.systemParam{iP}));
        end        
        fixed.(fixed.systemParam{iP})(fixedElems) = ...
          double(ssE.(ssE.systemParam{iP})(fixedElems));
      end
      
      fixed.tau = ssE.tau;
      a0Fixed = ssE.a0;
      a0Fixed(isnan(a0Fixed)) = 0;
      fixed.a0 = a0Fixed;
      P0Fixed = ssE.P0;
      P0Fixed(isnan(P0Fixed)) = 0;
      fixed.P0 = P0Fixed;
      
      % Create object
      tm = ThetaMap(fixed, index, transformationIndex, ...
        transformations, inverses, ...
        'explicita0', explicita0, 'explicitP0', explicitP0, ...
        'PsiIndexes', PsiIndexes, 'PsiTransformation', PsiTransformations, ...
        'PsiInverse', PsiInverses, 'Names', names);
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
      ss = StateSpace.setAllParameters(ss, nan);
      
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
      assert(~any(isnan(theta)), 'Theta must be non-nan');
      
      % Dimensions and preallocation
      nParameterMats = length(obj.fixed.systemParam);
      knownParams = cell(nParameterMats, 1);
      
      % Construct the new parameter matricies (including initial values)
      psi = obj.constructPsi(theta);
      
      for iP = 1:nParameterMats
        iName = obj.fixed.systemParam{iP};
        
        constructedParamMat = obj.constructParamMat(psi, iName);
        
        if obj.fixed.timeInvariant
          knownParams{iP} = constructedParamMat;
        else
          paramStruct = struct;
          paramStruct.([iName 't']) = constructedParamMat;
          paramStruct.(['tau' iName]) = obj.fixed.tau.(iName);
          knownParams{iP} = paramStruct;
        end        
      end
      
      % Create StateSpace using parameters just constructed
      ss = obj.cellParams2ss(knownParams);
      
      % Set initial values 
      % I think I need stationaryStates to be tracked by ThetaMap here. 
      % There can be no case where a state switches from stationary to
      % non-stationary of vice-versa. 
      if obj.usingDefaulta0
        a0 = [];
      else
        a0 = obj.constructParamMat(psi, 'a0');
      end
      
      if obj.usingDefaultP0
        P0 = [];
      else
        A0 = obj.fixed.A0;
        R0 = obj.fixed.R0;
        Q0 = obj.constructParamMat(psi, 'Q0');
        P0 = R0 * Q0 * R0';
        P0(A0 * A0' == 1) = Inf;
      end
      
      ss.a0 = a0;
      ss.P0 = P0;    
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
      
      vectorize = @(ssObj) ThetaMap.vectorizeStateSpace(ssObj, ...
        ~obj.usingDefaulta0, ~obj.usingDefaultP0);
      ssParamVec = vectorize(ss);
      lbValues = vectorize(obj.LowerBound);
      ubValues = vectorize(obj.UpperBound);
      indexValues = vectorize(obj.index);
      vecTransIndexes = vectorize(obj.transformationIndex);

      lowerViolation = ~(lbValues < ssParamVec | ~isfinite(ssParamVec) | indexValues == 0);
      upperViolation = ~(ubValues > ssParamVec | ~isfinite(ssParamVec) | indexValues == 0);
      
      if any(lowerViolation)
        lowerViolParams = ss.systemParam(cellfun(@(x) ...
          ~isempty(intersect(obj.index.(x)(:), ...
          unique(indexValues(lowerViolation)))), ss.systemParam));
        lowerViolStr = strjoin(lowerViolParams, ', ');
        error('system2theta:LBound', ...
          'Parameter(s) in %s violate lower bound.', lowerViolStr);
      end
      
      if any(upperViolation)
        upperViolParams = ss.systemParam(cellfun(@(x) ...
          ~isempty(intersect(obj.index.(x)(:), ...
          unique(indexValues(upperViolation)))), ss.systemParam));
        upperViolStr = strjoin(upperViolParams, ', ');
        
        error('system2theta:UBound', ...
          'Parameter(s) in %s violate upper bound.', upperViolStr);
      end
      
      % Loop over psi, identify elements determined by each psi element and
      % compute the inverse of the transformation to get the value
      psi = nan(obj.nPsi, 1);
      for iPsi = 1:obj.nPsi
        iIndexes = indexValues == iPsi;
        nParam = sum(iIndexes);

        iParamValues = ssParamVec(iIndexes);
        iTransformIndexes = vecTransIndexes(iIndexes);
        
        % Get the optimal theta value for each parameter, make sure they match
        iInverses = obj.inverses(iTransformIndexes);
        psiVals = arrayfun(@(x) iInverses{x}(iParamValues(x)), 1:nParam);        
        assert(all(psiVals - psiVals(1) < 1e4 * eps | ~isfinite(psiVals)), ...
          'Transformation inverses result in differing values of theta.');
        psi(iPsi) = psiVals(1);
      end
      
      % Loop over theta, construct from psi
      assert(size(obj.PsiInverse, 1) == obj.nTheta);
      
      % Explicit inverses
      theta = nan(obj.nTheta, 1);
      for iTheta = 1:obj.nTheta
        psiInvInx = find(cellfun(@(psiInx) any(psiInx == iTheta), obj.PsiIndexes));
        if ~isempty(obj.PsiInverse{iTheta})
          theta(iTheta) = obj.PsiInverse{iTheta}(psi, psiInvInx); 
        end
      end
      
      % Numeric inverses for those not specified
      while any(isnan(theta))
        % We want to find the i-th element of theta. To do so, we collect the set of 
        % codetermined elements of theta based on a psi vector. We will solve for that set
        % of theta elements at the same time. 
        iTheta = [];
        iThetaNew = find(isnan(theta), 1);
        while ~isequal(iTheta, iThetaNew)
          iTheta = iThetaNew;
          
          % Find which elements of psi are determined by the current subset of theta that
          % we're going to solve for
          psiInvInx = find(cellfun(@(psiInx) ~isempty(intersect(iTheta, psiInx)), ...
            obj.PsiIndexes));
          
          % Add all elements of theta that determine these elements of psi to the list
          % under consideration
          iThetaNew = unique([obj.PsiIndexes{psiInvInx}]);
        end
        iThetas = iThetaNew;
        
        % Get the indexes needed for the subset of theta to create the psi vector, ie. if
        % we're finding theta(2:4), we need to find the indexes that determine each psi
        % element from this 3-element vector. 
        smallThetaInx = cellfun(@(inx) arrayfun(@(iInx) find(iThetas == iInx), inx), ...
          obj.PsiIndexes(psiInvInx), 'Uniform', false);

        % Take numeric inverses
        computePartialPsi = @(theta) cellfun(@(trans, inx) trans(theta(inx)), ...
          obj.PsiTransformation(psiInvInx), smallThetaInx);
        psiErrors = @(theta) computePartialPsi(theta) - psi(psiInvInx);
        
        theta0 = randn(length(iThetas), 1);
        assert(isnumeric(sum(psiErrors(theta0').^2)));
        verbose = false;
        if verbose 
          plotFcn = {@optimplotfunccount, @optimplotresnorm, ...
          @optimplotstepsize, @optimplotfirstorderopt};
        else
          plotFcn = {};
        end
        
        theta(iThetas) = lsqnonlin(psiErrors, theta0', ... 
          obj.thetaLowerBound(iThetas), obj.thetaUpperBound(iThetas), ...
          optimoptions('LSQNONLIN', 'Display', 'off', ...
          'MaxFunctionEvaluations', 10000*length(iThetas), ...
          'MaxIterations', 10000, ...
          'PlotFcn', {}));
        
        if any(psiErrors(theta(iThetas)') > 1e-4)
          warning('Bad numeric inverse from psi to theta.');
        end
      end
    end
    
    %% Theta restrictions
    function transformedTheta = restrictTheta(obj, theta)
      % Create restricted version of theta
      trans = obj.getThetaTransformations();
      
      transformedTheta = nan(obj.nTheta, 1);
      for iTheta = 1:obj.nTheta
        transformedTheta(iTheta) = trans{iTheta}(theta(iTheta));
      end
    end
    
    function untransformedTheta = unrestrictTheta(obj, theta)
      % Get theta^U given theta
      
      assert(all(theta+eps > obj.thetaLowerBound), 'Theta lower bound violated.');
      assert(all(theta-eps < obj.thetaUpperBound), 'Theta upper bound violated.');
            
      [~, thetaInverses] = obj.getThetaTransformations();
      untransformedTheta = nan(obj.nTheta, 1);
      for iTheta = 1:obj.nTheta
        untransformedTheta(iTheta) = thetaInverses{iTheta}(theta(iTheta));
      end
    end
    
    function GtransformedTheta = thetaUthetaGrad(obj, thetaU)
      % Construct G_{theta^U}(theta)
      [~, ~, thetaUDeriv] = obj.getThetaTransformations();
      GtransformedTheta = zeros(obj.nTheta);
      for iTheta = 1:obj.nTheta
        GtransformedTheta(iTheta, iTheta) = thetaUDeriv{iTheta}(thetaU(iTheta));
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
      
      % Restrict parameter matricies
      for iP = 1:length(obj.LowerBound.systemParam)
        iParam = obj.LowerBound.systemParam{iP};
        
        [trans, inver, transInx, lbMat, ubMat] = ...
          obj.restrictParamMat(ssLB, ssUB, iParam);
        
        % Add transformations
        obj.transformations = [obj.transformations trans];
        obj.inverses = [obj.inverses inver];
        
        obj.transformationIndex.(iParam) = transInx;
        
        % Save new lower and upper bounds
        obj.LowerBound.(iParam) = lbMat;
        obj.UpperBound.(iParam) = ubMat;
      end
            
      % Restrict initial values
      if ~obj.usingDefaulta0
        if isempty(obj.LowerBound.a0)
          % Setting a0 for the first time
          obj.LowerBound.a0 = -Inf(size(obj.LowerBound.a0));
          obj.UpperBound.a0 = Inf(size(obj.UpperBound.a0)); 
        end
        [trans, inver, transInx, lbMat, ubMat] = ...
          obj.restrictParamMat(ssLB, ssUB, 'a0');
        
        % Add transformations
        obj.transformations = [obj.transformations trans];
        obj.inverses = [obj.inverses inver];
        
        obj.transformationIndex.a0 = transInx;
        
        % Save new lower and upper bounds
        obj.LowerBound.a0 = lbMat;
        obj.UpperBound.a0 = ubMat;
      end
      
      if ~obj.usingDefaultP0 && ~isempty(obj.fixed.R0)
        if isempty(obj.LowerBound.P0)
          % Setting A0/R0/Q0 for the first time
          % (4/21) I don't think we can actually get here anymore?
          finiteQ0 = ssLB.Q0;
          finiteQ0(~isfinite(finiteQ0)) = 1; % Value doesn't matter.
          P0 = ssLB.R0 * finiteQ0 * ssLB.R0';
          P0(obj.LowerBound.A0 * obj.LowerBound.A0' == 1) = Inf;
          
          obj.LowerBound.P0 = P0;
          obj.UpperBound.P0 = P0;      
        end
        
        [trans, inver, transInx, lbMat, ubMat] = obj.restrictParamMat(ssLB, ssUB, 'Q0');
        
        % Add transformations
        obj.transformations = [obj.transformations trans];
        obj.inverses = [obj.inverses inver];
        
        obj.transformationIndex.Q0 = transInx;
        
        % Save new lower and upper bounds
        obj.LowerBound.Q0 = lbMat;
        obj.UpperBound.Q0 = ubMat;
      end
    end
    
    function obj = addStructuralRestriction(obj, symbol, lb, ub)
      % Set the bounds on an element of theta
      % 
      % Inputs: 
      %   symbol: which element of theta to restrict. Index, symbol or character accepted.
      %   lb: lower bound 
      %   ub: upper bound
      %
      % If the lower or upper bounds passed are empty, the bound is unchanged. 
 
      % Find the element of theta we'll restrict
      if isnumeric(symbol)
        assert(symbol <= obj.nTheta) 
        iTheta = symbol;
      elseif ischar(symbol)
        iTheta = find(strcmp(symbol, obj.thetaNames));
      elseif isa(symbol, 'sym')
        iTheta = find(strcmp(char(symbol), obj.thetaNames));
      end
      
      % Restrict
      if ~isempty(lb)
        obj.thetaLowerBound(iTheta) = lb;
      end
      if nargin > 3 && ~isempty(ub)
        obj.thetaUpperBound(iTheta) = ub;
      end
    end
    
    function obj = validateThetaMap(obj)
      % Verify that the ThetaMap is valid after user modifications. 
      
      % Minimize the size of theta and psi needed after edits have been made to 
      % index: If the user changes an element to be a function of a different 
      % theta value, remove the old theta value - effectively shift all indexes 
      % down by 1.
      [obj.index, deletedTheta] = ThetaMap.eliminateUnusedIndexes(obj.index, ...
        ~obj.usingDefaulta0, ~obj.usingDefaultP0);
      obj = obj.compressTheta(deletedTheta);

      % Reset nTheta and nPsi if we've added/removed elements
      obj.nTheta = max(cellfun(@max, obj.PsiIndexes));
      obj.nPsi = max(ThetaMap.vectorizeStateSpace(obj.index, ...
        ~obj.usingDefaulta0, ~obj.usingDefaultP0));
      
      % Make sure the theta bounds are big enough
      if size(obj.thetaLowerBound, 1) < obj.nTheta
        obj.thetaLowerBound = [obj.thetaLowerBound; ...
          -Inf(obj.nTheta - size(obj.thetaLowerBound, 1), 1)];
      end  
      if size(obj.thetaUpperBound, 1) < obj.nTheta
        obj.thetaUpperBound = [obj.thetaUpperBound; ...
          Inf(obj.nTheta - size(obj.thetaUpperBound, 1), 1)];
      end
      
      % Remove duplicate and unused transformations
      obj = obj.compressTransformations();      
      
      % Make sure the lower bound is actually below the upper bound and
      % other error checking?
      assert(all(ThetaMap.vectorizeStateSpace(obj.LowerBound, ...
        ~obj.usingDefaulta0, ~obj.usingDefaultP0) <= ...
        ThetaMap.vectorizeStateSpace(obj.UpperBound, ...
        ~obj.usingDefaulta0, ~obj.usingDefaultP0)), ...
        'Elements of LowerBound are greater than UpperBound.');
    end
    
    function obj = compressTheta(obj, deletedTheta)
      % Remove unused elements of theta
      
      % Delete unused index elements, decrement those we're still keeping if
      % we're deleting indexes below them.
      deletedIndexes = obj.PsiIndexes(deletedTheta);
      obj.PsiIndexes(deletedTheta) = [];
      for iPsi = 1:length(obj.PsiIndexes)
        indexSubtract = arrayfun(@(x) sum(x > deletedTheta), obj.PsiIndexes{iPsi});
        obj.PsiIndexes{iPsi} = obj.PsiIndexes{iPsi} - indexSubtract;
      end
      
      % Delete unused transformations and gradients
      obj.PsiTransformation(deletedTheta) = [];
      
      obj.PsiInverse(unique([deletedIndexes{:}])) = [];
      
      % Delete unused parts of theta bounds
      obj.thetaLowerBound(deletedTheta) = [];
      obj.thetaUpperBound(deletedTheta) = [];
    end
    
    function obj = compressTransformations(obj)
      % Removes unused or duplicate transformations.
      % For duplicates, set their indexes to the lower-indexed version and 
      % delete the higher-indexed version.
      
      % Make sure we don't have any transformations on fixed elements
      possibleParamNames = [obj.transformationIndex.systemParam, {'a0', 'Q0'}];
      paramNames = possibleParamNames(...
        [true(1,9) ~obj.usingDefaulta0 ~obj.usingDefaultP0]);
      for iP = 1:length(paramNames)
        obj.transformationIndex.(paramNames{iP})(obj.index.(paramNames{iP}) == 0) = 0;        
      end
      
      % Remove unused transformations:
      % this should be very similar to removing missing index elements but
      % we also need to delete the transformations
      [obj.transformationIndex, unusedTransforms] = ...
        ThetaMap.eliminateUnusedIndexes(obj.transformationIndex, ...
        ~obj.usingDefaulta0, ~obj.usingDefaultP0);
      obj.transformations(unusedTransforms) = [];
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
        for iP = 1:length(paramNames)
          dupInds = arrayfun(@(x) any(x == duplicatesForRemoval), ...
            obj.transformationIndex.(paramNames{iP}));
          obj.transformationIndex.(paramNames{iP})(dupInds) = iTrans;
        end
        
        % Compress indexes
        [obj.transformationIndex, unusedTransforms] = ...
          ThetaMap.eliminateUnusedIndexes(obj.transformationIndex, ...
          ~obj.usingDefaulta0, ~obj.usingDefaultP0);
        assert(isempty(unusedTransforms) || ...
          all(unusedTransforms == duplicatesForRemoval));
        
        % Remove duplicates
        obj.transformations(duplicatesForRemoval) = [];
        obj.inverses(duplicatesForRemoval) = [];
      end
    end
    
    function obj = updateInitial(obj, a0, P0)
      % Set the initial values a0 and P0. 
      %
      % Inputs may contain nans indicating the elements to be estimated. Note
      % that this causes a0 and P0 to be freely estimated. 
      
      % Get the identity transformation to add later
      [trans, inverse] = obj.boundedTransform(-Inf, Inf);
      [transB, inverseB] = obj.boundedTransform(eps * 1e6, Inf);
        
      % Alter a0
      if ~isempty(a0) 
        validateattributes(a0, {'numeric'}, {'size', [obj.m 1]});
        obj.usingDefaulta0 = false;

        % Set the fixed elements
        a0fixed = a0;
        a0fixed(isnan(a0)) = 0;
        obj.fixed.a0 = a0fixed;
        
        % Add to the indexes for the *potentially* new elements of a0
        a0index = zeros(size(a0));
        a0index(isnan(a0)) = obj.nTheta + (1:sum(isnan(a0)));
        obj.index.a0 = a0index;
        obj.nTheta = obj.nTheta + sum(isnan(a0));
        
        % Add an identity transformation for all of the elements just added
        a0transIndex = zeros(size(a0, 1), 1);
        nTransform = length(obj.transformations);
        a0transIndex(isnan(a0)) = nTransform + 1;
        obj.transformationIndex.a0 = a0transIndex;
        
        obj.transformations = [obj.transformations {trans}];
        obj.inverses = [obj.inverses {inverse}];
        
        a0LB = -Inf * ones(size(a0));
        a0LB(isfinite(a0)) = a0(isfinite(a0));
        obj.LowerBound.a0 = a0LB;
        
        a0UB = Inf * ones(size(a0));
        a0UB(isfinite(a0)) = a0(isfinite(a0));
        obj.UpperBound.a0 = a0UB;
        
        % Make sure we're using a0
      else
        obj.usingDefaulta0 = true;
      end
      
      % Alter P0
      if ~isempty(P0) 
        validateattributes(P0, {'numeric'}, {'size', [obj.m obj.m]});
        obj.usingDefaultP0 = false;

        % Set the fixed elements
        P0fixed = P0;
        P0fixed(isnan(P0)) = 0;
        obj.fixed.P0 = P0fixed;
        
        % Add to the indexes for the *potentially* new elements of P0
        P0index = zeros(size(P0));
        nRequiredTheta = sum(sum(sum(isnan(tril(P0)))));
        P0index(isnan(tril(P0))) = obj.nTheta + (1:nRequiredTheta);
        P0index = P0index + P0index' - diag(diag(P0index));
        obj.index.P0 = P0index;
        
        % Add identity and exp transformation for all of the elements just added
        P0transIndex = zeros(size(P0));
        nTransform = length(obj.transformations);
        P0transIndex(isnan(P0)) = nTransform + 1;
        P0transIndex(isnan(diag(diag(P0)))) = nTransform + 2;
        obj.transformationIndex.P0 = P0transIndex;

        obj.transformations = [obj.transformations {trans, transB}];
        obj.inverses = [obj.inverses {inverse, inverseB}];
        
        % Restrict diagonal to be positive
        ssLB = StateSpace.setAllParameters(obj.fixed, -Inf);
        ssUB = StateSpace.setAllParameters(obj.fixed, Inf);
        
        % This needs to handle diffuse states better:
        % Can this mess up A0/R0? Yes. 
        % obj.fixed already has A0/R0 set. Just update Q0.
        Q0LB = -Inf(size(ssLB.Q0));
        Q0LB(1:size(Q0LB,1)+1:end) = eps * 10;
        ssLB.Q0 = Q0LB;
        
        obj = obj.addRestrictions(ssLB, ssUB);
      else
        obj.usingDefaultP0 = true;
      end
      
      % Validate & remove duplicate transformations and potentially unused
      % indexes we just added.
      obj = obj.validateThetaMap();
      
    end
    
    function thetaStr = paramString(obj)
      % Create a cell vector of which parameter each theta element influences
      
      % Find parameters affected
      thetaStr  = cell(obj.nTheta, 1);
      params = obj.fixed.systemParam;
      matParam = repmat({''}, [obj.nTheta, length(params)]);
      for iP = 1:length(params)
        indexes = obj.index.(params{iP});
        matParam(indexes(indexes~=0), iP) = repmat(params(iP), [sum(indexes(:)~=0), 1]);
      end
      
      % Combine into cell of strings
      for iT = 1:obj.nTheta
        goodStrs = matParam(iT,:);
        goodStrs(cellfun(@isempty, goodStrs)) = [];
        thetaStr{iT} = strjoin(goodStrs, ', ');
      end
    end
  end
  
  methods (Hidden)
    function psi = constructPsi(obj, theta)
      % Create psi as a function of theta
      
      psi = nan(obj.nPsi, 1);
      for iPsi = 1:obj.nPsi
        psi(iPsi) = obj.PsiTransformation{iPsi}(theta(obj.PsiIndexes{iPsi})');        
      end      
    end
    
    function constructed = constructParamMat(obj, psi, matName)

      % Create parameter value matrix from fixed and varried values
      
      % Get fixed values
      constructed = obj.fixed.(matName);
      
      % Fill non-fixed values with transformations of theta
      % We don't need to worry about symmetric matricies here since the index
      % matricies should be symmetric as well at this point.
      freeValues = find(logical(obj.index.(matName)));
      if isempty(freeValues)
        return
      end
      
      for jF = freeValues'
        jTrans = obj.transformations{obj.transformationIndex.(matName)(jF)};
        jPsi = psi(obj.index.(matName)(jF));
        constructed(jF) = jTrans(jPsi);
      end
      
    end
    
    function [thetaTrans, thetaInv, thetaDeriv] = getThetaTransformations(obj)
      thetaTrans = cell(obj.nTheta, 1);
      thetaInv = cell(obj.nTheta, 1);
      thetaDeriv = cell(obj.nTheta, 1);
      for iTheta = 1:obj.nTheta
        [thetaTrans{iTheta}, thetaInv{iTheta}, thetaDeriv{iTheta}] = obj.boundedTransform(...
          obj.thetaLowerBound(iTheta), obj.thetaUpperBound(iTheta));
      end
    end
    
    function [newTrans, newInver, transInx, newLBmat, newUBmat] = ...
        restrictParamMat(obj, ssLB, ssUB, iParam)
      % Get the new version of a parameter after new restrictions
      
      % Find the higher lower bound and lower upper bound
      oldLBmat = obj.LowerBound.(iParam);
      passedLBmat = ssLB.(iParam);
      newLBmat = max(oldLBmat, passedLBmat);
      
      oldUBmat = obj.UpperBound.(iParam);
      passedUBmat = ssUB.(iParam);
      newUBmat = min(oldUBmat, passedUBmat);
      
      % Alter transformations - just add a new ones for now, we'll delete the 
      % old ones later in checkThetaMap 
      transInx = obj.transformationIndex.(iParam);
      newTrans = cell(1, numel(newLBmat));
      newInver = cell(1, numel(newLBmat));
      
      additionalTrans = 0;
      for iElem = 1:numel(newLBmat)
        if newLBmat(iElem) ~= oldLBmat(iElem) || newUBmat(iElem) ~= oldUBmat(iElem)
          additionalTrans = additionalTrans + 1;
          [trans, inver] = ThetaMap.boundedTransform(newLBmat(iElem), newUBmat(iElem));
          transInx(iElem) = length(obj.transformations) + additionalTrans;
          
          newTrans{iElem} = trans;
          newInver{iElem} = inver;
        end
      end
      
      newTrans(cellfun(@isempty, newTrans)) = [];
      newInver(cellfun(@isempty, newInver)) = [];
    end
    
    function outStr = getMatList(obj)
      % Utility: create a cell vector of parameters each theta element affects
      
      outStr  = cell(obj.nTheta, 1);
      
      params = obj.fixed.systemParam;
      matParam = repmat({''}, [obj.nTheta, length(params)]);
      for iP = 1:length(params)
        indexes = obj.index.(params{iP});
        matParam(indexes(indexes~=0), iP) = repmat(params(iP), [sum(indexes(:)~=0), 1]);
      end
      
      for iT = 1:obj.nTheta
        goodStrs = matParam(iT,:);
        goodStrs(cellfun(@isempty, goodStrs)) = [];
        outStr{iT} = strjoin(goodStrs, ', ');
      end
    end
    
    function obj = reconstructBoundSystems(obj)
      % Find what the bounds on the system matricies are given the
      % transformations inclueded in the ThetaMap. 
      %
      % Note that this function assumes all transformations are monotonic.
      
      posInfSys = obj.theta2system(Inf([obj.nTheta 1]));
      negInfSys = obj.theta2system(-Inf([obj.nTheta 1]));
      
      lower = negInfSys;
      upper = negInfSys;
      
      for iM = 1:length(lower.systemParam)
        iMat = lower.systemParam{iM};
        lower.(iMat) = min(negInfSys.(iMat), posInfSys.(iMat));
        upper.(iMat) = max(negInfSys.(iMat), posInfSys.(iMat));        
      end
      
      % FIXME: a0 and Q0
      
      obj.LowerBound = lower;
      obj.UpperBound = upper;
    end
    
    function obj = setRestrictions(obj, ssLB, ssUB)
      % Utility function not to be used 
      warning('ThetaMap:setRestrictions', 'setRestrictions not to be used externally.');
      obj.LowerBound = ssLB;
      obj.UpperBound = ssUB;
    end
  end
  
  methods (Static, Hidden)
    %% Constructor helpers
    function validateInputs(fixed, index, transformationIndex, transformations, inverses, opts)
      % Validate inputs
      assert(isa(fixed, 'StateSpace'));
      assert(isa(index, 'StateSpace'));
      assert(isa(transformationIndex, 'StateSpace'));
      
      % Dimensions
      index.checkConformingSystem(transformationIndex);
      
      vectorize = @(ssObj) ThetaMap.vectorizeStateSpace(ssObj, ...
        opts.explicita0, opts.explicitP0);
      
      vecParam = vectorize(transformationIndex);     
      nTransform = max(vecParam);
      
      assert(length(transformations) >= nTransform, ...
        'Insufficient transformations provided for indexes given.');
      assert(length(transformations) == length(inverses), ...
        'Inverses must be provided for every transformation.');

      index.checkConformingSystem(transformationIndex);
      
      vecFixed = vectorize(fixed);
      assert(~any(isnan(vecFixed)), 'Nan not allowed in fixed.');
      vecIndex = vectorize(index);
      assert(~any(isnan(vecIndex)), 'Nan not allowed in index.'); 
      vecTransInx = vectorize(transformationIndex);
      
      % Non-zero elements of fixed are zero in index and vice-versa.
      assert(all(~(vecFixed ~= 0 & vecIndex ~= 0)), ...
        'Parameters determined by theta must be set to 0 in fixed.');
      
      % Make sure all of the state spaces have similarly sized parameters
      assert(length(vecFixed) == length(vecIndex));
      assert(length(vecFixed) == length(vecTransInx));      
    end
    
    function [indexSS, missingInx] = eliminateUnusedIndexes(indexSS, explicita0, explicitP0)
      % Decrement index values of a StateSpace so that therer are no unused
      % integer values from 1 to the maximum value in the system.
      
      assert(isa(indexSS, 'StateSpace'));
      vecIndex = ThetaMap.vectorizeStateSpace(indexSS, explicita0, explicitP0);
      maxVal = max(vecIndex);
      missingInx = setdiff(1:maxVal, vecIndex);
      if isempty(missingInx)
        return
      end
      
      possibleParamNames = [indexSS.systemParam, {'a0', 'Q0'}];
      paramNames = possibleParamNames([true(1,9) explicita0 explicitP0]);
 
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
      newVecIndex = ThetaMap.vectorizeStateSpace(indexSS, explicita0, explicitP0);
      
      assert(newMax == max(newVecIndex));
      assert(isempty(setdiff(1:newMax, newVecIndex)), ...
        'Development error. Index cannot skip elements of psi.');
    end
    
    function index = IndexStateSpace(ssE)
      % Set up index StateSpace for default case where all unknown elements of 
      % the parameters are to be estimated individually
      % 
      % Inputs 
      %   ssE:        StateSpaceEstimation with nan or symbolic values for elements to be 
      %               determined by a ThetaMap
      % Outputs
      %   transIndex: A StateSpace with indexes for each element determined by
      %               theta that indicates the element of thete to be used
      
      paramEstimIndexes = cell(length(ssE.systemParam), 1);
      
      paramVec = [ssE.Z(:); ssE.d(:); ssE.beta(:); ssE.H(:); ...
        ssE.T(:); ssE.c(:); ssE.gamma(:); ssE.R(:); ssE.Q(:)];
      
      if isa(paramVec, 'sym')
        % theta is ordered by symbolic variables, then nan variables
        symTheta = symvar(paramVec);
        % psi will be ordered by symbolic variables, then nan variables
        symPsi = unique(paramVec(has(paramVec, symTheta) & ~isnan(paramVec)));
      else
        symPsi = [];
      end
      
      indexCounter = length(symPsi)+1;

      ssZeros = StateSpace.setAllParameters(ssE, 0);
      for iP = 1:length(ssE.systemParam)
        iParam = ssE.(ssE.systemParam{iP});
        
        psiInds = ssZeros.(ssE.systemParam{iP});
        if ~any(strcmpi(ssE.systemParam{iP}, ssE.symmetricParams))
          % Unrestricted matricies - Z, d, beta, T, c, R
          % We need an element of theta for every missing element
          nRequiredPsi = sum(isnan(iParam(:)));
          psiInds(isnan(iParam)) = indexCounter:indexCounter + nRequiredPsi - 1;
        else
          % Symmetric variance matricies - H & Q. 
          % We only need as many elements of theta as there are missing elements
          % in the lower diagonal of these matricies. 
          nRequiredPsi = sum(sum(sum(isnan(tril(iParam)))));
          psiInds(isnan(tril(iParam))) = indexCounter:indexCounter + nRequiredPsi - 1;
          psiInds = psiInds + psiInds' - diag(diag(psiInds));
        end
        
        matchSym = arrayfun(@(iParamElem) any(iParamElem == symPsi), iParam);
        psiInds(matchSym) = arrayfun(@(iParamElem) find(iParamElem == symPsi), iParam(matchSym));
        
        paramEstimIndexes{iP} = psiInds;        
        indexCounter = indexCounter + nRequiredPsi;
      end
      
      index = StateSpace(paramEstimIndexes{[1 4 5 9]}, ...
        'd', paramEstimIndexes{2}, 'beta', paramEstimIndexes{3}, ...
        'c', paramEstimIndexes{6}, 'gamma', paramEstimIndexes{7}, 'R', paramEstimIndexes{8});
      if ~isempty(ssE.a0)
        a0 = zeros(size(ssE.a0));
        nRequiredPsi = sum(isnan(ssE.a0));
        a0(isnan(ssE.a0)) = indexCounter:indexCounter + (nRequiredPsi-1);
        indexCounter = indexCounter + nRequiredPsi;
        index.a0 = a0;
      end
      
      if ~isempty(ssE.P0) 
        index.P0 = ssE.P0;
        
        Q0inx = zeros(size(ssE.Q0));
        nRequiredPsi = sum(sum(sum(isnan(tril(ssE.Q0)))));

        Q0inx(isnan(tril(ssE.P0))) = indexCounter:indexCounter + nRequiredPsi - 1;
        Q0inx = Q0inx + Q0inx' - diag(diag(Q0inx));
        index.Q0 = Q0inx;
      end
    end
    
    function transIndex = TransformationIndexStateSpace(ssE)
      % Create the default transformationIndex - all parameters values are zeros
      % except where ss is nan, in which case they are ones. 
      % 
      % Inputs 
      %   ss:         StateSpaceEstimation with nan values for elements to be 
      %               determined by a ThetaMap
      % Outputs
      %   transIndex: A StateSpace with indexes for each element determined by
      %               theta that indicates the transformation to be applied
      
      transIndParams = cell(length(ssE.systemParam), 1);
      paramVec = [ssE.Z(:); ssE.d(:); ssE.beta(:); ssE.H(:); ...
        ssE.T(:); ssE.c(:); ssE.gamma(:); ssE.R(:); ssE.Q(:)];
      if isa(paramVec, 'sym')
        symTheta = symvar(paramVec);
      end
      
      % Create parameter matrix of zeros, put a 1 where ss parameters are 
      % missing since all transformation will start as the unit transformation
      ssZeros = StateSpace.setAllParameters(ssE, 0);
      for iP = 1:length(ssE.systemParam)
        iParam = ssE.(ssE.systemParam{iP});

        if isa(iParam, 'sym')
          symInx = has(iParam, symTheta);
        else
          symInx = false(size(iParam));
        end
        
        indexes = ssZeros.(ssE.systemParam{iP});
        indexes(isnan(iParam)| symInx) = 1;
        transIndParams{iP} = indexes;
      end
      
      % Create StateSpace with system parameters
      transIndex = ThetaMap.cellParams2ss(transIndParams);
      
      if ~isempty(ssE.a0)
        a0 = zeros(size(ssE.a0));
        a0(isnan(ssE.a0)) = 1;
        transIndex.a0 = a0;
      end
      
      if ~isempty(ssE.P0) 
        P0inx = ssE.P0;
        P0inx(isnan(P0inx)) = 1;
        transIndex.P0 = P0inx;
      end
      
    end  
  end
  
  methods (Static, Hidden)
    %% Helper functions
    function [trans, inver, deriv] = boundedTransform(lowerBound, upperBound)
      % Generate a restriction transformation from a lower and upper bound
      % Also returns the inverse of the transformation
      % 
      % Inputs
      %   lowerBound: Scalar lower bound
      %   upperBound: Scalar upper bound
      % Outputs
      %   trans: transformation mapping [-Inf, Inf] to the specified interval
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
      % Also accepts cell arrays of function handles. If only one element is a
      % cell array, each is checked to see if they are equal to the non-cell
      % array input. If both elements are cell arrays, they must be the same
      % size and will be checked element-wise.
      % David Kelley, 2017

      nComp = max(length(fn1), length(fn2));
      if iscell(fn1) && iscell(fn2)
        assert(size(fn1) == size(fn2), 'Cell array inputs must be the same size.');
      end
      
      % Needed since we can't use a certain word with sphinx-matlabdomain:
      fnc = ['f' 'unction'];
      
      if iscell(fn1)
        fnInfo1 = cellfun(@functions, fn1);
        fn1Strs = cellfun(@(x) x.(fnc), fnInfo1, 'Uniform', false);
        fn1Workspace = cellfun(@(x) x.workspace{1}, fnInfo1);
      else
        fnInfo1 = functions(fn1);
        fn1Strs = repmat({fnInfo1.(fnc)}, [1 nComp]);
        fn1Workspace = repmat({fnInfo1.workspace{1}}, [1 nComp]);
      end
      
      if iscell(fn2)
        fnInfo2 = cellfun(@functions, fn2, 'Uniform', false);
        fn2Strs = cellfun(@(x) x.(fnc), fnInfo2, 'Uniform', false);
        fn2Workspace = cellfun(@(x) x.workspace{1}, fnInfo2, 'Uniform', false);
      else
        fnInfo2 = functions(fn2);
        fn2Strs = repmat({fnInfo2.(fnc)}, [1 nComp]);
        fn2Workspace = repmat({fnInfo2.workspace{1}}, [1 nComp]);
      end
      
      result = strcmp(fn1Strs, fn2Strs) & cellfun(@isequal, fn1Workspace, fn2Workspace);      
    end
    
    function ssNew = cellParams2ss(cellParams)
      % Create StateSpace with system parameters passed in a cell array
      % 
      % Inputs
      %   cellParams: Cell array with 9 cells: Z, d, H, T, c, R, & Q
      %   ssOld:      StateSpace or ThetaMap with information on handling of
      %               initial values
      % Outputs
      %   ssNew:      StateSpace constructed with new parameters
      
      ssNew = StateSpace(cellParams{[1 4 5 9]}, ...
        'd', cellParams{2}, 'beta', cellParams{3}, ...
        'c', cellParams{6}, 'gamma', cellParams{7}, 'R', cellParams{8});
    end
    
    function vecParam = vectorizeStateSpace(ss, explicita0, explicitP0)
      % Vectorize all parameters of the state space
      % 
      % Arguements: 
      %   ss : StateSpace
      %   explicita0 (boolean) : indicates if a0 is explicit or a function of
      %   the state parameters
      %   explicitP0 (boolean) : indicates if P0 is explicit or a function of
      %   the state parameters
      %
      % Returns: 
      %   vecParam : the vectorized parameters
      
      param = ss.parameters;
      if ~explicita0
        param{10} = [];
      end
      if ~explicitP0
        param{11} = [];
      end
      
      vectors = cellfun(@(x) x(:), param, 'Uniform', false);
      vecParam = vertcat(vectors{:});
    end
  end
end
