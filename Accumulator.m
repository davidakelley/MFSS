classdef Accumulator < AbstractSystem
  % State space augmenting accumulators, enforcing sum and average aggregation
  %
  % Accumulators may be defined for each observable series in the accumulator
  % structure. Three fields need to be defined:
  %   index     - linear indexes of series needing accumulation
  %   calendar  - calendar of observations for accumulated series
  %   horizon   - periods covered by each observation
  % These fields may also be named xi, psi, and Horizon. For more
  % information, see the readme.
  %
  % David Kelley, 2016
  %
  % TODO (12/16/16)
  % ---------------
  %   - Generate funcion handles for ThetaMap
  %   - Create utiltiy methods for standard accumulator creation (descriptive
  %     specification as opposed to explicitly stating calendar/horizon values)
  %   - Write method that checks if a dataset follows the pattern an accumulator
  %     would expect. 
  
  properties % (Access = protected)
    % Linear index of observation dimensions under aggregation
    index
    
    % Timing of low-frequency periods for each index
    calendar
    
    % Length of each low-frequency period for each index
    horizon
  end
  
  properties (Hidden, Access = protected)
    % There are 2 accumulator "types": sum and average. A sum accumulator is
    % type == 0, an average accumulator is type == 1 (for both simple and
    % triangle averages).
    accumulatorTypes
  end
  
  methods
    function obj = Accumulator(index, calendar, horizon)
      % Constructor
      if islogical(index)
        index = find(index);
      end
      assert(length(index) == size(calendar, 2));
      assert(length(index) == size(horizon, 2));
      
      obj.index = index;
      obj.calendar = calendar;
      obj.horizon = horizon;
      
      obj.accumulatorTypes = any(obj.calendar == 0)';
      
      obj.m = [];
      obj.p = max(index);
      obj.g = [];
      obj.n = size(obj.calendar, 1) - 1;
      obj.timeInvariant = false;
    end
    
    function ssNew = augmentStateSpace(obj, ss)
      % Augment the state of a system to enforce accumulator constraints
      %
      % Note that augmenting a system removes initial values. 
      
      obj.checkConformingSystem(ss);
      aug = obj.computeAugSpecification(ss);

      ssNew = obj.buildAccumulatorStateSpace(ss, aug);
    end
    
    function tmNew = augmentThetaMap(obj, tm)
      % Create a ThetaMap that produces StateSpaces that obey the accumulator.
      % The size of the theta vector will stay the same.
      
      obj.checkConformingSystem(tm);

      % The augmentation spec we need to use is from the combined elements of 
      % fixed and index to make sure we get all of the accumulators (ie, taking 
      % into account the elements of Z that are possibly nonzero) and lags we 
      % need (ie, making sure the lag states are detected correctly).
      fullSys = tm.fixed;

      % A placeholder of 1 could potentially mess up LagsInState.
      placeholder = 0.5;
      
      % Add placeholders
      fullZ = tm.fixed.Z;
      fullZ(tm.index.Z ~= 0) = placeholder;
      fullSys.Z = fullZ;
      fullT = tm.fixed.T;
      fullT(tm.index.T ~= 0) = placeholder;
      fullSys.T = fullT;
      fullc = tm.fixed.c;
      fullc(tm.index.c ~= 0) = placeholder;
      fullSys.c = fullc;
      fullR = tm.fixed.R;
      fullR(tm.index.R ~= 0) = placeholder;
      fullSys.R = fullR;
      
      aug = obj.computeAugSpecification(fullSys); 
      
      % Since everything is elementwise in the augmentation, augmenting the
      % fixed system will work just the same as a regular StateSpace. 
      fixedNew = obj.buildAccumulatorStateSpace(tm.fixed, aug);
      
      % Augmenting the index is different. We need to simply copy the rows as
      % with the normal augmentation.
      indexNew = obj.augmentIndex(tm.index, aug);
      
      % The transformation indexes should be constructed somewhat similarly to
      % the regular indexes. They also need to take into acount the appropriate 
      % divisions by the calendar vector elements that will result in additional
      % transformations.
      [transIndexNew, transNew, derivNew, invNew] = obj.augmentTransIndex(tm, aug);
      
      % Construct the new ThetaMap
      tmNew = ThetaMap(fixedNew, indexNew, transIndexNew, ...
        transNew, derivNew, invNew);
    end
    
    function sseNew = augmentStateSpaceEstimation(obj, sse)
      % Augment a StateSpaceEstimation to obey the accumulators. 
      % This is actually simple: augment the system matricies (and let the nans
      % propogate) and augment the ThetaMap. 
      
      sseNew = obj.augmentStateSpace(sse);
      sseNew.ThetaMapping = obj.augmentThetaMap(sse.ThetaMapping);      
    end
  end
  
  methods (Static)
    function accum = GenerateRegular(data, types, horizons)
      % Generate calendar and horizon for regularly spaced observations
      % The first period of the data must be the first period for each
      % accumulator. 
      
      nPer = size(data, 1);
      nSeries = size(data, 2);
      [obsMissing, seriesMissing] = find(~isnan(data));
      
      % Accumulator definitions go one past the data length for Kalman filter
      cal = nan(nPer+1, nSeries);
      hor = nan(nPer+1, nSeries);
      
      for iSer = unique(seriesMissing)'
        % For each series with missing data, find the frequency it repeats at
        % and create the appropriate calendar. Use the user-determined horizon.
        dataFreq = unique(diff(obsMissing(seriesMissing == iSer)));
        
        assert(isscalar(dataFreq), 'Irregular data pattern in series %d', iSer);
        if dataFreq == 1
          % Series doesn't need accumulation, observed every period
          continue
        end
        
        % Create calendars
        if strcmpi(types{iSer}, 'avg')
          % Average accumulators: repeated counting from 1 to frequency
          tempCal = repmat((1:dataFreq)', [ceil(nPer/dataFreq)+1 1]);
          
        else
          % Sum accumulator: zeros with a one at the end of each period
          tempCal = repmat([zeros(dataFreq-1, 1); 1], [ceil(nPer/dataFreq)+1 1]);
          if dataFreq ~= horizons(iSer)
            % There really isn't a horizon for the sum accumulators, but let the
            % user know if the data doesn't match what they expected. 
            warning(['Sum accumulator does not have expected horizon. ' ... 
                     'Created using pattern in data.']);
          end
        end
        
        % Assign outputs
        cal(:, iSer) = tempCal(1:nPer+1);
        hor(:, iSer) = horizons(iSer);
      end
      
      % Select series that really need accumulation, create object.
      inx = all(~isnan(cal));
      accum = Accumulator(inx, cal(:,inx), hor(:,inx));
    end
  end
  
  methods (Hidden)
    %% Inner methods for using accumulators
    function augSpec = computeAugSpecification(obj, ss)
      % Find the states and corresponding observations that need the
      % accumulator and prepare for augmenting that system. This is neccessary
      % since we will be augmenting a system according to the accumulators used
      % in a different system with the ThetaMap.index StateSpace.
      augSpec = struct;
      augSpec.m = struct;
      augSpec.m.original = ss.m;
      
      % Set tau if it hasn't been set yet
      if isempty(ss.tau)
        ss.n = obj.n;
        ss = ss.setInvariantTau();
      end
      
      [augmentedStates, indexUsedInx] = obj.findUsedAccum(ss);
            
      if size(augmentedStates, 2) > 1
        warning('Flip!');
        augmentedStates = augmentedStates';
      end
      if size(indexUsedInx, 2) > 1
        warning('Flip!');
        indexUsedInx = indexUsedInx';
      end
      
      % Find the different accumulators we need: anything that has the same 
      % (state, type, horizon, calendar) doesn't need to be done twice. 
      possibleAccumDefs = [augmentedStates'; ...
        obj.accumulatorTypes(indexUsedInx)'; ...
        obj.horizon(:, indexUsedInx); ...
        obj.calendar(:, indexUsedInx)];
      [APsiSort, unsort, sortUsedAccum] = unique(possibleAccumDefs', 'rows');
      [~, order] = sort(indexUsedInx(sortUsedAccum(unsort)));
      
      % Added 1/5/17 to try and make final order of states make more sense.
      % Originally did not have unsort. 
      usedDefinitions = sortUsedAccum(unsort(order));
      APsi = APsiSort(usedDefinitions, :);
      
      nPer = size(obj.horizon, 1);
      
      augSpec.accumulatorTypes = APsi(:, 2)';
      usedHorizon = APsi(:, 3:nPer + 2)';
      usedCalendar = APsi(:, nPer + 3:end)';

      augSpec.baseFreqState = APsi(:, 1)';

      [augSpec.addLagsFrom, LagRowPos, augSpec.m.withLag] = ...
        obj.determineNeededLags(ss, augSpec.baseFreqState, usedHorizon);
      
      augSpec.nAccumulatorStates = size(usedCalendar, 2);
      
      augSpec.m.final = augSpec.m.withLag + augSpec.nAccumulatorStates;
      
      % Compute where Z elements should be moved from and to (linear indexes)
      % Original Z elements: any nonzero elements of rows of Z for used
      % accumulated observables
      accumulatorObservationsZinx = obj.index(indexUsedInx(usedDefinitions))';
      
      Zspec.originIndexes = sub2ind(size(ss.Z), ...
        accumulatorObservationsZinx, augSpec.baseFreqState');
      
      % New Z elements: same observables, now they just load onto the states in
      % order that the new accumulator states have been defined. 
      newStates = augSpec.m.withLag + (1:augSpec.nAccumulatorStates)';
      Zspec.finalIndexes = sub2ind([size(ss.Z, 1) augSpec.m.final], ...
        accumulatorObservationsZinx, newStates);
      
      augSpec.Z = Zspec;
      
      % T augmentation:
      Tspec = struct;
      Ttypes   = [ss.tau.T usedCalendar usedHorizon];
      [uniqueTs, ~, Tspec.newtau] = unique(Ttypes, 'rows');
      
      Tspec.oldtau = uniqueTs(:, 1);
      Tspec.cal = uniqueTs(:, 1 + (1:augSpec.nAccumulatorStates));
      Tspec.hor = uniqueTs(:, 1 + augSpec.nAccumulatorStates + (1:augSpec.nAccumulatorStates));
      
      Tspec.LagRowPos = LagRowPos;
      
      augSpec.T = Tspec;
      
      % c augmentation
      cSpec = struct;
      
      ctypes   = [ss.tau.c usedCalendar];
      [uniquecs, ~, cSpec.newtau] = unique(ctypes, 'rows');
      cSpec.oldtau = uniquecs(:, 1);
      cSpec.cal = uniquecs(:, 1 + (1:augSpec.nAccumulatorStates));      
      
      augSpec.c = cSpec;
      
      % R augmentation
      Rspec = struct;
      
      Rtypes   = [ss.tau.R usedCalendar];
      [uniqueRs, ~, Rspec.newtau] = unique(Rtypes, 'rows');
      Rspec.oldtau = uniqueRs(:, 1);
      Rspec.cal = uniqueRs(:, 1 + (1:augSpec.nAccumulatorStates));
      
      augSpec.R = Rspec;
    end
    
    function ssNew = buildAccumulatorStateSpace(obj, ss, aug)
      % Compute the new state parameters given a specification of which states
      % need to be augmented. 
      % Set up structure of new system
            
      % Set tau if it hasn't been set yet
      if isempty(ss.tau)
        ss.n = obj.n;
        ss = ss.setInvariantTau();
      end
      
      ssLag = obj.addLags(ss, aug.addLagsFrom, aug.m.withLag);
      
      Z.Zt = obj.augmentParamZ(ssLag.Z, aug);
      Z.tauZ = ssLag.tau.Z;
      d.dt = ssLag.d;
      d.taud = ssLag.tau.d;
      H.Ht = ssLag.H;
      H.tauH = ssLag.tau.H;
      
      T.Tt = obj.augmentParamT(ssLag.T, aug);
      T.tauT = aug.T.newtau;
      c.ct = obj.augmentParamc(ssLag.c, aug);
      c.tauc = aug.c.newtau;
      R.Rt = obj.augmentParamR(ssLag.R, aug);
      R.tauR = aug.R.newtau;
      Q.Qt = ssLag.Q;
      Q.tauQ = ssLag.tau.Q;
      
      % Create new system
      ssNew = StateSpace(Z, d, H, T, c, R, Q);
    end
    
    %% ThetaMap system augmentation methods
    function newIndex = augmentIndex(obj, index, aug)
      % Augment index: create a new StateSpace with appropriate indexes tracking
      % which elements of the theta vector map to the StateSpace parameters. 
      % 
      % For Z we need to move the indexes to the accumulator states and replace 
      % them with zeros - this is taken care of with the standard augmentation.
      % For T, c and R we simply need to copy the high frequency row to the 
      % correct accumulator state.
            
      newIndex = obj.buildAccumulatorStateSpace(index, aug);
      
      % Both accumulators are the same here. We're just copying integers.
      for iAccum = 1:aug.nAccumulatorStates
        iAccumState = aug.m.original + iAccum;
        
        % Copy rows of base part of T, c & R to the accumulator states
        newIndex.T(iAccumState, :, :) = newIndex.T(aug.baseFreqState(iAccum), :, :);
        newIndex.c(iAccumState, :) = newIndex.c(aug.baseFreqState(iAccum), :);
        newIndex.R(iAccumState, :, :) = newIndex.R(aug.baseFreqState(iAccum), :, :);
      end
      
    end
    
    function [transIndex, trans, deriv, inv] = augmentTransIndex(obj, tm, aug)
      % Augment transformation index: create a new StateSpace of indexes with 
      % the associated transformations to enforce the accumulators. 
      %
      % For Z, we're just moving elements so there's no change to the
      % transformations. For T, c & R, the transformation indexes propogate the 
      % same as the theta indexes. 
      % The sum accumulators only copy parameters so we don't need to alter the 
      % transformations. The average accumulators require the application of the
      % same linear transformation that is applied during a StateSpace 
      % augmentation after the existing transformations are made. 
      
      transIndex = obj.augmentIndex(tm.transformationIndex, aug);
      
      for iAcc = 1:aug.nAccumulatorStates
        if aug.accumulatorTypes(iAcc) ~= 1
          % No alteration to transformations neccessary for sum accumulators
          continue
        end
        
        iState = aug.m.withLag + iAcc;
        
        % Modify T transformations
        vec = @(M) reshape(M, [], 1);

        nCol = aug.m.original;
        nSlices = size(transIndex.T, 3);
        nTTrans = nCol * nSlices;
        Tinds = sub2ind(size(transIndex.T), ...
          repmat(iState, [nTTrans 1]), vec(repmat(1:nCol, [nSlices 1])'), ...
          vec(repmat(1:size(transIndex.T, 3), [nCol 1])));
        
        % Figure out what elements of T are getting added to them and mutliplied
        % by to create the new parameters:
        addendT = obj.augmentParamT(zeros(size(tm.fixed.T)), aug);
        factorT = obj.augmentParamT(ones(size(tm.fixed.T)), aug) - addendT;        
        
        % Compose new functions that take into account the transformation
        % occuring in the augmentation:
        newTrans = cell(1, numel(Tinds));
        newDeriv = cell(1, numel(Tinds));
        newInv = cell(1, numel(Tinds));
        for iTrans = 1:numel(Tinds)
          iT = Tinds(iTrans);
          iTransInd = transIndex.T(iT);
          
          if iTransInd == 0
            % Not a function of theta
            continue
          end
          if factorT(iT) == 1 && addendT(iT) == 0
            % Simply copy of elements, don't nest the function 
            continue
          end
          
          % Compose transformation by multiplying then adding
          newTrans{iTrans} = obj.composeLinearFunc(...
            tm.transformations{iTransInd}, factorT(iT), addendT(iT));
          % Compose the derivative with the chain rule
          newDeriv{iTrans} = obj.composeLinearFunc(...
            tm.derivatives{iTransInd}, factorT(iT), 0);
          % "Undo" the linear transformation by first subtracting what was added
          % then dividing by what was multiplied. % TODO: simplify
          invTemp = obj.composeLinearFunc(...
            tm.inverses{iTransInd}, 1, -addendT(iT)); 
          newInv{iTrans} = obj.composeLinearFunc(...
            invTemp, 1/factorT(iT), 0);
          
          % Move transformationIndex of the element we just changed
          nTransformations = max(transIndex.vectorizedParameters());
          transIndex.T(iT) = nTransformations + 1;
        end
                
        % Append the used transformations to the lists:
        trans = [tm.transformations newTrans(~cellfun(@isempty, newTrans))];
        deriv = [tm.derivatives newDeriv(~cellfun(@isempty, newDeriv))];
        inv = [tm.inverses newInv(~cellfun(@isempty, newInv))];
      end
      
    end
    
    function newFunc = composeLinearFunc(func, A, B)
      % A helper function for performing a linear transformation to the result
      % from an arbitrary function handle.
      % Provides g(x) = A * f(x) + B
      
      newFunc = @(x) A * func(x) + B;
    end
    
    %% Accumulator helper methods
    function [accumulatedStates, indexUsedInx] = findUsedAccum(obj, ss)
      % Find the elements of Z that care about the accumulators - anything
      % that loads onto an observation that has an associated accumulator
      usedZaccumLoc = any(ss.Z(obj.index,:,:) ~= 0, 3);
      [indexUsedInx, accumulatedStates] = find(usedZaccumLoc);  
    end
    
    function [lagsColPos, lagRowPos, mWLag] = determineNeededLags(obj, ...
        ss, hiFreqStates, augHorizon)
      % Checking we have the correct number of lags of the states that need
      % accumulation to be compatible with Horizon
      % 
      % ss - StateSpace that is being augmented
      % hiFreqStates - 
      % augHorizon - 
      
      % TODO: move lagRowPos to separate function (ss.LagsInState)
      
      % How many lags we need to add by each state we're concerned with
      usedStates = unique(hiFreqStates);
      AddLagsByState = zeros(length(usedStates), 1);

      % Index vectors for new lags we're going to add. The first element of 
      % these will be the state we're making lags of, lags come afterwards
      lagsColPos = zeros(length(usedStates), max(max(augHorizon)));
      lagRowPos = zeros(length(usedStates), max(max(augHorizon)));
      
      mWLag = ss.m;
      for iSt = 1:length(usedStates)
        % We need to find all of the previous lags of a state so that we know
        % which lag is the last one we have so far. Then we can start with that
        % state and add additional lags until we have the number we need. 
        [nLags, lagPositions] = ss.LagsInState(usedStates(iSt));
        
        % Locate existing lags states
        lagRowPos(iSt, 1:1 + nLags) = [usedStates(iSt) lagPositions'];
        
        % Add lags for this state to get to the max horizon of this accumulator 
        % and add as many as we need past the existing size of the state
        relevantHorizons = hiFreqStates == usedStates(iSt);
        iHorizon = max(max(augHorizon(:, relevantHorizons)));
        
        % We have (nLags+1) avaliable lags since we get one for free in the
        % transition equation. 
        % max(S) is different from the GeneralizedKFilterSmoother. Test this. 
        
        % Do we need H + S - 1 or H - 1?
        needLags = iHorizon - 1;
        haveLags = nLags + 1;
        addLags = max(needLags - haveLags, 0);
        
        extraLagsInds = 1:addLags;
        lagRowPos(iSt, 1 + nLags + extraLagsInds) = mWLag + extraLagsInds;
        
        if addLags > 0
          % Find the last lag we have already
          if ~isempty(lagPositions)
            lastLag = lagPositions(end);
          else
            lastLag = usedStates(iSt);
          end
          lagsColPos(iSt, 1 + nLags + 1) = lastLag;
        end
        if addLags > 1
          % Stick extra lags on the end of the current state
          tempInx = 1 + nLags + 1 + (1:addLags-1);
          lagsColPos(iSt, tempInx) = mWLag + extraLagsInds(1:end-1);
        end
        % Replace the above with something like this:
        % lagChain = [lastLag (mWLag + extraLagsInds(1:end-1))];
        % lagsColPos(iSt, 1+nLags+(1:iAddLags)) = lagChain;
        
        AddLagsByState(iSt) = addLags;
        mWLag = mWLag + addLags;
      end
            
      lagsColPos = reshape(lagsColPos(lagsColPos ~= 0), [], 1);
    end
    
    %% Helper methods
    function returnFlag = checkConformingSystem(obj, sys)
      % Check if the dimensions of a system match the current object
      assert(isa(sys, 'AbstractSystem'));
      
      assert(obj.p <= sys.p, 'System has fewer observables than needed for accumulator.');
      % No need to check m or g
      if ~sys.timeInvariant
        assert(obj.n == obj.n, 'Time dimension mismatch (n).');
      end

      returnFlag = true;
    end
  end
  
  methods (Static, Hidden)
    %% StateSpace parameter augmentation methods
    function ss = addLags(ss, addLagsFrom, mWithLag)
      % Add the needed lags to the system matricies.

      if ss.m == mWithLag
        % No need to add lags, don't expand the state
        return
      end
      
      % Z - add zeros to the right
      Z.Zt = zeros(ss.p, mWithLag, size(ss.Z, 3));
      Z.Zt(:, 1:ss.m, :) = ss.Z;
      Z.tauZ = ss.tau.Z;
      
      % d and H - do nothing 
      d.dt = ss.d;
      d.taud = ss.tau.d;
      H.Ht = ss.H;
      H.tauH = ss.tau.H;
      
      % T - Add ones to transmit lags
      T.Tt = zeros(mWithLag, mWithLag, size(ss.T, 3));
      T.Tt(1:ss.m, 1:ss.m, :) = ss.T;
      lagIndexes = sub2ind(...
        [mWithLag, mWithLag], (ss.m+1:mWithLag)', addLagsFrom);
      % FIXME: I don't think this handles slices correctly:
      T.Tt(lagIndexes) = 1;
      T.tauT = ss.tau.T;
      
      % c - just add zeros below
      c.ct = zeros(mWithLag, size(ss.c, 3));
      c.ct(1:ss.m, :) = ss.c;
      c.tauc = ss.tau.c;
      
      % R - just add zeros below
      R.Rt = zeros(mWithLag, ss.g, size(ss.R, 3));
      R.Rt(1:ss.m, :, :) = ss.R;
      R.tauR = ss.tau.R;
      
      % Q - do nothing
      Q.Qt = ss.Q;
      Q.tauQ = ss.tau.Q;
      
      % Create new system
      ss = StateSpace(Z, d, H, T, c, R, Q);      
    end
    
    function newT = augmentParamT(T, aug)
      % Construct new T matrix
      mNew = aug.m.withLag + aug.nAccumulatorStates;

      states = unique(aug.baseFreqState);  % used to be: aug.augmentedStates
      
      % Add accumulator elements
      Tslices = size(aug.T.oldtau, 1);
      newT = zeros(mNew, mNew, Tslices);
      for iT = 1:Tslices
        % Get the part of T already defined
        newT(1:aug.m.withLag, 1:aug.m.withLag, iT) = T(:, :, aug.T.oldtau(iT));
        
        % Define the new transition equations for the accumulator states
        for iAccum = 1:aug.nAccumulatorStates
          iState = aug.m.withLag + iAccum;
          iCal = aug.T.cal(iT, iAccum);
          iHor = aug.T.hor(iT, iAccum);
          
          hiFreqTelements = T(aug.baseFreqState(iAccum), :, aug.T.oldtau(iT));
          
          if aug.accumulatorTypes(iAccum) == 1
            % Sum accumulator
            newT(iState, 1:aug.m.withLag, iT) = hiFreqTelements;
            newT(iState, iState, iT) = iCal;
            
          else
            % Average accumulator
            newT(iState, 1:aug.m.withLag, iT) = (1 / iCal) * hiFreqTelements;
            newT(iState, iState, iT) = (iCal - 1) / iCal;
            
            if iHor > 1
              % Triangle accumulator - we need to add 1/cal to each accumulated
              % state element's loading on the high-frequency component. 
              
              % What is LagRowPos? What is this doing?
              iCols = aug.T.LagRowPos(aug.baseFreqState(iAccum) == states, 1:iHor - 1);
              
              newT(iState, iCols, iT) = newT(iState, iCols, iT) + (1/iCal);
            end
          end
        end
      end
    end
    
    function newc = augmentParamc(c, aug)
      % Construct new c vector      
      cSlices = size(aug.c.oldtau, 1);
      
      newc = zeros(aug.m.final, cSlices);
      for iC = 1:cSlices
        % Get the part of c already defined
        newc(1:aug.m.withLag, iC) = c(:, aug.c.oldtau(iC));
        for iAccum = 1:aug.nAccumulatorStates
          iState = aug.m.withLag + iAccum;
          
          hiFreqElements = c(aug.baseFreqState(iAccum), aug.c.oldtau(iC));
          
          if aug.accumulatorTypes(iAccum) == 1
            % Sum accumulator
            newc(iState, iC) = hiFreqElements;
          else
            % Average accumulator
            newc(iState, iC) = (1/aug.c.cal(iC, iAccum)) * hiFreqElements;
          end
        end
      end
    end
    
    function newR = augmentParamR(R, aug)
      % Construct new R matrix

      nShocks = size(R, 2);      
      
      Rslices = size(aug.c.oldtau, 1);
      newR = zeros(aug.m.final, nShocks, Rslices);
      for jj = 1:Rslices

        % Get the part of R already defined
        newR(1:aug.m.withLag, :, jj) = R(:, :, aug.R.oldtau(jj));
        
        for iAccum = 1:aug.nAccumulatorStates
          iState = aug.m.withLag + iAccum;
          
          hiFreqRElements = R(aug.baseFreqState(iAccum), :, aug.R.oldtau(jj));
                  
          if aug.accumulatorTypes(iAccum) == 1
            % Sum accumulator
            newR(iState,:,jj) = hiFreqRElements;
          else
            % Average accumulator
            newR(iState,:,jj) = (1/aug.R.cal(jj,iAccum)) * hiFreqRElements;
          end
        end
      end
    end
    
    function newZ = augmentParamZ(Z, aug)
      % Construct new Z matrix - we're just moving elements around.
      newZ = zeros(size(Z, 1), aug.m.final);
      newZ(aug.Z.finalIndexes) = Z(aug.Z.originIndexes);
      Z(aug.Z.originIndexes) = 0;
      newZ(1:size(Z, 1), 1:size(Z, 2), :) = Z;      
    end
  end
end