classdef Accumulator < AbstractSystem
  % State space augmenting accumulators, enforcing sum and average aggregation.
  %
  % Accumulators may be defined for each observable series in the accumulator
  % structure. Three fields need to be defined:
  %   index     - linear indexes of series needing accumulation
  %   calendar  - calendar of observations for accumulated series
  %   horizon   - periods covered by each observation
  % These fields may also be named xi, psi, and Horizon.
  %
  % David Kelley, 2016-2018
  
  properties
    % Linear index of observation dimensions under aggregation
    index
    
    % Timing of low-frequency periods for each index
    calendar
    
    % Length of each low-frequency period for each index
    horizon
  end
  
  properties (Hidden, Access = public)
    % There are 2 accumulator "types": sum and average. A sum accumulator is
    % type == 1, an average accumulator is type == 0 (for both simple and
    % triangle averages).
    accumulatorTypes
  end
  
  methods
    function obj = Accumulator(index, calendar, horizon)
      % Accumulator constructor
      %
      % Arguments:
      %   index (double): linear index of series needing aggregation
      %   calendar (double): calendar variable specific to accumulator type
      %   horizon (double): number of high-freq periods in low-freq period
      % Returns:
      %   obj (Accumulator): Accumulator object
      
      if islogical(index)
        index = find(index);
      end
      assert(length(index) == size(calendar, 2));
      assert(length(index) == size(horizon, 2));
      
      obj.index = index;
      obj.calendar = calendar;
      obj.horizon = horizon;
      
      obj.accumulatorTypes = any(obj.calendar == 0)';
      
      obj.p = max(index);
      obj.n = size(obj.calendar, 1) - 1;
      obj.timeInvariant = false;
    end
    
    function ssNew = augmentStateSpace(obj, ss)
      % Augment the state of a system to enforce accumulator constraints
      %
      % Arguments:
      %   ss (StateSpace): StateSpace to augment
      % Returns:
      %   ssNew (StateSpace): augmented StateSpace
      %
      % Note that augmenting a system removes initial values.
      
      if isempty(obj.index)
        ssNew = ss;
        return
      end
      
      obj.checkConformingSystem(ss);
      aug = obj.computeAugSpecification(ss);
      ssNew = obj.buildAccumulatorStateSpace(ss, aug);
    end
    
    function tmNew = augmentThetaMap(obj, tm)
      % Augment a ThetaMap so that it produces an augmented StateSpace.
      %
      % Arguments:
      %     tm (ThetaMap): ThetaMap to augment
      % Returns:
      %     tmNew (ThetaMap): augmented ThetaMap
      %
      % The size of the theta vector will stay the same.
      
      if isempty(obj.index)
        tmNew = tm;
        return
      end
      
      obj.checkConformingSystem(tm);
      aug = obj.comptueThetaMapAugSpecification(tm);
      
      % Since everything is elementwise in the augmentation, augmenting the
      % fixed system will work almost the same as a regular StateSpace.
      fixedNew = obj.buildAccumulatorStateSpace(tm.fixed, aug);
      
      % Augmenting the index is different. We need to simply copy the rows as
      % with the normal augmentation.
      indexNew = obj.augmentIndex(tm.index, aug);
      
      % We will need to delete the accumulated state elements in fixed that have
      % gotten the (1/cal) elements added so they don't conflict with the index
      % elements.
      fixedNew.T(indexNew.T ~= 0) = 0;
      
      % The transformation indexes should be constructed somewhat similarly to
      % the regular indexes. They also need to take into acount the appropriate
      % divisions by the calendar vector elements that will result in additional
      % transformations.
      [transIndexNew, transNew, invNew] = obj.augmentTransIndex(tm, aug);
      
      % Construct the new ThetaMap
      tmNew = ThetaMap(fixedNew, indexNew, transIndexNew, ...
        transNew, invNew, ...
        'explicita0', ~tm.usingDefaulta0, 'explicitP0', ~tm.usingDefaultP0);
      
      % The new ThetaMap should have the same theta -> psi properties as the old
      tmNew.PsiTransformation = tm.PsiTransformation;
      tmNew.PsiInverse = tm.PsiInverse;
      tmNew.PsiIndexes = tm.PsiIndexes;
      tmNew.thetaLowerBound = tm.thetaLowerBound;
      tmNew.thetaUpperBound = tm.thetaUpperBound;
      tmNew.thetaNames = tm.thetaNames;
      tmNew = tmNew.addRestrictions(obj.buildAccumulatorStateSpace(tm.LowerBound, aug), ...
        obj.buildAccumulatorStateSpace(tm.UpperBound, aug));
      
      % Have to use internal ThetaMap method to set nTheta
      tmNew = tmNew.validateThetaMap();
    end
    
    function sseNew = augmentStateSpaceEstimation(obj, sse)
      % Augment a StateSpaceEstimation to obey the accumulators.
      %
      % Arguments:
      %     sse (StateSpaceEstimation): StateSpaceEstimation to augment
      % Returns:
      %     sseNew (StateSpaceEstimation): augmented StateSpaceEstimation
      
      % This is actually simple: augment the system matricies (and let the nans
      % propogate) and augment the ThetaMap.
      
      if isempty(obj.index)
        sseNew = sse;
        return
      end
      
      % Agument the ThetaMap
      newTM = obj.augmentThetaMap(sse.ThetaMapping);
      
      % Augment the parameters but make sure to use the correct aug spec.
      aug = obj.comptueThetaMapAugSpecification(sse.ThetaMapping);
      ssNew = obj.buildAccumulatorStateSpace(sse, aug);
      
      % Create new StateSpaceEstimation
      sseNew = sse;
      sseNew.Z = ssNew.Z;
      sseNew.T = ssNew.T;
      sseNew.c = ssNew.c;
      sseNew.gamma = ssNew.gamma;
      sseNew.R = ssNew.R;
      
      sseNew.tau = ssNew.tau;
      
      sseNew.ThetaMapping = newTM;
      
      sseNew.m = aug.m.final;
      sseNew.timeInvariant = ssNew.timeInvariant;
      sseNew.n = ssNew.n;
      sseNew.tau = ssNew.tau;
    end
  end
  
  methods (Static)
    function accum = GenerateRegular(data, types, horizons)
      % Generate calendar and horizon for regularly spaced observations
      %
      % Arguments:
      %     data (double): observed data (y) in sample
      %     types (cell): cell array of strings indicating 'sum' or 'avg' accumulators
      %     horizons (double): horizon of each accumulator
      % Returns:
      %     accum (Accumulator): accumulator
      %
      % Automatically detects the alignment of each low-frequency series. 
      
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
          tempCal = repmat((1:dataFreq)', [ceil(nPer/dataFreq)+2 1]);
          
        else
          % Sum accumulator: all ones except for a zero in the first period
          tempCal = repmat([0; ones(dataFreq-1, 1)], [ceil(nPer/dataFreq)+2 1]);
          if dataFreq ~= horizons(iSer)
            % There really isn't a horizon for the sum accumulators, but let the
            % user know if the data doesn't match what they expected.
            warning(['Sum accumulator does not have expected horizon. ' ...
              'Created using pattern in data.']);
          end
        end
        
        % Assign outputs
        firstObsPer = find(~isnan(data(:, iSer)), 1, 'first');
        calStart = dataFreq - mod(firstObsPer, dataFreq) + 1;
        cal(:, iSer) = tempCal(calStart + (0:nPer));
        hor(:, iSer) = horizons(iSer);
      end
      
      % Select series that really need accumulation, create object.
      inx = all(~isnan(cal));
      accum = Accumulator(inx, cal(:,inx), hor(:,inx));
    end
    
    function accum = GenerateFromDates(dates, index, frequencies, types)
      % Generate object from a vector of dates and other properties
      %
      % Arguments: 
      %     dates (datenum): vector of dates for each observation
      %     index (double): index of series that need accumulators
      %     frequencies (cell): cell array of strings for accumulated
      %     series. 
      %     types (cell): type of accumulator used
      % Returns:       
      %     accum (Accumulator): accumulator
      %
      % Automatically detects the alignment of each low-frequency series. 
      
      assert(issorted(dates), 'Date vector must be sorted');
      if max(diff(dates)) > 3 && max(diff(dates)) < 27
        warning(['This function was designed for daily data. ' ...
          'If used for weekly data, ensure the end of year period classifications are correct.']);
      elseif max(diff(dates)) >= 31
        warning(['If using data at a monthly frequency or lower, ' ...
          'Accumulator.GenerateRegular is likely the better way to create an accumulator object']);
      end
      
      % Append one extra day to allow for filter timing. 
      % This assumes that the last date is the end of a period, but that
      % was effectively an assumption already. 
      dates(end+1) = dates(end) + 1;
      
      % Create low-frequency period variables
      [year, monthOfYear, dayOfMonth] = datevec(dates);
      quarterOfYear = ceil(monthOfYear / 3);
      semiannualOfYear = monthOfYear <= 6;

      % Define periods by combinations of year and variable within year
      nInx = length(index);
      cals = nan(length(dates),nInx);
      hors = nan(length(dates),nInx);
      for iInx = 1:nInx
        switch frequencies{iInx}
          case 'Annual'
            lowFDefinition = year;
          case 'Semiannual'
            lowFDefinition = [year semiannualOfYear];
          case 'Quarter'
            lowFDefinition = [year quarterOfYear];
          case 'Month'
            lowFDefinition = [year monthOfYear];
          otherwise 
            error(['Supported accumulation frequencies are ' ...
              'Annual, Semiannual, Quarter, and Month']);
        end
        [~, ~, i_accumm] = unique(lowFDefinition, 'rows', 'stable');
        % Create vector that tracks first high-freq period of each low-freq
        % period, and the P1 index for each low-freq period. 
        firstOfLowFPeriod = [true; diff(i_accumm)];
        P1 = [find(firstOfLowFPeriod); length(dates)+1];
        P1Reps = arrayfun(@(x) repmat(P1(x), P1(x+1)-P1(x), 1), 1:length(P1)-1, ...
          'Uniform', false);
        pt = cat(1, P1Reps{:});
        
        switch types{iInx}
          case 'sum'
            cals(:,iInx) = 1 - firstOfLowFPeriod;
            hors(:,iInx) = ones(length(dates),1);
          case 'avg'
            cals(:,iInx) = (1:length(dates))' - pt + 1;
            hors(:,iInx) = ones(length(dates),1);
          otherwise
            error('Unknown accumulator type');
        end
      end
      
      accum = Accumulator(index, cals, hors);
    end
    
    function calendar = group2calendar(group)
      % Utility method to generate average calendar from a group indicator.
      %
      % Arguments:
      %     group (double): vector of group (ie, month of year)
      % Returns:
      %     calendar (double): calendar variable for average accumulator
      
      calendar = nan(size(group));
      for iVec = 1:size(group,2)
        counter = 1;
        for iObs = 1:size(group, 1)
          calendar(iObs,iVec) = counter;
          if iObs ~= size(group, 1) && group(iObs,iVec) == group(iObs+1,iVec)
            counter = counter + 1;
          else
            counter = 1;
          end
        end
      end
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
      m = struct;
      m.original = ss.m;
      
      % Set tau if it hasn't been set yet
      if isempty(ss.tau)
        ss.n = obj.n;
        ss = ss.setInvariantTau();
      end
      
      [used, Zinx, Zend] = obj.computeUsed(ss);
      augSpec.baseFreqState = used.state;
      
      % Types of accumulators: 0 for averages, 1 for sums
      augSpec.accumulatorTypes = used.Types;
      
      [augSpec.addLagsFrom, LagRowPos, m.withLag] = ...
        obj.determineNeededLags(ss, ...
        augSpec.baseFreqState(~used.Types), used.Horizon(:,~used.Types));
      
      % Dimension calculations
      augSpec.nAccumulatorStates = size(used.Calendar, 2);
      m.final = m.withLag + augSpec.nAccumulatorStates;
      augSpec.m = m;
      
      % State will be added so that the additional states needed for a given observation
      % are groupped together before any additional states needed for other observations
      % are added.
      Zspec.originIndexes = find(Zinx);
      [accumObs, ~] = find(Zinx);
      accumStates = m.withLag + Zend(Zinx);
      Zspec.finalIndexes = sub2ind([size(ss.Z, 1) augSpec.m.final], accumObs, accumStates);
      
      augSpec.Z = Zspec;
      
      % T augmentation:
      Tspec = struct;
      Ttypes = [ss.tau.T used.Calendar used.Horizon];
      [uniqueTs, ~, Tspec.newtau] = unique(Ttypes, 'rows');
      Tspec.oldtau = uniqueTs(:, 1);
      Tspec.cal = uniqueTs(:, 1 + (1:augSpec.nAccumulatorStates));
      Tspec.hor = uniqueTs(:, 1 + augSpec.nAccumulatorStates + (1:augSpec.nAccumulatorStates));
      Tspec.LagRowPos = LagRowPos;
      augSpec.T = Tspec;
      
      % c augmentation
      % The only c's that vary with time are those used in average accumulators. We can
      % drop the calendars for the sum accumulators and just use those with Type == 0.
      cSpec = struct;
      ctypes = [ss.tau.c used.Calendar(:,used.Types == 0)];
      [~, iA_c, cSpec.newtau] = unique(ctypes, 'rows');
      cSpec.cal = used.Calendar(iA_c,:);
      cSpec.oldtau = ss.tau.c(sort(iA_c));
      augSpec.c = cSpec;
      
      % gamma augmentation - same as c
      gammaSpec = struct;
      if ~isempty(ss.gamma)
        gammatypes = [ss.tau.gamma used.Calendar(:,used.Types == 0)];
        [~, iA_gamma, gammaSpec.newtau] = unique(gammatypes, 'rows');
        gammaSpec.cal = used.Calendar(iA_gamma,:);
        gammaSpec.oldtau = ss.tau.gamma(sort(iA_gamma));
      else
        gammaSpec.oldtau = ss.tau.gamma(1);
        gammaSpec.newtau = ss.tau.gamma;
        gammaSpec.cal = ones(length(ss.tau.gamma), size(used.Calendar,2));
      end
      augSpec.gamma = gammaSpec;
      
      % R augmentation
      % The only R's that vary with time are those used in average accumulators. We can
      % drop the calendars for the sum accumulators and just use those with Type == 0.
      Rspec = struct;
      Rtypes = [ss.tau.R used.Calendar(:,used.Types == 0)];
      [~, iA_R, Rspec.newtau] = unique(Rtypes, 'rows');
      Rspec.cal = used.Calendar(iA_R,:);
      Rspec.oldtau = ss.tau.c(sort(iA_R));
      augSpec.R = Rspec;
    end
    
    function [stateAgg, Zinx, Zend] = computeUsed(obj, ss)
      % Compute which of the possible aggregated states we need are being used based on
      % the structure of the given state space model.
      %
      % Determines which states need to be aggregated based on whether a low-frequency
      % observation occurs from that state. The ordering of the aggregated states matters
      % here since it will be used to determine the structure of the Z and T matricies.
      
      % Compute definitions of possible aggregated states
      % An accumulator variable is defined by (original state, type, cal, hor) that says
      % what the low-frequency version of a latent state will be. Get the list of states
      % we need aggregated versions for ordered by the observation they will be used for.
      [lowFreqObsInx, statesToAggregate] = obj.findUsedAccum(ss.Z);
      [lowFreqObsInxOrdered, tempUnsort] = sort(lowFreqObsInx);
      statesToBeAugmentedOrdered = statesToAggregate(tempUnsort);
      
      % Find the different accumulators we need: anything that has the same
      % (state, type, horizon, calendar) doesn't need to be done twice.
      [~, inxOrder] = sort(obj.index);
      sortTypes = obj.accumulatorTypes(inxOrder);
      sortHorizon = obj.horizon(:,inxOrder);
      sortCalendar = obj.calendar(:,inxOrder);
      
      possibleAggregateDefs = [statesToBeAugmentedOrdered'; ...
        sortTypes(lowFreqObsInxOrdered)'; ...
        sortHorizon(:, lowFreqObsInxOrdered); ...
        sortCalendar(:, lowFreqObsInxOrdered)];
      [~, iA_neededDefs, iC_usedDefs] = unique(possibleAggregateDefs', 'rows');
      
      % The definitions were already sorted in the order we want them. Make sure we
      % maintain that order going forward.
      aggregateStateDefs = possibleAggregateDefs(:, sort(iA_neededDefs));
      
      % Pick out structural elements now that we know what we need:
      nPer = size(obj.horizon, 1);
      stateAgg.state = aggregateStateDefs(1, :);
      stateAgg.Types = aggregateStateDefs(2, :);
      stateAgg.Calendar = aggregateStateDefs(nPer + 3:end, :);
      stateAgg.Horizon = aggregateStateDefs(3:nPer + 2, :);
      
      % (2) Find the order that the aggregated states will be used in.
      %
      % Compute where Z elements should be moved from and to (linear indexes)
      % Original Z elements: any nonzero elements of rows of Z for used
      %
      % Logical index of original Z matrix elements that need to be moved.
      ZUsedRows = zeros(ss.p, ss.m);
      accumObs = unique(obj.index(lowFreqObsInx));
      ZUsedRows(accumObs,:) = 1;
      Zinx = ZUsedRows & logical(ss.Z~=0);
      
      % Find where the index of the aggregated states each observation needs. This will be
      % sorted by observation then by state but any state that's used for multiple
      % observations will occur according to the first observation that needs it.
      %
      % To find where these go, start with the list of used aggregate state definitions.
      % Iterate through each one, adding one to the current count as you come across a new
      % definition.
      endState = nan(sum(Zinx(:)), 1);
      count = 0;
      for iE = 1:length(iC_usedDefs)
        if any(iC_usedDefs(iE) == iC_usedDefs(1:iE-1))
          endState(iE) = endState(find(iC_usedDefs(iE) == iC_usedDefs(1:iE-1), 1));
        else
          count = count + 1;
          endState(iE) = count;
        end
      end
      ZendT = zeros(ss.m, ss.p); 
      ZendT(Zinx') = endState;
      Zend = ZendT';
    end
    
    function augSpec = comptueThetaMapAugSpecification(obj, tm)
      % Determine how to augment a ThetaMap.
      
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
      fullgamma = tm.fixed.gamma;
      fullgamma(tm.index.gamma ~= 0) = placeholder;
      fullSys.gamma = fullgamma;
      fullR = tm.fixed.R;
      fullR(tm.index.R ~= 0) = placeholder;
      fullSys.R = fullR;
      
      augSpec = obj.computeAugSpecification(fullSys);
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
      beta.betat = ssLag.beta;
      beta.taubeta = ssLag.tau.beta;
      H.Ht = ssLag.H;
      H.tauH = ssLag.tau.H;
      
      T.Tt = obj.augmentParamT(ssLag.T, aug);
      T.tauT = aug.T.newtau;
      c.ct = obj.augmentParamc(ssLag.c, aug);
      c.tauc = aug.c.newtau;
      gamma.gammat = obj.augmentParamgamma(ssLag.gamma, aug);
      gamma.taugamma = aug.gamma.newtau;
      R.Rt = obj.augmentParamR(ssLag.R, aug);
      R.tauR = aug.R.newtau;
      Q.Qt = ssLag.Q;
      Q.tauQ = ssLag.tau.Q;
      
      % Create new system
      ssNew = StateSpace(Z, H, T, Q, 'd', d, 'beta', beta, 'c', c, 'gamma', gamma, 'R', R);
      
      if ~isempty(ss.a0) || ~isempty(ss.P0)
        [ssNew.a0, ssNew.P0] = obj.augmentParamInit(ssLag.a0, ssLag.P0, ssLag.T, aug);
      end
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
      
      % Delete any "indexes" for lag states we just added
      lagStates = aug.m.original + 1: aug.m.withLag;
      newIndex.T(lagStates, :, :) = 0;
      newIndex.c(lagStates, :, :) = 0;
      newIndex.gamma(lagStates, :, :) = 0;
      newIndex.R(lagStates, :, :) = 0;
      
      % Both accumulators are the same here. We're just copying integers.
      for iAccum = 1:aug.nAccumulatorStates
        iAccumState = aug.m.withLag + iAccum;
        
        % Copy rows of base part of T, c & R to the accumulator states
        newIndex.T(iAccumState, :, :) = newIndex.T(aug.baseFreqState(iAccum), :, :);
        newIndex.c(iAccumState, :) = newIndex.c(aug.baseFreqState(iAccum), :);
        newIndex.gamma(iAccumState, :) = newIndex.gamma(aug.baseFreqState(iAccum), :);
        newIndex.R(iAccumState, :, :) = newIndex.R(aug.baseFreqState(iAccum), :, :);
      end
      
    end
    
    function [transIndex, trans, inv] = augmentTransIndex(obj, tm, aug)
      % Augment transformation index: create a new StateSpace of indexes with
      % the associated transformations to enforce the accumulators.
      %
      % For Z, we're just moving elements so there's no change to the
      % transformations.
      % For T, c & R, the transformation indexes almost propogate the same as
      % the theta indexes.
      % The sum accumulators only copy parameters so we don't need to alter the
      % transformations. The average accumulators require the application of the
      % same linear transformation that is applied during a StateSpace
      % augmentation after the existing transformations are made.
      
      transIndex = obj.augmentIndex(tm.transformationIndex, aug);
      nTrans = max(ThetaMap.vectorizeStateSpace(tm.transformationIndex, ...
        ~tm.usingDefaulta0, ~tm.usingDefaultP0));
      augStates = aug.m.withLag + (1:aug.nAccumulatorStates);
      
      % Modify T transformations
      % The transformations are linear. The full definitions are then given by
      % what elements of T are getting added to them and mutliplied by:
      addendT = Accumulator.augmentParamT(zeros(aug.m.withLag), aug);
      factorT = Accumulator.augmentParamT(ones(aug.m.withLag), aug) - addendT; 
      isAugElemT = false(size(transIndex.T));
      isAugElemT(augStates, 1:aug.m.original, :) = true;
      [transIndex.T, newTransT, newInvT] = Accumulator.computeNewTrans(...
        transIndex.T, factorT, addendT, find(isAugElemT), tm, nTrans); %#ok<FNDSB>
      
      % Modify c transformations (similar structure to T transformations)
      addendc = Accumulator.augmentParamc(zeros(aug.m.withLag, 1), aug);
      factorc = Accumulator.augmentParamc(ones(aug.m.withLag, 1), aug) - addendc;
      isAugElemc = false(size(transIndex.c));
      isAugElemc(augStates, :) = true;
      [transIndex.c, newTransc, newInvc] = Accumulator.computeNewTrans(...
        transIndex.c, factorc, addendc, find(isAugElemc), tm, ...
        nTrans + length(newTransT)); %#ok<FNDSB>
      
      % Modify gamma transformations (similar structure to T transformations)
      addendgamma = Accumulator.augmentParamgamma(zeros(aug.m.withLag, tm.fixed.l, 1), aug);
      factorgamma = Accumulator.augmentParamgamma(ones(aug.m.withLag, tm.fixed.l, 1), aug) - addendgamma;
      isAugElemgamma = false(size(transIndex.gamma));
      isAugElemgamma(augStates, :) = true;
      [transIndex.gamma, newTransgamma, newInvgamma] = Accumulator.computeNewTrans(...
        transIndex.gamma, factorgamma, addendgamma, find(isAugElemgamma), tm, ...
        nTrans + length(newTransT) + length(newTransc)); %#ok<FNDSB>
      
      % Modify R transformations (similar structure to T transformations)
      addendR = Accumulator.augmentParamR(zeros(aug.m.withLag, tm.g), aug);
      factorR = Accumulator.augmentParamR(ones(aug.m.withLag, tm.g), aug) - addendR;
      isAugElemR = false(size(transIndex.R));
      isAugElemR(augStates, :, :) = true;
      [transIndex.R, newTransR, newInvR] = Accumulator.computeNewTrans(...
        transIndex.R, factorR, addendR, find(isAugElemR), tm, ...
        nTrans + length(newTransT) + length(newTransc) + length(newTransgamma)); %#ok<FNDSB>
      
      trans = [tm.transformations newTransT newTransc newTransgamma newTransR];
      inv = [tm.inverses newInvT newInvc newInvgamma newInvR];
    end
    
    %% Helper methods
    function [lowFreqObsIndex, statesToAggregate] = findUsedAccum(obj, ssZ)
      % Find the elements of Z that care about the accumulators - anything
      % that loads onto an observation that has an associated accumulator
      usedZaccumLoc = any(ssZ(sort(obj.index),:,:) ~= 0, 3);
      [lowFreqObsIndex, statesToAggregate] = find(usedZaccumLoc);
      
      % Make sure we return column vectors since find gives row vectors for row
      % vector inputs:
      lowFreqObsIndex = reshape(lowFreqObsIndex, [], 1);
      statesToAggregate = reshape(statesToAggregate, [], 1);
    end
    
    function [lagsColPosOut, lagRowPos, mWLag] = determineNeededLags(obj, ...
        ss, hiFreqStates, augHorizon)
      % Checking we have the correct number of lags of the states that need
      % accumulation to be compatible with Horizon
      %
      % ss - StateSpace being augmented
      % hiFreqStates - The states that will be accumulated to a lower frequency
      %   before they are observed.
      % augHorizon - horizon the high-frequency states will be accumulated by.
      
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
      
      lagsColPosT = lagsColPos';      
      lagsColPosOut = reshape(lagsColPosT(lagsColPosT ~= 0), [], 1);
    end
    
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
      
      assert(ss.m + length(addLagsFrom) == mWithLag, 'Development error.');
      
      if ss.m == mWithLag
        % No need to add lags, don't expand the state
        return
      end
      
      % Z - add zeros to the right
      Z.Zt = zeros(ss.p, mWithLag, size(ss.Z, 3), class(ss.Z));
      Z.Zt(:, 1:ss.m, :) = ss.Z;
      Z.tauZ = ss.tau.Z;
      
      % d, beta and H - do nothing
      d.dt = ss.d;
      d.taud = ss.tau.d;
      beta.betat = ss.beta;
      beta.taubeta = ss.tau.beta;
      H.Ht = ss.H;
      H.tauH = ss.tau.H;
      
      % T - Add ones to transmit lags
      T.Tt = zeros(mWithLag, mWithLag, size(ss.T, 3), class(ss.T));
      T.Tt(1:ss.m, 1:ss.m, :) = ss.T;
      lagsIndex = sub2ind([mWithLag mWithLag], (ss.m+1:mWithLag)', addLagsFrom);
      T.Tt(lagsIndex) = 1;
      T.tauT = ss.tau.T;
      
      % c - just add zeros below
      c.ct = zeros(mWithLag, size(ss.c, 3), class(ss.c));
      c.ct(1:ss.m, :) = ss.c;
      c.tauc = ss.tau.c;
      
      % c - just add zeros below
      gamma.gammat = zeros(mWithLag, size(ss.gamma, 2), size(ss.gamma, 3), class(ss.gamma));
      gamma.gammat(1:ss.m, :) = ss.gamma;
      gamma.taugamma = ss.tau.gamma;
      
      % R - just add zeros below
      R.Rt = zeros(mWithLag, ss.g, size(ss.R, 3), class(ss.R));
      R.Rt(1:ss.m, :, :) = ss.R;
      R.tauR = ss.tau.R;
      
      % Q - do nothing
      Q.Qt = ss.Q;
      Q.tauQ = ss.tau.Q;
      
      % Create new system
      ss = StateSpace(Z, H, T, Q, 'd', d, 'beta', beta, 'c', c, 'R', R, 'gamma', gamma);
      
      if ~isempty(ss.a0)
        error('Unable to add lags of initial state.');
      end
      if ~isempty(ss.P0)
        error('Unable to add lags of initial state.');
      end
    end
    
    function newT = augmentParamT(T, aug)
      % Construct new T matrix
      mNew = aug.m.withLag + aug.nAccumulatorStates;
      
      statesForAvg = unique(aug.baseFreqState(~aug.accumulatorTypes)); 
      
      % Add accumulator elements
      Tslices = size(aug.T.oldtau, 1);
      newT = zeros(mNew, mNew, Tslices, class(T));
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
              % state element's loading on the high-frequency component to account for the
              % lags that need to be added for the average.
              iCols = aug.T.LagRowPos(aug.baseFreqState(iAccum) == statesForAvg, 1:iHor - 1);
              
              newT(iState, iCols, iT) = newT(iState, iCols, iT) + (1/iCal);
            end
          end
        end
      end
    end
    
    function newc = augmentParamc(c, aug)
      % Construct new c vector
      cSlices = size(aug.c.oldtau, 1);
      
      newc = zeros(aug.m.final, cSlices, class(c));
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
    
    function newgamma = augmentParamgamma(gamma, aug)
      % Construct new gamma matrix
      gammaSlices = size(aug.gamma.oldtau, 1);
      
      newgamma = zeros(aug.m.final, size(gamma,2), gammaSlices, class(gamma));
      if isempty(gamma)
        return
      end
      
      for igamma = 1:gammaSlices
        % Get the part of c already defined
        newgamma(1:aug.m.withLag, :, igamma) = gamma(:, :, aug.gamma.oldtau(igamma));
        for iAccum = 1:aug.nAccumulatorStates
          iState = aug.m.withLag + iAccum;
          
          hiFreqElements = gamma(aug.baseFreqState(iAccum), :, aug.gamma.oldtau(igamma));
          
          if aug.accumulatorTypes(iAccum) == 1
            % Sum accumulator
            newgamma(iState, :, igamma) = hiFreqElements;
          else
            % Average accumulator
            newgamma(iState, :, igamma) = (1/aug.gamma.cal(igamma, iAccum)) * hiFreqElements;
          end
        end
      end
    end
    
    function newR = augmentParamR(R, aug)
      % Construct new R matrix
      
      nShocks = size(R, 2);
      
      Rslices = size(aug.R.oldtau, 1);
      newR = zeros(aug.m.final, nShocks, Rslices, class(R));
      for iR = 1:Rslices
        
        % Get the part of R already defined
        newR(1:aug.m.withLag, :, iR) = R(:, :, aug.R.oldtau(iR));
        
        for iAccum = 1:aug.nAccumulatorStates
          iState = aug.m.withLag + iAccum;
          
          hiFreqRElements = R(aug.baseFreqState(iAccum), :, aug.R.oldtau(iR));
          
          if aug.accumulatorTypes(iAccum) == 1
            % Sum accumulator
            newR(iState,:,iR) = hiFreqRElements;
          else
            % Average accumulator
            newR(iState,:,iR) = (1/aug.R.cal(iR,iAccum)) * hiFreqRElements;
          end
        end
      end
    end
    
    function newZ = augmentParamZ(Z, aug)
      % Construct new Z matrix - we're just moving elements around.
      newZ = zeros(size(Z, 1), aug.m.final, class(Z));
      newZ(aug.Z.finalIndexes) = Z(aug.Z.originIndexes);
      Z(aug.Z.originIndexes) = 0;
      newZ(1:size(Z, 1), 1:size(Z, 2), :) = Z;
    end
    
    function [newa0, newP0] = augmentParamInit(a0, P0, T, aug)
      % Compute augmented state initial values.
      %
      % Currently nonfunctional. Set any initial values (a0 and P0) after augmenting the
      % StateSpace or StateSpaceEstimation. 
      %
      % For more details, see the function body.
      
      % Ideally, you want to think about what these values are in expectation given the
      % existing elements of a0. That is like running the smoother with no data
      % where we know the terminal values of the high-frequency states, the accumulating
      % those to get the augmented state values of a0.
      %
      % For series where we observe the sum, we technically can't do anything other than
      % the diffuse initialization because we don't know how long the current sum has been
      % running for. This means that for the two cases:
      %   (a) When the 1st period calendar is 1 a0 is 0 and P0 is Inf for augmented state.
      %   (b) When the 1st period calendar is 0 a0 is equal to the high-frequency a0 with
      %       the same variance as the high-frequency state and a correlation of 1.
      %
      % For series where we observe the average, we can run the expectations backwards and
      % get the partial average for the accumulator. For most cases where the calendar is
      % eqaul to one in the first period, this is just the existng a0 value. But it will
      % always work to get the previous c high-frequency a0 values and take the average of
      % them.
      
      error('Initial values of augmented system not yet solved for.');
      
      maxCal = max(aug.T.cal(aug.T.newtau(1), :));
      
      % Run smoother to get prior a0 values
      % This has the wrong P0 right now. I want V0 = P0
      m = length(a0);
      Z = eye(m);
      H = P0;
      Q = eye(m) * 10 * eps;
      ssT = StateSpace(Z, H, T, Q);
      y = [nan(m, maxCal), a0];
      a0Prev = ssT.smooth(y);
      
      a0Prev = nan(size(a0,1), maxCal);
      a0Prev(:,end) = a0;
      pinvT = pinv(T(1:aug.m.withLag,1:aug.m.withLag,aug.T.newtau(1)));
      for iT = 1:maxCal-1
        a0Prev(:,end-iT) = pinvT * a0Prev(:,end-iT+1);
      end
      
      newa0 = zeros(aug.m.final, 1);
      newa0(1:aug.m.withLag) = a0;
      for iA = 1:aug.nAccumulatorStates
        nPers = [];
        a0mean = mean(a0Prev(:,end-nPers), 2);
        newa0(aug.m.withLag+iA) = a0mean(aug.baseFreqState(iA));
      end
      
    end
    
    %% ThetaMap augmentation methods
    function [newParamMat, newTrans, newInv] = ...
        computeNewTrans(augParamMat, factor, addend, indexElems, tm, nTrans)
      % Compose new functions that take into account the linear transformation
      % occuring in the augmentation.
      
      newTrans = cell(1, numel(indexElems));
      newInv = cell(1, numel(indexElems));
      
      newParamMat = augParamMat;
      for iTrans = 1:numel(indexElems)
        iElem = indexElems(iTrans);
        iTransInd = augParamMat(iElem);
        
        if iTransInd == 0
          % Not a function of theta
          continue
        end
        if factor(iElem) == 1 && addend(iElem) == 0
          % Simply copy of elements, don't nest the function
          continue
        end
        
        [newTrans{iTrans}, newInv{iTrans}] = ...
          Accumulator.createTransforms(tm.transformations{iTransInd}, ...
          tm.inverses{iTransInd}, factor(iElem), addend(iElem));
        
        % Move transformationIndex of the element we just changed
        newParamMat(iElem) = nTrans + 1;
        nTrans = nTrans + 1;
      end
      
      % Concentrate out unused new transformations
      newTrans = newTrans(~cellfun(@isempty, newTrans));
      newInv = newInv(~cellfun(@isempty, newInv));
    end
    
    function [newT, newI] = createTransforms(oldT, oldI, A, B)
      % Create new transformations and inverses from the linear
      % scaling factor A and addend B.
      
      % Compose transformation by multiplying then adding
      newT = Accumulator.composeLinearFunc(oldT, A, B);
      
      % "Undo" the linear transformation with its inverse
      newI = @(y) oldI((y - B) ./ A);
    end
    
    function newFunc = composeLinearFunc(func, A, B)
      % A helper function for performing a linear transformation to the result
      % from an arbitrary function handle. Provides g(x) = A * f(x) + B
      newFunc = @(x) A * func(x) + B;
    end
  end
end