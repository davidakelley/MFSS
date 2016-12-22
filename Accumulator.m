classdef Accumulator < AbstractSystem
  % State space augmenting accumulators, enforcing sum and average aggregation
  %
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
    
    % Properties used when augmenting a StateSpace
    tempSet
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
      % Note that there will be no initial values set once augmented. 
      
      obj.checkConformingSystem(ss);
      
      % Set tau if it hasn't been set yet
      if isempty(ss.tau)
        ss.n = obj.n;
        ss = ss.setInvariantTau();
      end      
      obj = obj.setAugmenting(ss);

      % Set up structure of new system
      ssLag = obj.addLags(ss);
      
      Z.Zt = obj.augmentAccumZ(ssLag);
      Z.tauZ = ssLag.tau.Z;
      d.dt = ssLag.d;
      d.taud = ssLag.tau.d;
      H.Ht = ssLag.H;
      H.tauH = ssLag.tau.H;
      
      [T.Tt, T.tauT] = obj.augmentAccumT(ssLag);
      [c.ct, c.tauc] = obj.augmentAccumc(ssLag);
      [R.Rt, R.tauR] = obj.augmentAccumR(ssLag);
      Q.Qt = ssLag.Q;
      Q.tauQ = ssLag.tau.Q;
      
      % Create new system
      ssNew = StateSpace(Z, d, H, T, c, R, Q);      
    end
    
    function tmNew = augmentThetaMap(obj, tm)
      % Create a ThetaMap that produces StateSpaces that obey the accumulator.
      % The size of the theta vector shouldn't change when the state is
      % augmented. 
      
      % Nan values should propogate correctly through augmentStateSpace
      fixedNew = obj.augmentStateSpace(tm.fixed);
      
      % Augment index: For Z we need to move the indexes to the accumulator 
      % states and replace them with zeros. For T, c and R we need to copy the 
      % high frequency row to the accumulator state.
      indexNew = fixedNew.setAllParameters(0);
      
      % Move the existing indexes to the same places in the potentially larger 
      % system. 
      systemParams = indexNew.systemParam;
      for iP = 1:length(systemParams)
        indexNew.(systemParams{iP})(1:iN, 1:iM) = tm.index.(systemParams{ip})(1:iN, 1:iM);
      end
      
      % Move Z elements
      indexNew.Z = obj.augmentAccumZ(indexNew.Z);
      
      % Copy rows of T, c and R to the accumulator states
      accumStateParam = {'T', 'c', 'R'};
      for iP = 1:length(accumStateParam)
        
      end
      
      % The transformation indexes should be constructed somewhat similarly to
      % the regular transformation indexes. They'll be out of order from what's
      % normal but it should otherwise just be working through 
      transformIndexNew = [];
      
      augTrans = [];
      augDerivs = [];
      augInv = [];
      transNew = [tm.transformations augTrans];
      derivNew = [tm.derivatives augDerivs];
      invNew = [tm.inverses augInv];
            
      tmNew = ThetaMap(fixedNew, indexNew, transNew, derivNew, invNew, ...
        transformIndexNew);
    end
    
    function sseNew = augmentStateSpaceEstimation(obj, sse)
      % Augment a StateSpaceEstimation to obey the accumulators. 
      % This is actually simple: augment the system matricies and let the nans
      % propogate as they should and then augment the ThetaMap. 
      
      sseNew = obj.augmentStateSpace(sse);
      sseNew.ThetaMapping = obj.augmentThetaMap(sse);      
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
      
      cal = nan(nPer+1, nSeries);
      hor = nan(nPer+1, nSeries);
      
      for iSer = unique(seriesMissing)'
        maxCal = unique(diff(obsMissing(seriesMissing == iSer)));
        assert(isscalar(maxCal));
        if maxCal == 1
          continue
        end
        if strcmpi(types{iSer}, 'avg')
          tempCal = repmat((1:maxCal)', [ceil(nPer/maxCal)+1 1]);
        else
          tempCal = repmat([zeros(maxCal-1, 1); 1], [ceil(nPer/maxCal)+1 1]);
        end
        cal(:, iSer) = tempCal(1:nPer+1);
        hor(:, iSer) = horizons(iSer);
      end
      inx = all(~isnan(cal));
      
      accum = Accumulator(inx, cal(:,inx), hor(:,inx));
    end
  end
  
  methods (Hidden)
    %% Parameter augmentation methods
    function ss = addLags(obj, ss)
      % Add the needed lags to the system matricies.
      [mNew, ColAddLags] = obj.accumAugmentedStateDims(ss);
      
      if ss.m == mNew
        % No need to add lags, don't expand the state
        return
      end
      
      % Z - add zeros to the right
      Z.Zt = zeros(ss.p, mNew, size(ss.Z, 3));
      Z.Zt(:, 1:ss.m, :) = ss.Z;
      Z.tauZ = ss.tau.Z;
      
      % d and H - do nothing 
      d.dt = ss.d;
      d.taud = ss.tau.d;
      H.Ht = ss.H;
      H.tauH = ss.tau.H;
      
      % T - Add ones to transmit lags
      T.Tt = zeros(mNew, mNew, size(ss.T, 3));
      T.Tt(1:ss.m, 1:ss.m, :) = ss.T;
      lagIndexes = sub2ind([mNew, mNew], (ss.m+1:mNew)', ColAddLags);
      T.Tt(lagIndexes) = 1;
      T.tauT = ss.tau.T;
      
      % c - just add zeros below
      c.ct = zeros(mNew, size(ss.c, 3));
      c.ct(1:ss.m, :) = ss.c;
      c.tauc = ss.tau.c;
      
      % R - just add zeros below
      R.Rt = zeros(mNew, ss.g, size(ss.R, 3));
      R.Rt(1:ss.m, :, :) = ss.R;
      R.tauR = ss.tau.R;
      
      % Q - do nothing
      Q.Qt = ss.Q;
      Q.tauQ = ss.tau.Q;
      
      % Create new system
      ss = StateSpace(Z, d, H, T, c, R, Q);      
    end
    
    function [newT, newtauT] = augmentAccumT(obj, ss)
      % Construct new T matrix
      aug = obj.tempSet;
      
      [~, ~, LagRowPos] = obj.accumAugmentedStateDims(ss);
      
      nAccumStates = size(aug.calendar, 2);
      mNew = ss.m + nAccumStates;

      accumStates = obj.findUsedAccum(ss);
      states = unique(accumStates);
      
      % Add accumulator elements
      Ttypes   = [ss.tau.T aug.calendar aug.horizon];
      [uniqueTs, ~, newtauT] = unique(Ttypes, 'rows');
      oldTs.tau = uniqueTs(:, 1);
      oldTs.cal = uniqueTs(:, 1 + (1:nAccumStates));
      oldTs.hor = uniqueTs(:, 1 + nAccumStates + (1:nAccumStates));
      Tslices = size(uniqueTs, 1);

      newT = zeros(mNew, mNew, Tslices);
      for iT = 1:Tslices
        % Get the part of T already defined
        newT(1:ss.m, 1:ss.m, iT) = ss.T(:, :, oldTs.tau(iT));
        
        % Define the new transition equations for the accumulator states
        for iAccum = 1:nAccumStates
          iState = ss.m + iAccum;
          iCal = oldTs.cal(iT, iAccum);
          iHor = oldTs.hor(iT, iAccum);
          
          hiFreqTelements = ss.T(aug.states(iAccum), :, oldTs.tau(iT));
          
          if aug.accumulatorTypes(iAccum) == 1
            % Sum accumulator
            newT(iState, 1:ss.m, iT) = hiFreqTelements;
            newT(iState, iState, iT) = iCal;
            
          else
            % Average accumulator
            newT(iState, 1:ss.m, iT) = (1 / iCal) * hiFreqTelements;
            newT(iState, iState, iT) = (iCal - 1) / iCal;
            
            if iHor > 1
              % Triangle accumulator
              % What is LagRowPos?
              % This should really be a call to ss.LagsInState here instead of
              % carrying LagRowPos around.
              cols = LagRowPos(aug.states(iAccum) == states, 1:iHor - 1);
              
              [~, lagPositions] = ss.LagsInState(aug.states(iAccum));
              tempLagRowPos = [aug.states(iAccum); lagPositions];
              cols_dk2 = tempLagRowPos(1:iHor-1);
              assert(all(cols == cols_dk2'));
              
              newT(iState, cols_dk2, iT) = newT(iState, cols_dk2, iT) + (1/iCal);
            end
          end
        end
      end
    end
    
    function [newc, newtauc] = augmentAccumc(obj, ss)
      % Construct new c vector
      aug = obj.tempSet;
      nAccumStates = size(aug.calendar, 2);

      % Add accumulator elements
      ctypes   = [ss.tau.c aug.calendar];
      [uniquecs, ~, newtauc] = unique(ctypes, 'rows');
      oldcs.tau = uniquecs(:, 1);
      oldcs.cal = uniquecs(:, 1 + (1:nAccumStates));
      cSlices = size(uniquecs, 1);
      
      nAccumStates = size(aug.calendar, 2);
      mNew = ss.m + nAccumStates;
      
      newc = zeros(mNew, cSlices);
      for jj = 1:cSlices
        % Get the part of c already defined
        newc(1:ss.m, jj) = ss.c(:, oldcs.tau(jj));
        for h = 1:nAccumStates
          hiFreqCelements = ss.c(aug.states(h), oldcs.tau(jj));
          
          if aug.accumulatorTypes(h) == 1
            % Sum accumulator
            newc(ss.m+h, jj) = hiFreqCelements;
          else
            % Average accumulator
            newc(ss.m+h, jj) = (1/oldcs.cal(jj,h)) * hiFreqCelements;
          end
        end
      end
    end
    
    function [newR, newtauR] = augmentAccumR(obj, ss)
      % Construct new R matrix
      aug = obj.tempSet;
      nAccumStates = size(aug.calendar, 2);

      % Add accumulator elements
      Rtypes   = [ss.tau.R aug.calendar];
      [uniqueRs, ~, newtauR] = unique(Rtypes, 'rows');
      oldRs.tau = uniqueRs(:, 1);
      oldRs.cal = uniqueRs(:, 1 + (1:nAccumStates));
      
      nAccumStates = size(aug.calendar, 2);
      mNew = ss.m + nAccumStates;
      
      Rslices = size(uniqueRs, 1);
      newR = zeros(mNew, ss.g, Rslices);
      for jj = 1:Rslices
        % Get the part of R already defined
        newR(1:ss.m, 1:ss.g, jj) = ss.R(:, :, oldRs.tau(jj));
        
        for h = 1:nAccumStates
          hiFreqRElements = ss.R(aug.states(h), :, oldRs.tau(jj));
          
          if aug.accumulatorTypes(h) == 1
            % Sum accumulator
            newR(ss.m+h,:,jj) = hiFreqRElements;
          else
            % Average accumulator
            newR(ss.m+h,:,jj) = (1/oldRs.cal(jj,h)) * hiFreqRElements;
          end
        end
      end
    end
    
    function newZ = augmentAccumZ(obj, ss)
      % Construct new Z matrix
      aug = obj.tempSet;
      [accumStates, accumObsDims] = obj.findUsedAccum(ss);

      nonZeroZpos = zeros(length(obj.index), ss.m);
      nonZeroZpos(sub2ind([length(obj.index) ss.m], accumObsDims, accumStates)) = aug.usedAccumulators;
      
      % Transfer loadings from where they are to the accumulator we just added
      nSlices = size(ss.Z, 3);
      newZ = ss.Z;
      newZ(obj.index, :, :) = zeros(length(obj.index), ss.m, nSlices);
      for jSlice = 1:nSlices
        for iAccum = 1:length(obj.index)
          % 
          cols = find(ss.Z(obj.index(iAccum), :, jSlice));
          newZ(obj.index(iAccum), ss.m + nonZeroZpos(iAccum,cols), jSlice) = ...
            ss.Z(obj.index(iAccum), cols, jSlice);
        end
      end
    end
    
    %% Accumulator helper methods
    function [accumStates, accumObsDims] = findUsedAccum(obj, ss)
      % Find the elements of Z that care about the accumulators - anything
      % that loads onto an observation that has an associated accumulator
      usedZaccumLoc = any(ss.Z(obj.index,:,:) ~= 0, 3);
      [accumObsDims, accumStates] = find(usedZaccumLoc);  
    end
    
    function obj = setAugmenting(obj, ss)
      %
      [accumStates, accumObsDims] = obj.findUsedAccum(ss);
      
      % Find the different accumulators we need: anything that has the same 
      % (state, type, horizon, calendar) doesn't need to be done twice. 
      possibleAccumDefs = [accumStates'; ...
                           obj.accumulatorTypes(accumObsDims)'; ...
                           obj.horizon(:, accumObsDims); ...
                           obj.calendar(:, accumObsDims)];
      [APsi, ~, obj.tempSet.usedAccumulators] = unique(possibleAccumDefs', 'rows');
      
      obj.tempSet.states = APsi(:, 1);
      obj.tempSet.accumulatorTypes = APsi(:, 2);
      obj.tempSet.horizon = APsi(:, 3:size(obj.horizon, 1) + 2)';
      obj.tempSet.calendar = APsi(:, size(obj.horizon, 1) + 3:end)';
    end
    
    function [mWLag, LagColPos, LagRowPos] = accumAugmentedStateDims(obj, ss)
      % Checking we have the correct number of lags of the states that need
      % accumulation to be compatible with Horizon
      
      aug = obj.tempSet; % TODO: Make this unncessary
      [accumStates, accumObsDims] = obj.findUsedAccum(ss);
      
      % How many lags we need to add by each state we're concerned with
      usedStates = unique(accumStates);
      AddLagsByState = zeros(length(usedStates), 1);

      maxHor = max(aug.horizon, [], 1);
      % Index vectors for new lags we're going to add. The first element of 
      % these will be the state we're making lags of, lags come afterwards
      LagColPos  = zeros(length(usedStates), max(maxHor));
      LagRowPos  = zeros(length(usedStates), max(maxHor));
      
      mWLag = ss.m;
      for iSt = 1:length(usedStates)
        % We need to find all of the previous lags of a state so that we know
        % which lag is the last one we have so far. Then we can start with that
        % state and add additional lags until we have the number we need. 
        [nLags, lagPos] = ss.LagsInState(usedStates(iSt));
        
        % Add enough lags for this state to get to the max horizon of this
        % accumulator
        mHor = max(maxHor(aug.usedAccumulators(accumObsDims(usedStates(iSt) == accumStates))));
        % 
        AddLagsByState(iSt) = max(mHor-1-nLags-1, 0);
        
        % Locate existing lags states...
        LagRowPos(iSt, 1:1 + nLags) = [usedStates(iSt) lagPos'];
        % ... and add as many as we need past the existing size of the state
        extraLagsInds = 1:AddLagsByState(iSt);
        LagRowPos(iSt, 1 + nLags + extraLagsInds) = mWLag + extraLagsInds;
        
        if AddLagsByState(iSt) > 0
          % Find the last lag we have already
          LagColPos(iSt, 1 + nLags + 1) = lagPos(end);
        end
        if AddLagsByState(iSt) > 1
          % Stick extra lags on the end of the current state
          tempInx = 1 + nLags + 1 + (1:AddLagsByState(iSt)-1);
          LagColPos(iSt, tempInx) = mWLag + extraLagsInds(1:end-1);
        end
        
        % We need to expand the state so the lags we just identified will fit
        mWLag = mWLag + AddLagsByState(iSt);
      end
      
      LagColPos = reshape(LagColPos(LagColPos ~= 0), [], 1);
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
end