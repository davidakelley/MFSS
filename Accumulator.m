classdef Accumulator < AbstractSystem
  % State space augmenting accumulators, enforcing sum and average aggregation
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
    augment
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
    
    function ss = augmentStateSpace(obj, ss)
      % Augment the state of a system to enforce accumulator constraints
      
      obj.checkConformingSystem(ss);
      
      % Set tau if it hasn't been set yet
      if isempty(ss.tau)
        ss.n = obj.n;
        ss = ss.setInvariantTau();
      end      
      obj = obj.setAugmenting(ss);

      % Set up structure of new system
      [Zlag, Tlag, clag, Rlag] = obj.addLags(ss);
      
      Z.Zt = obj.augmentAccumZ(ss, Zlag);
      Z.tauZ = ss.tau.Z;
      d.dt = ss.d;
      d.taud = ss.tau.d;
      H.Ht = ss.H;
      H.tauH = ss.tau.H;
      
      [T.Tt, T.tauT] = obj.augmentAccumT(ss, Tlag);
      [c.ct, c.tauc] = obj.augmentAccumc(ss, clag);
      [R.Rt, R.tauR] = obj.augmentAccumR(ss, Rlag);
      Q.Qt = ss.Q;
      Q.tauQ = ss.tau.Q;
      
      % Create new system
      ss = StateSpace(Z, d, H, T, c, R, Q);      
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
      
      cal = nan(nPer, nSeries);
      hor = nan(nPer, nSeries);
      
      for iSer = unique(seriesMissing)'
        maxCal = unique(diff(obsMissing(seriesMissing == iSer)));
        assert(isscalar(maxCal));
        if maxCal == 1
          continue
        end
        if strcmpi(types{iSer}, 'avg')
          tempCal = repmat((1:maxCal)', [ceil(nPer/maxCal) 1]);
        else
          tempCal = repmat([zeros(maxCal-1, 1); 1], [ceil(nPer/maxCal) 1]);
        end
        cal(:, iSer) = tempCal(1:nPer);
        hor(:, iSer) = horizons(iSer);
      end
      inx = all(~isnan(cal));
      
      accum = Accumulator(inx, cal(:,inx), hor(:,inx));
    end
  end
  
  methods (Hidden)
    %% Parameter augmentation methods
    function [Z, T, c, R] = addLags(obj, ss)
      % Add the needed lags to the system matricies.
      aug = obj.augment;
      
      if ss.m == aug.mNew
        % Not adding any lags
        Z = ss.Z;
        T = ss.T;
        c = ss.c;
        R = ss.R;
      else
        % Z - add zeros to the right        
        Z = zeros(ss.p, aug.mNew, size(ss.Z, 3));
        Z(:, 1:ss.m, :) = ss.Z;
        
        % T - Add ones to transmit lags
        T = zeros(aug.mNew, aug.mNew, size(ss.T, 3));
        T(1:ss.m, 1:ss.m, :) = ss.T;        
        lagIndexes = sub2ind(size(T), (ss.m+1:aug.mNew)', aug.ColAdd);
        T(lagIndexes) = 1;
        
        % c - just add zeros below        
        c = zeros(aug.mNew, size(ss.c, 3));
        c(1:ss.m, :) = ss.c;
        
        % R - just add zeros below
        R = zeros(aug.mNew, ss.g, size(ss.R, 3));
        R(1:ss.m, :, :) = ss.R;
      end
    end
    
    function [newT, newtauT] = augmentAccumT(obj, ss, T)
      % Construct new T matrix
      aug = obj.augment;
      
      nAccumStates = size(aug.calendar, 2);
      mNew = ss.m + nAccumStates;

      states = unique(aug.accumStates);
      
      % Add accumulator elements
      Ttypes   = [ss.tau.T' aug.calendar aug.horizon];
      [uniqueTs, ~, newtauT] = unique(Ttypes, 'rows');
      newTs.tau = uniqueTs(:,1);
      newTs.cal = uniqueTs(:,2:2);
      newTs.hor = uniqueTs(:,3:3);
      Tslices = size(uniqueTs, 1);

      newT = zeros(mNew, mNew, Tslices);
      for iT = 1:Tslices
        % Get the part of T already defined
        newT(1:ss.m, 1:ss.m, iT) = T(1:ss.m, 1:ss.m, newTs.tau(iT));
        
        % Define the new transition equations for the accumulator states
        for iAccum = 1:nAccumStates
          iState = ss.m + iAccum;
          if aug.accumulatorTypes(iAccum) == 1
            % Sum accumulator
            newT(iState, [1:ss.m iState], iT) = ...
              [T(aug.states(iAccum), :, newTs.tau(iT)) uniqueTs(iT, 1+iAccum)];
          else
            % Average accumulator
            newT(iState, [1:ss.m iState], iT) = ...
              [(1/uniqueTs(iT, 1+iAccum)) * T(aug.states(iAccum), :, newTs.tau(iT))...
              (uniqueTs(iT, 1+iAccum)-1) / uniqueTs(iT, 1+iAccum)];
            if uniqueTs(iT, nAccumStates+1+iAccum) > 1
              cols = aug.RowPos(aug.states(iAccum) == states, 1:uniqueTs(iT, nAccumStates+1+iAccum)-1);
              
              newT(iState, cols, iT) = newT(iState,cols,iT) + (1/uniqueTs(iT,1+iAccum));
            end
          end
          
        end
      end
    end
    
    function [newc, newtauc] = augmentAccumc(obj, ss, c)
      % Construct new c vector
      aug = obj.augment;
      
      % Add accumulator elements
      ctypes   = [ss.tau.c' aug.calendar];
      [uniquecs, ~, newtauc] = unique(ctypes,'rows');
      Numc = size(uniquecs, 1);
      
      nAccumStates = size(aug.calendar, 2);
      mNew = ss.m + nAccumStates;
      
      newc = zeros(mNew, Numc);
      for jj = 1:Numc
        newc(1:ss.m, jj) = c(:, uniquecs(jj,1));
        for h = 1:nAccumStates
          if aug.accumulatorTypes(h) == 1
            newc(ss.m+h, jj) = c(aug.states(h), uniquecs(jj,1));
          else
            newc(ss.m+h, jj) = (1/uniquecs(jj,1+h)) * c(aug.states(h), uniquecs(jj,1));
          end
        end
      end
    end
    
    function [newR, newtauR] = augmentAccumR(obj, ss, R)
      % Construct new R matrix
      aug = obj.augment;

      % Add accumulator elements
      Rtypes   = [ss.tau.R' aug.calendar];
      [uniqueRs, ~, newtauR] = unique(Rtypes,'rows');
      
      nAccumStates = size(aug.calendar, 2);
      mNew = ss.m + nAccumStates;
      
      NumR = size(uniqueRs, 1);
      newR = zeros(mNew, ss.g, NumR);
      for jj = 1:NumR
        newR(1:ss.m, 1:ss.g, jj) = R(:, :, uniqueRs(jj,1));
        for h = 1:nAccumStates
          if aug.accumulatorTypes(h) == 1
            newR(ss.m+h,:,jj) = R(aug.accumulatorTypes(h), :, uniqueRs(jj,1));
          else
            newR(ss.m+h,:,jj) = (1/uniqueRs(jj, 1+h)) * ...
              R(aug.calendar(h,1), :, uniqueRs(jj,1));
          end
        end
      end
    end
    
    function newZ = augmentAccumZ(obj, ss, Z)
      % Construct new Z matrix
      aug = obj.augment;

      nonZeroZpos = zeros(length(obj.index), aug.mNew);
      nonZeroZpos(sub2ind([length(obj.index) aug.mNew], aug.accumObsDims, aug.accumStates)) = aug.usedAccumulators;
      
      newZ = zeros(obj.p, aug.mNew, size(Z, 3));
      for jj = 1:size(Z, 3)
        newZ(:,:,jj) = Z(:,:,jj);
        newZ(obj.index,:,jj) = zeros(length(obj.index), aug.mNew);
        for h = 1:length(obj.index)
          cols = find(Z(obj.index(h),:,jj));
          newZ(obj.index(h), ss.m + nonZeroZpos(h,cols), jj) = Z(obj.index(h),cols,jj);
        end
      end
    end
    
    %% Accumulator helper methods
    function obj = setAugmenting(obj, ss)
      %
      
      % Find the elements of Z that care about the accumulators - anything
      % that loads onto an observation that has an associated accumulator
      usedZaccumLoc = any(ss.Z(obj.index,:,:) ~= 0, 3);
      [obj.augment.accumObsDims, obj.augment.accumStates] = find(usedZaccumLoc);
      
      % Find the different accumulators we need: anything that has the same 
      % (state, type, horizon, calendar) doesn't need to be done twice. 
      possibleAccumDefs = [obj.augment.accumStates'; ...
        obj.accumulatorTypes(obj.augment.accumObsDims)'; ...
        obj.horizon(:, obj.augment.accumObsDims); ...
        obj.calendar(:, obj.augment.accumObsDims)];
      [APsi, ~, obj.augment.usedAccumulators] = unique(possibleAccumDefs', 'rows');
      
      obj.augment.states = APsi(:, 1);
      obj.augment.accumulatorTypes = APsi(:, 2);
      obj.augment.horizon = APsi(:, 3:size(obj.horizon, 1) + 2)';
      obj.augment.calendar = APsi(:, size(obj.horizon, 1) + 3:end)';
      
      [mNew, ColAdd, RowPos] = obj.accumAugmentedStateDims(ss);
      obj.augment.mNew = mNew;
      obj.augment.ColAdd = ColAdd;
      obj.augment.RowPos = RowPos;
    end
    
    function [mNew, LagColPos, LagRowPos] = accumAugmentedStateDims(obj, ss)
      % Checking we have the correct number of lags of the states that need
      % accumulation to be compatible with Horizon
      aug = obj.augment;
      
      % How many lags we need to add by each state we're concerned with
      states = unique(aug.accumStates);
      AddLagsByState = zeros(length(states), 1);

      maxHor = max(aug.horizon, [], 2);
      % Index vectors for new lags we're going to add. The first element of 
      % these will be the state we're making lags of, lags come afterwards
      LagColPos  = zeros(length(states), max(maxHor));
      LagRowPos  = zeros(length(states), max(maxHor));
      
      mNew = ss.m;
      for iSt = 1:length(states)
        % We need to find all of the previous lags of a state so that we know
        % which lag is the last one we have so far. Then we can start with that
        % state and add additional lags until we have the number we need. 
        [nLags, lagPos] = ss.LagsInState(states(iSt));
        
        % Add enough lags for this state to get to the max horizon of this
        % accumulator
        mHor = max(maxHor(aug.accumObsDims(states(iSt) == aug.accumStates)));
        AddLagsByState(iSt) = max(mHor-1-nLags-1, 0);
        
        % Locate existing lags states...
        LagRowPos(iSt, 1:1 + nLags) = [states(iSt) lagPos'];
        % ... and add as many as we need past the existing size of the state
        extraLagsInds = 1:AddLagsByState(iSt);
        LagRowPos(iSt, 1 + nLags + extraLagsInds) = mNew + extraLagsInds;
        
        if AddLagsByState(iSt) > 0
          % Find the last lag we have already
          LagColPos(iSt, 1 + nLags + 1) = lagPos(end);
        end
        if AddLagsByState(iSt) > 1
          % Stick extra lags on the end of the current state
          tempInx = 1 + nLags + 1 + (1:AddLagsByState(iSt)-1);
          LagColPos(iSt, tempInx) = mNew + extraLagsInds(1:end-1);
        end
        
        % We need to expand the state so the lags we just identified will fit
        mNew = mNew+AddLagsByState(iSt);
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