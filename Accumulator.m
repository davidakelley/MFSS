classdef Accumulator < AbstractSystem
  % State space augmenting accumulators, enforcing sum and average aggregation
  %
  
  % David Kelley, 2016
  %
  % TODO (12/14/16)
  % ---------------
  %   - Generate funcion handles for ThetaMap
  %   - Create utiltiy methods for standard accumulator creation (descriptive
  %     specification as opposed to explicitly stating calendar/horizon values)
  
  properties
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
    augmentingIndex
    augmentingCalendar
    augmentingHorizon
    augmentingStates
    augmentingAccumulatorTypes
    
    usedAccumulators
    
    accumObsDims 
    accumStates
  end
  
  methods
    function obj = Accumulator(index, calendar, horizon)
      % Constructor
      if islogical(index)
        index = find(index);
      end
      assert(length(index) == size(calendar, 1));
      assert(length(index) == size(horizon, 1));
      
      obj.index = index;
      obj.calendar = calendar;
      obj.horizon = horizon;
      
      obj.accumulatorTypes = any(obj.calendar == 0, 2);
      
      obj.m = [];
      obj.p = max(index);
      obj.g = [];
      obj.n = size(obj.calendar, 2) - 1;
      obj.timeInvariant = false;
    end
    
    function ss = augmentStateSpace(obj, ss)
      % Augment the state of a system to enforce accumulator constraints
      
      obj = obj.setAugmenting(ss);

      % Set tau
      if isempty(ss.tau)
        ss.n = obj.n;
        ss = ss.setInvariantTau();
      end
      
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
  
  methods (Hidden)
    %% Parameter augmentation methods
    function [Z, T, c, R] = addLags(obj, ss)
      % Add the needed lags to the system matricies.
      
      [mNew, ColAdd] = obj.accumAugmentedStateDims(ss);
      
      if ss.m == mNew
        % Not adding any lags
        Z = ss.Z;
        T = ss.T;
        c = ss.c;
        R = ss.R;
      else
        % Z - add zeros to the right        
        Z = zeros(ss.p, mNew, size(ss.Z, 3));
        Z(:, 1:ss.m, :) = ss.Z;
        
        % T - Add ones to transmit lags
        T = zeros(mNew, mNew, size(ss.T, 3));
        T(1:ss.m, 1:ss.m, :) = ss.T;
        lagInds = sub2ind(size(T), (ss.m+1:mNew)', reshape(ColAdd(ColAdd>0),[],1));
        T(lagInds) = 1;
        
        % c - just add zeros below        
        c = zeros(mNew, size(ss.c, 3));
        c(1:ss.m, :) = ss.c;
        
        % R - just add zeros below
        R = zeros(mNew, ss.g, size(ss.R, 3));
        R(1:ss.m, :, :) = ss.R;
      end
    end
    
    function [newT, newtauT] = augmentAccumT(obj, ss, T)
      % Construct new T matrix
      nAccumStates = size(obj.augmentingCalendar, 1);
      mNew = ss.m + nAccumStates;
%       mNew2 = obj.accumAugmentedStateDims(ss);
%       assert(mNew == mNew2);
      [~, ~, RowPos] = obj.accumAugmentedStateDims(ss);

      states = unique(obj.accumStates);
      
      % Add accumulator elements
      Ttypes   = [ss.tau.T' obj.augmentingCalendar' obj.augmentingHorizon'];
      [uniqueTs, ~, newtauT] = unique(Ttypes, 'rows');
      Tslices = size(uniqueTs, 1);

      newT = zeros(mNew, mNew, Tslices);
      for jj = 1:Tslices
        newT(1:ss.m, :, jj) = [T(:,:,uniqueTs(jj,1)) zeros(ss.m, nAccumStates)];
        for h = 1:nAccumStates
          
          if obj.augmentingAccumulatorTypes(h) == 1
            newT(ss.m + h, [1:ss.m ss.m + h], jj) = ...
              [T(obj.augmentingStates(h), :, uniqueTs(jj, 1)) uniqueTs(jj, 1+h)];
          else
            newT(ss.m+h, [1:ss.m ss.m+h],jj) =...
              [(1/uniqueTs(jj, 1+h)) * T(obj.augmentingStates(h), :, uniqueTs(jj, 1))...
              (uniqueTs(jj, 1+h)-1) / uniqueTs(jj, 1+h)];
            if uniqueTs(jj, nAccumStates+1+h) > 1
              cols = RowPos(obj.augmentingStates(h)==states, 1:uniqueTs(jj, nAccumStates+1+h)-1);
              newT(ss.m+h, cols, jj) = newT(ss.m+h,cols,jj) + (1/uniqueTs(jj,1+h));
            end
          end
          
        end
      end
    end
    
    function [newc, newtauc] = augmentAccumc(obj, ss, c)
      % Construct new c vector
      
      % Add accumulator elements
      ctypes   = [ss.tau.c' obj.augmentingCalendar'];
      [uniquecs, ~, newtauc] = unique(ctypes,'rows');
      Numc = size(uniquecs, 1);
      
      nAccumStates = size(obj.augmentingCalendar, 1);
      mNew = ss.m + nAccumStates;
      
      newc = zeros(mNew, Numc);
      for jj = 1:Numc
        newc(1:ss.m, jj) = c(:, uniquecs(jj,1));
        for h = 1:nAccumStates
          if obj.augmentingAccumulatorTypes(h) == 1
            newc(ss.m+h, jj) = c(obj.augmentingStates(h), uniquecs(jj,1));
          else
            newc(ss.m+h, jj) = (1/uniquecs(jj,1+h)) * c(obj.augmentingStates(h), uniquecs(jj,1));
          end
        end
      end
    end
    
    function [newR, newtauR] = augmentAccumR(obj, ss, R)
      % Construct new R matrix
      
      % Add accumulator elements
      Rtypes   = [ss.tau.R' obj.augmentingCalendar'];
      [uniqueRs, ~, newtauR] = unique(Rtypes,'rows');
      
      nAccumStates = size(obj.augmentingCalendar, 1);
      mNew = ss.m + nAccumStates;
      
      NumR = size(uniqueRs, 1);
      newR = zeros(mNew, ss.g, NumR);
      for jj = 1:NumR
        newR(1:ss.m, 1:ss.g, jj) = R(:, :, uniqueRs(jj,1));
        for h = 1:nAccumStates
          if obj.augmentingAccumulatorTypes(h) == 1
            newR(ss.m+h,:,jj) = R(obj.augmentingAccumulatorTypes(h), :, uniqueRs(jj,1));
          else
            newR(ss.m+h,:,jj) = (1/uniqueRs(jj, 1+h)) * ...
              R(obj.augmentingCalendar(h,1), :, uniqueRs(jj,1));
          end
        end
      end
    end
    
    function newZ = augmentAccumZ(obj, ss, Z)
      % Construct new Z matrix
      mNew = obj.accumAugmentedStateDims(ss);

      nonZeroZpos = zeros(length(obj.index), mNew);
      nonZeroZpos(sub2ind([length(obj.index) mNew], obj.accumObsDims, obj.accumStates)) = obj.usedAccumulators;
      
      newZ = zeros(obj.p, mNew, size(Z, 3));
      for jj = 1:size(Z, 3)
        newZ(:,1:ss.m,jj) = Z(:,:,jj);
        newZ(obj.index,:,jj) = zeros(length(obj.index), mNew);
        for h = 1:length(obj.index)
          cols = find(Z(obj.index(h),:,jj));
          newZ(obj.index(h), ss.m + nonZeroZpos(h,cols), jj) = Z(obj.index(h),cols,jj);
        end
      end
    end
    
    %% Accumulator helper methods
    function obj = setAugmenting(obj, ss)
      %
      
      % Find the elements of Z that care about the accumulators
      usedZaccumLoc = any(ss.Z(obj.index,:,:) ~= 0, 3);
      [obj.accumObsDims, obj.accumStates] = find(usedZaccumLoc);
      
      % Indentify the elements of the state that need accumulation - anything
      % that loads onto an observation that has an associate accumulator
      possibleRows = [obj.accumStates ...
        obj.accumulatorTypes(obj.accumObsDims) ...
        obj.horizon(obj.accumObsDims, :) ...
        obj.calendar(obj.accumObsDims, :)];
      
      [APsi, ~, obj.usedAccumulators] = unique(possibleRows, 'rows');
      obj.augmentingStates = APsi(:, 1);
      
      obj.augmentingAccumulatorTypes = APsi(:, 2);
      obj.augmentingHorizon = APsi(:, 3:size(obj.horizon, 2) + 2);
      obj.augmentingCalendar = APsi(:, size(obj.horizon, 2) + 3:end);
    end
    
    function [mNew, ColAdd, RowPos] = accumAugmentedStateDims(obj, ss)
      % Checking we have the correct number of lags of the states that need
      % accumulation to be compatible with Horizon
      
      states = unique(obj.accumStates);
      maxHor = max(obj.horizon, [], 2);
      
      AddLags = zeros(length(states), 1);
      ColAdd  = zeros(length(states), max(maxHor));
      RowPos  = zeros(length(states), max(maxHor));
      mNew = ss.m;
      for iSt = 1:length(states)
        mHor = max(maxHor(obj.accumObsDims(states(iSt) == obj.accumStates)));
        [numlagsjj, lagposjj] = ss.LagsInState(states(iSt));
        
        AddLags(iSt) = max(mHor-1-numlagsjj-1, 0);
        RowPos(iSt, 1:numlagsjj+1) = [states(iSt) lagposjj'];
        if AddLags(iSt) > 0
          ColAdd(iSt, numlagsjj+2) = lagposjj(end);
        end
        
        if AddLags(iSt) > 1
          ColAdd(iSt, numlagsjj+3:numlagsjj+1+AddLags(iSt)) = mNew:mNew+AddLags(iSt)-1;
        end
        
        RowPos(iSt, numlagsjj+2:numlagsjj+1+AddLags(iSt)) = mNew+1:mNew+AddLags(iSt);
        mNew = mNew+AddLags(iSt);
      end
    end
    
  end
end