function [factor, Gamma, Sigma, YHat, XHat, A] = StockWatsonMF(X, varargin)
% STOCKWATSONMF Latent factor estimation of time-series data
%
% Calculate the balanced panel and time series of the factor(s) and
% loadings for a panel of time series with potentially missing observations
%
% Input:
%   X        ~ time series panel (T x N)
% Optional inputs: Passed as either a structure (as the second input)
% or as name-value pairs. A structure and name-value pair inputs can be
% used in combination, in which case the name-value pair arguments will
% override any structure option inputs.
%   factors  ~ number of factors to be estimated (default: 1)
%   accum    ~ structure detailing accumulation
%     index     ~ linear indexes of series needing accumulation
%     calendar  ~ calendar of observations for accumulated series
%     horizon   ~ periods covered by each observation
%   tol      ~ convergence tolerance (default: 1e-6)
%   verbose  ~ display progress (default: true)
%
% Output:
%   F         ~ estimate of the factors  (T x factors)
%   Gamma     ~ estimate of the loadings (N x factors)
%   Sigma     ~ estimate of the error variance matrix (N x N)
%   YHat      ~ estimate of the balanced, disaggregated time series (T x N)
%   XHat      ~ estimate of the balanced, aggregated time series (T x N)
%
% Lower-frequency data with a regular calendar are assumed to be triangle-
% averages of the appropriate period (the default accumulator).
%
% Given the accumulator definitions, aggregation matricies (A) will be
% generated for each type of accumulator. For an observed X, the underlying
% data Y will be estimated such that X_i = A * Y_i:
%   E(Y_i | X) = E(Y_i | X_i, F, Gamma_i)
%              = F * Gamma_i + A' * pinv(A * A') * (X_i - A * F * Gamma_i)
%
% See Stock & Watson (2002, JBES) Appendix A for details.

% David Kelley, 2016-2017

[X, opts] = handleInputs(X, varargin);

%% Aggregation properties
accum = createAccumulator(X, opts);
[firstObs, lastObs] = determineSample(X, accum, opts);
balanceInx = (firstObs:lastObs)';

[A, Aselect, aggType] = genAggregationMats(X, accum, balanceInx);

% Error distribution matricies
errAdj = cellfun(@(x) x' * pinv(x * x'), A, 'Uniform', false);

% Find out where we can evaluate errors and get the low-frequency series in
% their low-frequency form to speed up the iterative evaluations.
allowableErrorsPos = cellfun(@(x, y) x * y, errAdj, Aselect, 'Uniform', false);
cleanData = X(balanceInx, :);
cleanData(isnan(X(balanceInx, :))) = 0;

hasHiFreqInfo = false(size(balanceInx, 1), size(X, 2));
aggSer = cell(size(X, 2), 1);
for iS = 1:size(X, 2)
  hasHiFreqInfo(:,iS) = ~logical(allowableErrorsPos{aggType(iS)} * isnan(X(balanceInx, iS)));
  aggSer{iS} = Aselect{aggType(iS)} * cleanData(:,iS);
end

%% Estimate unobserved data, loadings, and factors
nSeries = size(X, 2);

% Initialize balanced panel as zeros, starting point shouldn't matter
YHat = X(balanceInx, :);
YHat(isnan(YHat)) = 0;

gap = 1e10; SSR0 = 1e10; iter = -1;
while abs(gap) > opts.tol
  iter = iter+1;
  
  % Take factors from PCA using fitted high-frequency data
  Yhat = zscore(YHat);
  [Gamma, factor] = pca(Yhat, 'NumComponents', opts.factors, ...
    'Centered', false);
  
  % Get corrected projected data
  for iS = 1:nSeries
    YHat(:,iS) = enforceAgg(aggSer{iS}, factor *  Gamma(iS,:)', ...
      A{aggType(iS)}, errAdj{aggType(iS)}, hasHiFreqInfo(:,iS));
  end
  
  % Report improvement, store values for next iteration
  residuals = (YHat - factor * Gamma');
  SSR = sum(sum(residuals.^2)) ./ numel(residuals);
  reportIter(iter, SSR, SSR0, opts);
  
  gap = SSR - SSR0;
  SSR0 = SSR;
end

%% Pass outputs that match time series of inputs
nPeriods = size(X, 1);

prePeriods = firstObs - 1;
postPeriods = nPeriods - lastObs;

if nargout >= 5
  XHat = nan(size(YHat));
  for iS = 1:nSeries
    XHat(:, iS) = Aselect{aggType(iS)}' * A{aggType(iS)} * YHat(:, iS);
    XHat(~sum(Aselect{aggType(iS)}), iS) = nan;
  end
  
  XHat = [nan(prePeriods, nSeries); XHat; nan(postPeriods, nSeries)];
end

factor = [nan(prePeriods, opts.factors); factor; nan(postPeriods, opts.factors)];
YHat = [nan(prePeriods, nSeries); YHat; nan(postPeriods, nSeries)];

Sigma = diag(diag(residuals' * residuals)) ./ size(balanceInx,1);

if opts.verbose
  if iter <= 1, fprintf('Input data already balanced.\n');  end
  fprintf('%s\n', opts.line('-'));
end

end

function accum = createAccumulator(X, opts)
% Define an accumulator structure by finding any series that have regular
% patterns of observations and assigning them a triangle accumulator with a
% horizon equal to the observation frequency.

% Find series with regular observation patterns
seriesFreqs = cell(1, size(X, 2));
for iS = 1:size(X, 2)
  obsInx = find(~isnan(X(:, iS)));
  obsDiff = obsInx(2:end) - obsInx(1:end-1);
  seriesFreqs{iS} = unique(obsDiff);
end

% Define the default accumulator for those found above
needHarvey = cellfun(@(x) length(x) == 1 && x ~= 1, seriesFreqs);
defAccum = struct;
defAccum.index = find(needHarvey);
defAccum.horizon = repmat(cell2mat(seriesFreqs(needHarvey)), [size(X, 1)+1 1]);
defAccum.calendar = nan(size(defAccum.horizon));
for iS = 1:sum(needHarvey)
  dataSeries = defAccum.index(iS);
  firstObs = find(~isnan(X(:, dataSeries)), 1, 'first');
  offset = mod(firstObs, seriesFreqs{dataSeries});
  defAccum.calendar(:, iS) = mod((1:size(X, 1)+1) + offset - 1, ...
    seriesFreqs{dataSeries})' + 1;
end

% Merge default and provided accum structures
useDefaultAccum = setdiff(defAccum.index, opts.accum.index);

accum = struct;
[accum.index, sortOrder] = sort([opts.accum.index useDefaultAccum]);
calendar = [opts.accum.calendar defAccum.calendar];
accum.calendar = calendar(:, sortOrder);
horizon = [opts.accum.horizon defAccum.horizon];
accum.horizon = horizon(:, sortOrder);

if opts.verbose && ~isempty(useDefaultAccum)
  fprintf('Default accumulator for series %s\n%s\n', ...
    strjoin(arrayfun(@num2str, useDefaultAccum, 'Uniform', false), ', '), ...
    opts.line('-'));
end

end

function [firstObs, lastObs] = determineSample(y, accum, opts)
% Find index of X that can be balanced - get the first period covered by
% enough series' accumulators so we can get the number of factors requested

nSeries = size(y, 2);

seriesFirst = nan(nSeries, 1);
seriesLast = nan(nSeries, 1);
for iS = 1:nSeries
  if ~ismember(iS, accum.index)
    seriesFirst(iS) = find(~isnan(y(:,iS)), 1, 'first');
  else
    iAccum = find(iS == accum.index, 1);
    isAvg = ~any(accum.calendar(:, iAccum) == 0);
    seriesFirst(iS) = find(~isnan(y(:,iS)) & ...
      any(cumsum(accum.calendar(1:end-1, iAccum) == isAvg), 2), 1);
  end
  seriesLast(iS) = find(~isnan(y(:,iS)), 1, 'last');
end
sortFirst = sort(seriesFirst);
firstObs = sortFirst(opts.factors);
sortLast = sort(seriesLast, 'descend');
lastObs = sortLast(opts.factors);

end

function [A, Aselect, aggType] = genAggregationMats(X, accum, balanceInx)
% Construct aggregation and selection matricies from the Harvey-style
% accumulator strucutres.

nPeriods = size(X, 1);
nSeries = size(X, 2);

aggType = ones(1, nSeries);
aggType(accum.index) = 2:size(accum.index, 2)+1;

A = cell(length(unique(aggType)), 1);
Aselect = cell(size(A));

% Generate an aggregation matrix for each series
for iA = 1:1+length(accum.index)
  if iA == 1
    % No aggregation case: Sum accumulator with zeros everywhere
    calendar = zeros(size(X, 1)+1, 1);
    horizon = ones(size(X, 1)+1, 1);
  else
    calendar = accum.calendar(:,iA-1);
    horizon = accum.horizon(:,iA-1);
  end

  isAverage = ~any(calendar == 0);
  assert(isAverage || all(ismember(calendar, [0 1])), 'Bad calendar');
  
  % Get the high-frequency start and end of each low-frequency period
  periodStart = find(calendar == isAverage);
  periodEnd = periodStart - 1;
  if periodEnd(1) < periodStart(1), periodEnd(1) = []; end
  periodStart = periodStart(1:size(periodEnd, 1));
  assert(size(periodStart, 1) == size(periodEnd, 1));

  % Set up aggregation matricies
  if isAverage
    % See Stock & Watson (2002) for details behind computation here
    S = calendar(periodEnd);
    H = horizon(periodEnd);

    minSH = @(iLowF) min(S(iLowF), H(iLowF));
    aggWeights = @(iLowF) [1:minSH(iLowF)-1 ...
      repmat(S(iLowF), [1 H(iLowF)+S(iLowF)-2*minSH(iLowF)+1]) ...
      minSH(iLowF)-1:-1:1] ./ S(iLowF);
    
    nWeights = @(iLowF) H(iLowF)+S(iLowF)-1;
    getBasePeriods = @(iLowF) (periodEnd(iLowF) - nWeights(iLowF) + 1):periodEnd(iLowF);
  else
    % Simple case: sum all data in low-frequency period
    aggWeights = @(x) 1;
    getBasePeriods = @(iLowF) periodStart(iLowF):periodEnd(iLowF);
  end
  
  nAggPeriods = length(periodStart); 
  tempSelect = zeros(nAggPeriods, nPeriods);
  generatedA = zeros(nAggPeriods, nPeriods);
  for iLowF = 1:nAggPeriods
    basePeriods = getBasePeriods(iLowF);
    if any(basePeriods < 1) || max(basePeriods) > nPeriods
      continue
    end
    generatedA(iLowF, basePeriods) = aggWeights(iLowF);
    tempSelect(iLowF, periodEnd(iLowF)) = 1;
  end
  
  A{iA} = generatedA(:, balanceInx);
  Aselect{iA} = tempSelect(:, balanceInx);
end

end

function predictedData = enforceAgg(aggSer, naivePred, A, errAdj, hasHiFreqInfo)
% Predict missing/unobserved high-frequency values and enforce aggregation
%   E(Y_i | X) = F * l_i + A' * pinv(A * A') * (X_i - A * F * l_i)

naievePredClean = naivePred;
naievePredClean(~hasHiFreqInfo) = 0;

projErrors = aggSer - A * naievePredClean;
predictedData = naivePred + errAdj * projErrors;

end

function reportIter(iter, SSR, SSR0, opts)
% Command line reporting of estimation progress.
if iter <= 0
  return
end

% assert(SSR - SSR0 <= 0, 'Increasing likelihood.');

if opts.verbose && ~(iter == 1 && (SSR - SSR0) == 0)
  if iter > 2
    bspace = repmat('\b', [1 30]);
  else
    if iter == 1
      fprintf(' Iteration | SSE differences\n%s\n', opts.line('-'));
    end
    bspace = [];
  end
  tolLvl = num2str(abs(floor(log10(opts.tol)))+1);
  
  diff_SSE = sprintf(['%16.' tolLvl 'f'], SSR - SSR0);
  fprintf([bspace '%10.0d | %s\n'], iter, diff_SSE(1:16));
end
end

function [X, opts] = handleInputs(X, vararginCell)
% Parse inputs, determine options to be used later.
if istable(X)
  tableX = X;
  X = tableX{:,:};
end

% Take an options structure as a potential first optional input
% (Doesn't inputParser handle this already?)
if nargin > 1 && ~isempty(vararginCell) && ~ischar(vararginCell{1})
  opts = vararginCell{1};
  vararginCell(1) = [];
else
  opts = struct;
end

inP = inputParser();
inP.addParameter('factors', 1, @(x) x == floor(x));
inP.addParameter('accum', Accumulator([], [], []), @(x) isa(x, 'Accumulator'));
inP.addParameter('tol', 1e-6, @isnumeric);
inP.addParameter('verbose', true, @islogical);
inP.parse(opts, vararginCell{:});
opts = inP.Results;

% Check the number of accumulators
nAccum = size(opts.accum.index, 2);
assert(all(ismember(opts.accum.index, 1:size(X, 2))));
if size(opts.accum.calendar, 1) == nAccum && ...
    size(opts.accum.calendar, 2) ~= nAccum
  opts.accum.calendar = opts.accum.calendar';
end
assert(size(opts.accum.index, 2) == size(opts.accum.calendar, 2));
if size(opts.accum.horizon, 1) == nAccum && ...
    size(opts.accum.horizon, 2) ~= nAccum
  opts.accum.horizon = opts.accum.horizon';
end
assert(size(opts.accum.index, 2) == size(opts.accum.horizon, 2));

% Check accumulator time series length. Must be one period longer than data
assert(isempty(opts.accum.calendar) || ...
  size(opts.accum.calendar, 1) == size(X, 1)+1);
assert(isempty(opts.accum.horizon) || ...
  size(opts.accum.horizon, 1) == size(X, 1) + 1);

if opts.verbose
  opts.line = @(char) repmat(char, [1 29]);
  fprintf('\nStock & Watson (2002)\n%s\n', opts.line('='));
end
end
