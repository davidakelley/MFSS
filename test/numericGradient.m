function [numeric, G] = numericGradient(ss, tm, y, delta)
% Iterate through theta, find changes in likelihood

numeric = nan(tm.nTheta, 1);
for iP = 1:length(ss.systemParam)
  iParam = ss.systemParam{iP};
  
  if isempty(ss.(iParam))
    G.(iParam) = [];
    continue;
  end
  
  if any(strcmpi(iParam, {'d', 'c', 'a0'}))
    sliceSize = numel(ss.(iParam)(:,1));
    nSlices = size(ss.(iParam), 2);
  else
    sliceSize = numel(ss.(iParam)(:,:,1));
    nSlices = size(ss.(iParam), 3);
  end
    
  G.(iParam) = zeros(tm.nTheta, sliceSize, nSlices);
end
  
% We need to use the multivariate filter to make sure we're consisent.
ss.filterUni = false;
[~, logl_fix] = ss.filter(y);

theta = tm.system2theta(ss);
for iT = 1:tm.nTheta
  iTheta = theta;
  iTheta(iT) = iTheta(iT) + delta;
  
  ssTest = tm.theta2system(iTheta);
  ssTest = ssTest.checkSample(y);
  ssTest = ssTest.setDefaultInitial();
  
  % We need to use the multivariate filter to make sure we're consisent.
%   ssTest.filterUni = false;
      
  [~, logl_delta] = ssTest.filter(y);
  numeric(iT) = (logl_delta - logl_fix) ./ delta;
  for iP = 1:length(ss.systemParam)                   % FIXME: systemParam doesn't include initial values
    iParam = ss.systemParam{iP};
    if isempty(ss.(iParam))
      continue;
    end
    
    if any(strcmpi(iParam, {'d', 'c', 'a0'}))
      sliceSize = numel(ss.(iParam)(:,1));
      nSlices = size(ss.(iParam), 2);
    else
      sliceSize = numel(ss.(iParam)(:,:,1));
      nSlices = size(ss.(iParam), 3);
    end
    
    G.(iParam)(iT, :, :) = reshape((ssTest.(iParam) - ss.(iParam)) ./ delta, ...
      [1 sliceSize nSlices]);
  end
end

end

