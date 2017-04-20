function [numeric, G] = numericGradient(ss, tm, y, delta)
% Iterate through theta, find changes in likelihood

%% Gradient of likelihood
ss.useAnalyticGrad = false;
ss.delta = delta;
[~, numeric] = ss.gradient(y, tm);

%% Gradients of parameters
G = [];

if nargout > 1
  % Preallocate
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
  
  % Iterate
  for iT = 1:tm.nTheta
    iTheta = theta;
    iTheta(iT) = iTheta(iT) + delta;
    
    ssTest = tm.theta2system(iTheta);
%     ssTest = ssTest.checkSample(y);
%     ssTest = ssTest.setDefaultInitial();
       
    ssTest = ssTest.prepareFilter(y);
    
    % FIXME?: systemParam doesn't include initial values
    for iP = 1:length(ss.systemParam)                   
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
end

