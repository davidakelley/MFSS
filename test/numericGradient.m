function [numeric, G, GfOut] = numericGradient(ss, tm, y)
% Iterate through theta, find changes in likelihood

theta = tm.system2theta(ss);
delta = -ss.delta;

%% Gradient of likelihood
[~, logl_fix] = ss.filter(y);

theta = tm.system2theta(ss);
for iT = 1:tm.nTheta
  iTheta = theta;
  iTheta(iT) = iTheta(iT) + delta;
  
  ssTest = tm.theta2system(iTheta);
  [~, logl_delta] = ssTest.filter(y);
  
  numeric(iT) = (logl_delta - logl_fix) ./ delta;
end

% %% Gradient of likelihood
% [~, numeric, fOutFix] = ss.gradient(y, [], tm);

ss = ss.prepareFilter(y, []);

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
    
    ssTest = ssTest.prepareFilter(y);
    
    % FIXME?: systemParam doesn't include initial values
    for iP = 1:length(ss.systemParam)
      iParam = ss.systemParam{iP};
      if isempty(ss.(iParam))
        continue;
      end
      
      G.(iParam)(iT, :, :) = multidimGrad(ss.(iParam), ssTest.(iParam), ...
        delta, size(G.(iParam)(iT, :, :)));
    end
  end
end

%% Gradients of fOut
GfOut = [];

if nargout > 2
  % Preallocate
  params = fieldnames(fOutFix);

  for iP = 1:length(params)
    iParam = params{iP};
    
    if isempty(fOutFix.(iParam))
      GfOut.(iParam) = [];
      continue;
    end
    shape = size(fOutFix.(iParam));
    if length(shape) < 3
      shape = [shape 1]; %#ok<AGROW>
    end
    
    GfOut.(iParam) = zeros(tm.nTheta, prod(shape(1:2)), shape(3));
  end
  
  fOutTest = cell(tm.nTheta,1);
  fOutTestDown = cell(tm.nTheta,1);
  % Iterate
  for iT = 1:tm.nTheta
    iTheta = theta;
    iTheta(iT) = iTheta(iT) + delta;
    
    ssTest = tm.theta2system(iTheta);
    [~, ~, fOutTest{iT}] = ssTest.filter(y);
    
    iTheta = theta;
    iTheta(iT) = iTheta(iT) - delta;
    
    ssTestDown = tm.theta2system(iTheta);
    [~, ~, fOutTestDown{iT}] = ssTestDown.filter(y);
    
    for iP = 1:length(params)
      iParam = params{iP};
      if isempty(fOutTest{iT}.(iParam))
        continue;
      end
      
      GfOut.(iParam)(iT, :, :) = multidimGrad(fOutTestDown{iT}.(iParam), fOutTest{iT}.(iParam), ...
        delta);
    end
  end
end

end


function grad = multidimGrad(ptEval, deltaEval, delta, shape)

assert(all(size(ptEval) == size(deltaEval)));

switch ndims(ptEval)
  case 1
    % Vector
    sliceSize = numel(ptEval);
    gradShape = [1 sliceSize 1];
  case 2
    % Matrix
    sliceSize = numel(ptEval);
    gradShape = [1 sliceSize 1];
  case 3
    % Time varrying matrix
    sliceSize = numel(ptEval(:,:,1));
    nSlices = size(ptEval, 3);
    gradShape = [1 sliceSize nSlices];
  otherwise
    error('Development error.');
end
if nargin > 3 && ~isempty(shape)
  gradShape = shape;
end

grad = reshape((deltaEval - ptEval) ./ delta, gradShape);

end