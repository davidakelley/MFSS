function outStr = tm2matList(tm)
% Create a cell vector of which parameter each theta element influences

outStr  = cell(tm.nTheta, 1);

params = tm.fixed.systemParam;
matParam = repmat({''}, [tm.nTheta, length(params)]);
for iP = 1:length(params)
  indexes = tm.index.(params{iP});
  matParam(indexes(indexes~=0), iP) = repmat(params(iP), [sum(indexes(:)~=0), 1]);
end

for iT = 1:tm.nTheta
  goodStrs = matParam(iT,:);
  goodStrs(cellfun(@isempty, goodStrs)) = [];
  outStr{iT} = strjoin(goodStrs, ', ');
end