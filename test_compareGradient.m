function test_compareGradient(ss)
timePers = 5;

rng(0);
[y, alpha] = generateData(ss, timePers);
tm = ThetaMap.ThetaMapAll(ss);
% tm.index.H = diag(diag(tm.index.H));
% tm = tm.validateThetaMap();

% plot(alpha');

tic;
numeric = numericGradient(ss, tm, y, 1e-8);
tocN = toc;

theta = tm.system2theta(ss);
tic;
[~, grad] = ss.gradient(y, tm, theta);
tocA = toc;

fprintf('Analytic took %3.2f%% of the numeric time.\n', tocA./tocN*100);

diffTab = array2table([numeric grad (grad - numeric) (grad - numeric)./numeric ], ...
  'VariableNames', {'Numeric', 'Univariate', 'Numeric_Diff', 'RelativeDiff'});
diffTab.Variable = tm2matList(tm);
diffTab = diffTab(:,[5 1:4]);
disp(diffTab);

end

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

end