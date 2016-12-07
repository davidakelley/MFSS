function printNice(ss, tm, analytic, numeric)
% Utiltiy function to print the gradient output nicely, currently not working

%       paramLen = structfun(@length, ss.thetaMap.elem);

systemParam = {'Z', 'd', 'H', 'T', 'c', 'R', 'Q', 'a0', 'P0'}';

separateNames = arrayfun(@(len, name) repmat(name, [len 1]), ...
  cellfun(@length, ss.parameters)', systemParam, 'Uniform', false);

nameVec = cat(1, separateNames{:});
nameVec(~ss.thetaMap.estimated) = [];

out = [{'Param', 'Analytic', 'Numeric', 'Difference', 'Relative'}; ...
  [nameVec num2cell(analytic) num2cell(numeric) ...
  num2cell(analytic - numeric) num2cell(analytic ./ numeric -1)]];
disp(out);

end