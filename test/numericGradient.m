function numeric = numericGradient(ss, tm, y, delta)
% Iterate through theta, find changes in likelihood

numeric = nan(tm.nTheta, 1);

% We need to use the multivariate filter to make sure we're consisent.
ss.filterUni = false;
[~, logl_fix] = ss.filter(y);

theta = tm.system2theta(ss);
for iT = 1:tm.nTheta
  iTheta = theta;
  iTheta(iT) = iTheta(iT) + delta;
  
  ssTest = tm.theta2system(iTheta);
  ssTest = ssTest.checkSample(y);
  
  % We need to use the multivariate filter to make sure we're consisent.
  ssTest.filterUni = false;
      
  [~, logl_delta] = ssTest.filter(y);
  numeric(iT) = (logl_delta - logl_fix) ./ delta;
end

end

