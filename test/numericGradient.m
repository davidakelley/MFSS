function numeric = numericGradient(ss, tm, y, delta)
% Iterate through theta, find changes in likelihood

numeric = nan(tm.nTheta, 1);
[~, logl_fix] = ss.filter(y);

theta = tm.system2theta(ss);
for iT = 1:tm.nTheta
  iTheta = theta;
  iTheta(iT) = iTheta(iT) + delta;
  
  ssTest = tm.theta2system(iTheta);
  
  [~, logl_delta] = ssTest.filter(y);
  numeric(iT) = (logl_delta - logl_fix) ./ delta;
end

end

