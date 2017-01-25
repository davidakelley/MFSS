function K = genCommutation(m)
% Generate commutation matrix
%
% K = genCommutation(m) returns a commutation matrix for a
% square matrix A of size m such that K * vec(A) = vec(A').

% From Magnus & Neudecker (1979), Definition 3.1, a commutation matrix is
% "a suqare mn-dimensional matrix partitioned into mn sub-matricies of
% order (n, m) such that the ij-th submatrix has a 1 in its ji-th position
% and zeros elsewhere."


K = zeros(m^2);
for iComm = 1:m
  for jComm = 1:m
    K = K + kron(eyePart(m, iComm, jComm), eyePart(m, iComm, jComm)');
  end
end

end

function out = eyePart(m, i, j) 
  out = [zeros(i-1, m); zeros(1, j-1), 1, zeros(1, m-j); zeros(m-i, m)];
end