% W_test

nRepl = 1000;
nList = [1:10 15:5:50 60:10:100];

times = zeros(length(nList), 4);
for in = 1:length(nList)
  n = nList(in);
  
  disp(n);
  ind = rand(1, n) > 0.5;
  
%   times0 = zeros(nRepl, 1);
%   for iT = 1:nRepl
%     tic;
%     W = eye(n);
%     W = W((ind==1),:);
%     kronWW = kron(W', W');
%     times0(iT) = toc;
%   end
  
  times1 = zeros(nRepl, 1);
  for iT = 1:nRepl
    tic;
    W = sparse(eye(n));
    W = W((ind==1),:);
    kronWW = kron(W', W');
    times1(iT) = toc;
  end
  
  times2 = zeros(nRepl, 1);
  for iT = 1:nRepl
    tic;
    W = logical(sparse(eye(n)));
    W = W((ind==1),:);
    kronWW = kron(W', W');
    times2(iT) = toc;
  end
  
  times3 = zeros(nRepl, 1);
  for iT = 1:nRepl
    tic;
    W = sparse(logical(eye(n)));
    W = W((ind==1),:);
    kronWW = kron(W', W');
    times3(iT) = toc;
  end
  
  W_base = sparse(logical(eye(n)));
  
  times4 = zeros(nRepl, 1);
  for iT = 1:nRepl
    tic;
    W = W_base((ind==1),:);
    kronWW = kron(W', W');
    times4(iT) = toc;
  end
  
  times(in, :) = [mean(times1) mean(times2) mean(times3) mean(times4)];
  
end


