function data = loadData(dataName)

thisFile = mfilename('fullpath');

testDir = [subsref(strsplit(pwd, 'StateSpace'), ...
  struct('type', '{}', 'subs', {{1}})) 'StateSpace\test\data'];
dataStr = fileread(fullfile(testDir, 'Nile.dat'));
lineBreaks = strfind(dataStr, sprintf('\n'));
dataStr(1:lineBreaks(1)) = [];
data = sscanf(dataStr, '%d');
