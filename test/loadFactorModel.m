function [data, opts, dims] = loadFactorModel()
% Get data and model options from factor model

baseDir =  [subsref(strsplit(mfilename('fullpath'), 'StateSpace'), ...
  struct('type', '{}', 'subs', {{1}})) 'StateSpace'];

saveFile = fullfile(baseDir, 'test', 'data', 'bbk_data.mat');
if exist(saveFile, 'file')
  fileInfo = dir(saveFile);
  if floor(datenum(fileInfo.date)) == floor(now)
    saveData = load(saveFile);
    data = saveData.data; opts = saveData.opts; dims = saveData.dims;
    return
  end
end

ssDir = pwd;
cd('C:\Users\g1dak02\Documents\MATLAB\bbk');
[opts, dims, paths] = estimationOptions();
opts.targets = {'GDPH'};
[opts, dims] = getTargetSpec(opts, dims);
[data, opts, dims] = loadData(opts, dims, paths);
cd(ssDir);
resetPath;

save(saveFile, 'data', 'opts', 'dims');

end