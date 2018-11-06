function make_toolbox(major, minor, build)
% Build script to create the toolbox.
%
% To create the toolbox: 
%   1. Add the /src and /build paths to your Matlab path. Make sure you haven't added
%   anything else to your Matlab path. 
%   2. Run this file.
%   3. Test that the toolbox works by removing the earlier added paths, installing the
%   toolbox and running a model.

buildDir = fileparts(mfilename('fullpath'));

%% Get prj file
% Edit .prj file to update version numbers
prjFile = fullfile(buildDir, '..', 'toolbox.prj');
prjText = fileread(prjFile);

% Save new .prj file with updated version numbers
prjTextNew = regexprep(prjText, '<param.version>.*<\/param.version>', ...
  sprintf('<param.version>%d.%d.%d</param.version>', major, minor, build));

fid = fopen(prjFile, 'w');
fwrite(fid, prjTextNew);
fclose(fid);

%% Make sure the path is set correctly
% Make sure the source files are included
baseDir = fileparts(buildDir);
addpath(fullfile(baseDir, 'src'));

%% Create toolbox file
matlab.addons.toolbox.packageToolbox(...
  fullfile(buildDir, '..', 'toolbox.prj'), ...
  fullfile(buildDir, '..', 'MFSS.mltbx'))

fprintf('Completed build of MFSS v%d.%d.%d\n', major, minor, build);
