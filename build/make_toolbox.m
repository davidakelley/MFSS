% make_toolbox
% Build script to create the toolbox.

buildDir = fileparts(mfilename('fullpath'));

matlab.addons.toolbox.packageToolbox(...
  fullfile(buildDir, '..', 'toolbox.prj'), ...
  fullfile(buildDir, 'MFSS.mltbx'))