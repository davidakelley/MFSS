function make_toolbox(major, minor)
% Build script to create the toolbox.

buildDir = fileparts(mfilename('fullpath'));

%% Iterate version number


% Edit .prj file to update version numbers
prjFile = fullfile(buildDir, '..', 'toolbox.prj');
prjText = fileread(prjFile);

prjVers = regexp(prjText, ...
  '<param.version>([\d*])\.([\d*])\.([\d*])<\/param.version>', 'tokens');
oldMajor = str2double(prjVers{1}{1});
oldMinor = str2double(prjVers{1}{2});
oldBuild = str2double(prjVers{1}{3});

% Set new version number
if nargin < 1
  major = oldMajor;
  newMajorVersion = false;
else
  assert(major == oldMajor | major == oldMajor + 1, ...
    ['Cannot decrease major version or iterate version by more than one. ' ...
    'Current minor version %d.'], oldMajor);
  newMajorVersion = major == oldMajor + 1;
end

if nargin < 2
  minor = oldMinor;
  newMinorVersion = false;
else
  assert(newMajorVersion | minor == oldMinor | minor == oldMinor + 1, ...
    ['Cannot decrease minor version within major release or iterate ', ...
    'version by more than one. Current minor version %d.'], oldMajor);
  newMinorVersion = minor == oldMinor + 1;
end

if newMajorVersion
  assert(nargin < 2 | minor == oldMinor, 'Minor release set to 0 for new major release.');
  minor = 0;
end
if newMajorVersion || newMinorVersion
  build = 0;
else
  build = oldBuild + 1;
end
    
% Save new .prj file with updated version numbers
prjTextNew = regexprep(prjText, '<param.version>.*<\/param.version>', ...
  sprintf('<param.version>%d.%d.%d</param.version>', major, minor, build));

fid = fopen(prjFile, 'w');
fwrite(fid, prjTextNew);
fclose(fid);


%% Create toolbox file

matlab.addons.toolbox.packageToolbox(...
  fullfile(buildDir, '..', 'toolbox.prj'), ...
  fullfile(buildDir, '..', 'MFSS.mltbx'))