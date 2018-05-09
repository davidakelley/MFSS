function make_toolbox(major, minor)
% Build script to create the toolbox.
%
% To create the toolbox: 
%   1. Add the /src and /build paths to your Matlab path. Make sure you haven't added
%   anything else to your Matlab path. 
%   2. Run this file, calling it with a new major and minor build number unless you are
%   creating a bug-fix release. 
%   3. Test that the toolbox works by removing the earlier added paths, installing the
%   toolbox and running a model.
%   4. Commit the changes to git with a commit that starts with the version number.

buildDir = fileparts(mfilename('fullpath'));

%% Get version number from prj file
% Edit .prj file to update version numbers
prjFile = fullfile(buildDir, '..', 'toolbox.prj');
prjText = fileread(prjFile);

prjVers = regexp(prjText, ...
  '<param.version>([\d*]*)\.([\d*]*)\.([\d*]*)<\/param.version>', 'tokens');
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

%% Compile HTML documentation
confFile = fullfile(buildDir, '..', 'docs', 'source', 'conf.py');
confText = fileread(confFile);

confTextTemp = regexprep(confText, 'version = u''([\d*]*)\.([\d*]*)''', ...
  sprintf('version = u''%d.%d''', major, minor));
confTextNew = regexprep(confTextTemp, 'release = u''([\d*]*)\.([\d*]*)''', ...
  sprintf('version = u''%d.%d.%d''', major, minor, build));
if major ~= oldMajor || minor ~= oldMinor
  assert(~isequal(confTextNew, confText), ...
    'No change in documentation version.');
end

% docs_status = compile_docs;
% if docs_status == 0
%   fprintf('Successfully compiled documentation.\n');
% else
%   disp('Error in compiling documentation. Run ''make html'' in the docs folder.');
% end

%% Iterate version number in prj file
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