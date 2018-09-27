% Test basic functions of AbstractSystem and utility functions
% David Kelley, 2017

classdef AbstractSystem_test < matlab.unittest.TestCase
  
  properties
   
  end

  methods(TestClassSetup)
    function setupOnce(testCase) %#ok<MANU>
      baseDir = fileparts(fileparts(mfilename('fullpath')));
      addpath(baseDir);
    end
  end
  
  methods (Test)  

  end
end