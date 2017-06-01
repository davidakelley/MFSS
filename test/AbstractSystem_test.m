% Test basic functions of AbstractSystem and utility functions
% David Kelley, 2017

classdef AbstractSystem_test < matlab.unittest.TestCase
  
  properties
   
  end

  methods(TestClassSetup)
    function setupOnce(testCase) %#ok<MANU>
      baseDir =  [subsref(strsplit(mfilename('fullpath'), 'MFSS'), ...
        struct('type', '{}', 'subs', {{1}})) 'MFSS'];
      addpath(baseDir);
    end
  end
  
  methods (Test)  
%     function testPseudoinv(testCase)
%       
%     end
    
    function testCommutation(testCase)
      mat1 = rand(5);
      comm = AbstractSystem.genCommutation(5);
      vec = @(M) reshape(M, [], 1);

      % Definition of commutation matrix: K * vec(A) = vec(A')
      testCase.verifyEqual(comm * vec(mat1), vec(mat1'));
    end
    
  end
end