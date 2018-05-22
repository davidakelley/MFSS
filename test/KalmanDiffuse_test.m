% Test the diffuse Kalman filter/smoother
% David Kelley, 2017

classdef KalmanDiffuse_test < matlab.unittest.TestCase
  
  properties
   
  end

  methods(TestClassSetup)
    function setupOnce(testCase) %#ok<MANU>
      baseDir = fileparts(fileparts(mfilename('fullpath')));
      addpath(baseDir);
    end
  end
  
  methods (Test)
    function testDiffusedt(testCase)
      % Set up a model where we observe the level and growth rate of a
      % process. With no errors, even when we don't observe the level we
      % should be able to recover it so long as we still observe the growth
      % rate. 
      
      % AR(1) process for growth rate
      phi = 0.999;
      % Variances of level and growth rate process
      sigma_level = 0.5;
      sigma_gr = 1;
      
      Z = [1 0; 0 1];
      H = [0 0; 0 0];
      T = [1 0.01; 0 phi];
      Q = diag([sigma_level sigma_gr]);
      ss = StateSpace(Z, H, T, Q);
      [y, alpha] = generateData(ss, 600);
      
      y(1,1:300) = nan;
      [~, ~, fOut] = ss.filter(y);
      alphaHat = ss.smooth(y);
      
      sskappa = ss.setDefaultInitial();
      sskappa.P0(~isfinite(sskappa.P0)) = 1e8;
      alphaHatKappa = sskappa.smooth(y);
      
      testCase.verifyEqual(fOut.dt, 301);
      testCase.verifyEqual(alpha(2,:), alphaHat(2,:), 'AbsTol', 1e-15);
      %testCase.verifyEqual(alpha(1,1:300), alphaHat(1,1:300), 'AbsTol', 1e-14);
      testCase.verifyEqual(alpha(1,301:end), alphaHat(1,301:end), 'AbsTol', 1e-14);
    end
    
  end
end