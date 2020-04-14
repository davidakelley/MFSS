% Replication code for A Practitioner’s Guide and Matlab Toolbox for Mixed 
% Frequency State Space Models by Scott A. Brave, R. Andrew Butters and
% David Kelley. 
%
% To replicate the figures in the paper, run this file. 

%% Check that the toolbox is installed
assert(exist('StateSpaceEstimation.m', 'file') == 2, ...
  'MFSS toolbox not installed.');

%% Run the examples
clear
pgmtmfss1_dfm

clear
pgmtmfss2_var_irf

clear
pgmtmfss3_var_disagg

clear
pgmtmfss4_trend_cycle

clear
pgmtmfss5_r_star
