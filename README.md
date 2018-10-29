MFSS - Mixed-Frequency State Space Modeling
===========================================

MFSS provides functions to create and estimate state space models that allow for mixed-frequency data.

For an introduction to mixed-frequency state space modeling see "A Practitioner's Guide and Matlab Toolbox for Mixed Frequency State Space Models" by Scott Brave, Andrew Butters, and David Kelley.

Installation
------------
The easiest way to use MFSS is to install it via the toolbox (MFSS.mltbx) by dragging it into the Matlab command window. It should install as an Add-On and be available on your path whenever using Matlab. 

MFSS requires the following Matlab toolboxes: 
- Optimization Toolbox
- Statistics Toolbox
Further functionality is available through the use of additional toolboxes: 
- Global Optimization Toolbox
- Symbolic Toolbox
- Parallel Computnig Toolbox
- Econometrics Toolbox

Building the mex files
----------------------
To build, a mex compiler is required. The required library (armadillo) is included in the mex directory. A complete build of the mex files should be done by running mex/make.m. 

Before running make.m, run 

	mex -setup

to set the compiler preferences. 


Building the Toolbox
---------------------
To build the toolbox (MFSS.tlbx), run build/make_toolbox.m.

Using the make_toolbox.m script will update the minor version number of the toolbox.  It will producea toolbox file which should be installed in other copies of Matlab by dragging it to the command window. 

When updating the installed version of the toolbox, previous versions should be
uninstalled first throug the Matlab Add-On Manager to avoid name conflicts. 

Credits
--------
This library uses [Armadillo](https://www.google.com)	