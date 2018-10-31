MFSS - Mixed-Frequency State Space Modeling
===========================================

Estimation and inference on state space models that allow for mixed-frequency time series data.

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
- Parallel Computing Toolbox

Some examples and tests additionally use the Econometrics Toolbox. 

Compatibility
-------------
MFSS has been tested on Matlab 2017b-2018b. 

It is incompatible with Octave (through at least 4.4.1). 

Building the mex files
----------------------
Substantial performance improvements are possible through using compiled versions of the Kalman filter/smoother functions. 

To build, a mex C/C++ compiler is required. The required library (Armadillo) is included in the mex directory. A complete build of the mex files should be done by running mex/make.m. 

Before running make.m, set up the mex compiler by following the instructions [here](https://www.mathworks.com/help/matlab/matlab_external/choose-c-or-c-compilers.html). 

Building the Toolbox
---------------------
To build the toolbox (MFSS.mltbx), run `build/make_toolbox.m`. This process is sensitive to the paths available when running this file. The top level directory and `/src` should be the only directories on the path when running make_toolbox

Using the `make_toolbox.m` script will update the minor version number of the toolbox. It will produce a toolbox file which can be installed in other copies of Matlab by dragging it to the command window. 

When updating the installed version of the toolbox, previous versions should be uninstalled first through the Matlab Add-On Manager to avoid name conflicts. 

Credits
--------
This library uses [Armadillo](http://arma.sourceforge.net/).