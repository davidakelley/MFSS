MFSS - Mixed-Frequency State Space Modeling
===========================================

MFSS provides functions to create and estimate state space models that allow for mixed-frequency data.

For user documentation, see the /doc directory.

Building the mex files
----------------------
To build, a mex compiler (likely Visual Studio 2017) is required. The 
required library (armadillo) is included in the mex directory.

A complete build of the mex files should be done by running mex/make.m. 

Before running make.m, run 

	mex -setup

to set the compiler preferences. 


Building the Toolbox
---------------------
To build the toolbox (MFSS.tlbx), run build/make_toolbox.m.

In the process of building the toolbox, the HTML documentation will be regenerated. 
The documentation generation (see build/compile_docs.m) required a python conda 
environment named py27 to be installed. For more information, see compile_docs.m.

Using the make_toolbox.m script will update the minor version number of the toolbox. 
It will producea toolbox file which should be installed in other copies of Matlab by
dragging it to the command window. 

When updating the installed version of the toolbox, previous versions should be
uninstalled first throug the Matlab Add-On Manager to avoid name conflicts. 