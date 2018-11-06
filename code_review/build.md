# General comments

* I think the organization here can be improved
	* Move compile_docs.m to the docs folder
	* Move make_toolbox to the root of the directory
	* That's what people want if they clone this directory
	* Don't put this in subfolder

# make_toolbox.m

* You should put this doc string in a contributing section of the readme
	* most people are not contributing
	* most people are not committing their changes 
	* Keep this seperate to cater for these people
	* "Add the /src and /build paths" should be part of the readme
	* Provide a few lines of code in the readme to do this
	* In which way is the build script sensitive to other paths?
		* I think it's fine to say this to make sure it works
		* If it's not necessary though, or you can work around it, you should
		  take it out
* This is not working ...

# compile_docs.m

* This should be documented in the readme
* The requirements for this are way to specific
	* "A conda environment named py27" is not okay!!
	* You should be able to work on any python 2.7 distribution with sphinx and
	  shpincontrib-matlabdomain installed
	* Why is python 2.7 required? - there are no python files... Sphinx can be
	  pip installed on either python 2.7 or 3.x. 
	* If python version is not necessary, you can say you tested this on 2.7 so
	  compatability with 3.x is not guaranteed.
	* Do they need anything about conda to actually make this or just sphinx.
	  Almost certainly they do not need anything with conda.
	* Consider having the matlab function simply check for python and the
	  appropriate python packages
* You provided the make.bat and makefile so that it is compatible with any
  install right? This is good! If they are on windows, they will auto use the
  .bat file. If they are on linux, it will come with make.
	* Do 5 minutes of research to see if mac OS x comes with make preinstalled.
	  Otherwise this will be a problem
