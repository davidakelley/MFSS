# General comments

* Do you have any raised errors / exceptions anywhere that are custom to this
  code. Consider documenting them in the docstring. Having good documentation
  for exceptions can immensely improve code usability!

* Docstring sections should be consistent
	* Decide on section titles - I think you have used Arguments: and Returns:
	  mostly throughout.
	* Make sure all of the section titles conform to what you decided above
	* Make sure they are formatted properly - use a colon (:) after the section
	  title

# AbstractStateSpace

# AbstractSystem

# Accumulator

# EstimationProgress

# MFVAR

# StateSpace

# StateSpaceEstimation

# ThetaMap

Other comments
==============
* Consider providing more documentation in docstring for filter_mex
	* Consider also providing a few more comments in this function (there is
	  only one)...
	* Consider improving the one liner as well. Say "Call mex function
	  filter_uni with appropriate arguments." or something similar
	* This is a helper function to help with the inputs and outputs of the mex
	  function so describe that in the docstring, either in the paragraph
	  description or in the "one liner" 
	* I think the term "Helper function" can be used because it has the
	  connotation that this is used to improve code readability rather than to
	  implement something novel

* Almost all of the hidden functions need more documentation in their
  docstrings. See my comments on the StateSpace file for my thoughts on where
  it is appropriate
	* Very simple functions do no need comments or expansive docstrings like
	  composeLinearFunc or createTransforms. 

* Consider providing more documentation in docstring for prepareFilter 
	* Best: Same level of documentation as other functions
	* Beter: All of below and describe inputs and outputs
	* Good: All of below and input and output types
	* Okay: Below and  where it gets used (See Also).
	  Second is particularly useful for if it is used in other code outside of
	  this file.
	* Minimum: One liner