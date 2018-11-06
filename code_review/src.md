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

* line 23:spelled wrong
* Do you want 29 to ever change

# AbstractSystem

* See my comments on Hidden function docstrings

# Accumulator

* Comments for Accumulator function is consistent with how you've commented the
  other constructors
	* They should be consistent throughout
	* See StateSpace and AbstractSystem for how you have commented other
	  functions

* Delete lines 54, 56-57
* Line 280-281 is phrased strangely
* Line 402 (FIXME)
* Line 534 (FIXME)
* line 594 (TODO)
* Line 700 (FIXME)
* line 738 unecessary comment
* Line 892 comment is strange? Was this a note to yourself or a note about the
  code

* Almost all of the hidden functions need more documentation in their
  docstrings. See my comments on the StateSpace file for my thoughts on where
  it is appropriate
	* Very simple functions do no need comments or expansive docstrings like
	  composeLinearFunc or createTransforms. 

# EstimationProgress

* This is good

# MFVAR

* Is this because a VAR can't be put into a state space?
* This seems like a strange class - or maybe I'm being dumb?

# StateSpace

* line 58-59 should be deleted
* line 224

* Consider providing more documentation in docstring for prepareFilter 
	* Best: Same level of documentation as other functions
	* Beter: All of below and describe inputs and outputs
	* Good: All of below and input and output types
	* Okay: Below and  where it gets used (See Also).
	  Second is particularly useful for if it is used in other code outside of
	  this file.
	* Minimum: One liner

* Consider providing more documentation in docstring for gradFinDiff
	* See above for my taxonomy
* Should provide commenents for the first for loop block 647:669
* Remove line 697 (blank line)

* Consider providing more documentation in docstring for getInputParameters
	* Consider answering the question of why this is used - is this just a
	  helper function to make code cleaner. If it, denote this by calling a
	  "helper function".

* Consider providing more documentation in docstring for filter_m
* line 789: "Initialize - Using the FRBC timing"
	* are you going to keep this reference?
	* Is this kosher?

* Remove 891:892

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

* Consider providing more documentation in docstring for smoother_m
	* Consider also providing a few more comments in this function
	* You commented in filter_m for the "Preallocate" section so I think you
	  should do the same here
	* lines 950-959 could do with a comment, or make a comment describing the
	  whole for loop started in 938
	* delete line 1006
	* 1002-1033 could do with some more comments

* Consider providing more documentation in docstring for smoother_mex
	* I think this one liner can be improved as well
	* See filter_mex for some thoughts on what can be put here
	* The docstring should be consistent with filter_mex's level of detail and
	  phrasing

* On filter_weights
	* Are comments on 1167-1230 appropriately detailed?

* On smoother_weights
	* Are comments on 1292-1400 appropriately detailed?
	* Do we need more comments on 1405-1442 or does 1402 suffice

* Consider providing more documentation in docstring for r_weights

* Consider providing more documentation in docstring for r_weight_recursion

* Consider providing more documentation in docstring for
  build_smoother_weight_parts
	* Lines 1666-1735 could do with some more comments

* Consider providing more documentation in docstring for build_M0ti
	* This function needs more comments

* Consider providing more documentation in dosctring for build_Ldagger
	* This function needs more comments

# StateSpaceEstimation

* lines 266-349 don't have comments
* Consider commenting on the raised exception in the docstring
* See my comments on Hidden function docstring

# ThetaMap

* The construct function needs more documentation in the docstring

* Thetamap estimation needs some more comments
	* Lines 215 to 236

* Hidden function docstrings. See above comments (particularly in StateSpace)
  for my thoughts

* IndexStateSpace does not use the same terminology in the docstring as other
  functions. Inputs/Outputs should be changed to Arguments: and Returns: to be
  consistent with other docstrings in other files
	* This is true of a lot of functions in this file
	* Be sure to be consistent with the colon (:) after the section name

* isequalTransform docstring needs some work - consider removing 1264 since it
  applies to the whole file 
