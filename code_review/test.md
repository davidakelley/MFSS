# General Comments

* How do you use this folder. Consider adding it to the readme

Make sure that files are consistently styles:
* 80 character line lengths
* Line break after the docstring with between your name, year (PULL)
* Make sure tests are briefly commented
	* I think the comments in Accumulator_IntegrationTest are the bare minimum
	* The comments in Accumulator_test were much better
	* Use the one of the above two as models for comments in the rest where
	  missing
* Make a line somewhere, either in the doc string or in the readme to refer to
  the references in the paper for any references in the code. That is, you
  refer to Durbin & Koopman (2012) in estimate_test. Either mention that
  references should be check in the "A Practitioner's Guide..." here or refer
  this in the readme.

# execTests.m

* Rewrite lines 25-33 to reduce reperition in code. Consider using a cellfun on
  TestSuite.fromFile over cell arrays <blank>Tests. (PULL)
* Consider reformatting the docstring to have an "Inputs" section which details
  tests. Maybe something like the following (PULL)

```matlab
% Inputs:
% tests ~ cell array of character vectors (optional)
% 	Contains the shortcuts to run subsets of the tests. Valid options are
%   'basic', 'kalman', 'Accumulator', 'ml', and 'decomp'. Defaults to {'basic',
%	'kalman', 'Accumulator', 'ml', 'decomp'}
```

# ThetaMap_test.m

* Comment thetaSystemThetaSymbolComplex
* Bill is references. Is he referenced anywhere else? Consider putting him in
  the credits in the readme (maybe with his association).

# mfvar_test.m

* Delete lines 66 and 67 if not needed, as well as comment on 68 (PULL)

# kalman_test

* These comments can be improved, see general comments for which files I
  thought can be used as guidelines

# AbstractSystem_test

* There are no test cases? Is this file necessary
