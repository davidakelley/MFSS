# General comments

* Is there a general order to take the examples in. Easy, medium, hard?
* You should reference where the data is coming from. That is, what is
  durbin_koopman.mat? Is this something you made to replicate durbin_koopman or
  is this directly from their website.
	* Where is the data_gdp coming from - haver? This might be against our
	  haver policy - though no one will know
	* Still a few lines of where the data is coming from is good for
	  replication or for updating
* Give a brief description or table of contents for the examples. A compilation
  of their docstrings will suffice - put this in the readme
* consider putting examples in their own namespace which allow them to be
  easily accessible after the package is is installed
	* something like +mfss_examples
	* Examples are referencing local datasets, would need to figure out the path issue

* I would generally like the figures to be in a figures folder, the data to be
  in a data folder, but this isn't necessary
	* In particular, in can cause problems with path issues (maybe not)
	* If examples are in the addons folder and are namespaced, then exporting
	  of figures would be unadvised - they will drop in the current folder that
	  the user is in. Consider thinking about where the best placement of these
	  is

# nile_llm

* Percent Contribution to "Estimated" (PULL)
	* Line 40
	* spelled wrong
	* "Percent Contribution to Estiamted Trend by Observation"
* This should probably be the starting point
* Describe the StateSpaceEstimation Progress window (PULL)
	* What are the main features of it
	* Consider moving the titles up some pixels for "Current Theta Value" and
	  "Current Log-likelihood: <log likelihood>". The Current Log-... is
	  overlapping with the 10^-4 which is aesthetically unpleasing. Move the
	  "Current Theta Value to be consistent"

# gdp_arma21.m

* Document what line 5 is doing
	* This begs the question of what you want the examples to do
	* Should they be self-explanatory or do we want the user to dig a little
	* Do we want to user to let them try the examples with their own data? If
	  so, we should document the Get Data section with where the data is coming
	  from and why we are slicing in the way line 5 is doing. 
	* Staring at it for a moment for an experienced matlab user should be
	  enough to understand what is going on - but in this case a few words
	  would be nice
* What is this plot, label it with all the niceties a plot should have.

# generateARmodel.m

* What is the usage for this function

# generateData.m

* What is the usage for this function

# pgmtmfss_replication.m

* line 7 is not needed (PULL)
* Line 8 suffices
* Give a warning that this might take a while... (PULL)

# pgmtmfss1_dfm.m

* This has the exact level of documentation on the data that the other examples 
  should have
	* Consider putting also the Fred codes but not necessary
	* Consider putting this documentation in a readme in the examples folder.
	  Then you can refer to the readme whenever yo uneed data like this
* Do a see also pgmtmfss_replication (PULL) 

# General Comments on pgmtmfss replication files

* These are generally commented well

* Why are there two Code Example 2s (PULL)
	* see pgmtmfss3 should presumably be Code Example 3

* Make the docstrings more consistent (PULL)
	* I like how pgmtmfss3_trend defines what y and dates are in that format
	* But I much prefer the detail given, for example, in pgmtmfss1_dfm
	* Combine the both for good consistent docstring

* Title all the figures (PULL)

# pgmtmfss4_r_star

* What does this docstring mean. Link me to the code
* Be more specific
* Where are the calibrated ratios coming from

# pgmtmfss4_r_star_monthly

* Throw a see also for pgmtmfss4_r_star.m (PULL)
* People will be confused otherwise - they can't run this directly

# pgmtmfss4_r_star_quarterly

* Throw a see also for pgmtmfss4_r_star.m
* People will be confused otherwise - they can't run this directly
