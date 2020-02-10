##### IMPORTANT NOTES BEFORE USING #####
1. The codes in this project are written in Python 2.7
2. The following files are needed:
   - domain_functions.py
   - system_functions.py
   - project_trajectories.py
   - final_project_hybridsystems.py (main project file)
3. File usages are described in comments at top of each file.

##### USAGE INSTRUCTIONS #####

- - - USAGE 1: Same nodes, LTL formula (EASIEST) - - -

In this case, keep the same node definitions (lines 141-142 in final_project_hybridsystems.py).
NOTE: If you run the code "final_project_hybridsystems.py" as provided, it will reproduce the plots from the project report.

You can also try other initial conditions (line 204), for which a new output of sol_T_states will be produced. You can then select a row of this yourself by setting a new value to sol_idx (row index number) on line 299.

- - - Usage 2: Same nodes, different LTL formula - - -

If desired, in "final_project_hybridsystems.py", set new transition booleans corresponding to a new Buchi automaton produced by a different LTL formula. You have to re-specify the Buchi automaton conditions in SECTION 2 (lines 158 - 207), subject to the instructions that are commented throughout.

- - - Usage 3: Different nodes, observations, etc - - -

In this case, it would be best to define nodes as desired on line 192 in "domain_functions.py" and examine the form of the resulting system. Be sure to un-comment line ~293 to do the unit test of "domain_functions.py" if you are experimenting. From the optionally displayed domain data, choose which simplices you want to make into S, K, and R regions. 

This will be more involved, necessitating some trial and error to design a new system, but the code base permits such modifications. Once the desired form of the nodes and new labeling of the simplices is known, this information can be updated in "final_project_hybridsystems.py" along with perhaps new Buchi automaton from a new LTL formula as well.
