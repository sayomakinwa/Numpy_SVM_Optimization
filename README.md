# Numpy SVM Impementation

Implementation of Support Vector Machines with numpy and optimization with KKT conditions.

## The Dataset

Blah blah blah


## The code

Each `run_#.py` file is an implementation for the task described in the list below (`#` corressponds to each number):

1. `run_1.py` executes full minimization for the SVM dual quadratic problem using the cvxopt routine. To do a search over `C` and `gamma`, edit the search values in `run_grid_search.py` and run the script, then set the values in `run_1.py` and run it. The kernel used can also be changed between `polynomial` and `rbf`.

2. `run_2.py` implements a decomposition method to solve the SVM dual quadratic problem, with q set to 100 (this can be changed). Use the same `C` and `gamma` values from the grid search before.

3. `run_3.py` implements the Most Violating Pair (MVP) decomposition method to solve the SVM dual quadratic problem, with q set to 2, using the analytic solution of the subproblems. Use the same `C` and `gamma` values from the grid search before.

4. `run_4.py` executes a One-vs-All multi-class SVM over the dataset, for three classes (digits 3, 6, 8), executing full minimization for the SVM dual quadratic problem for each model.