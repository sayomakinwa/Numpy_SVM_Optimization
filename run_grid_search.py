import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path[0:-1])
from utils import *
from model.svm_code import *


# RBF Kernel
params.kernel_method = "rbf"
params_attr = ["gamma", "C"]
params_vals = [[1e-3, 1e-4, 1e-5], [0.5, 1, 1.5, 2, 2.5, 3]]

result_list = grid_search(params, SVM, params_attr, params_vals, 5, x_train38, y_train38)


# Poly Kernel 
params.kernel_method = "poly"
params_attr = ["gamma", "C"]
params_vals = [[1, 2, 3, 4], [0.5, 1, 1.5, 2, 2.5, 3]]

result_list = grid_search(params, SVM, params_attr, params_vals, 5, x_train38, y_train38)