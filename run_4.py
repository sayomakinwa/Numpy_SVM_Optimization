import sys, os
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path[0:-1])
from utils import *
from svm_multi_code import *

x_train368, x_test368, y_train368, y_test368 = bonus_task_data

print("MULTI CLASS SVM: One against All. It'll be a moment...")
params.gamma = 3
params.C = 2.5
params.kernel_method = "poly"
svm = SVM_Multi(params)
solutions, total_time, all_kkt_viol = svm.fit_one_v_all(x_train368, y_train368)

print("C: {}".format(svm.C))
print("Gamma: {}".format(svm.gamma))
print("Train accuracy: {}".format(np.sum(svm.predict_one_v_all(x_train368) == y_train368) / y_train368.shape[0]))
y_pred = svm.predict_one_v_all(x_test368)
print("Test accuracy: {}".format(np.sum(y_pred == y_test368) / y_test368.shape[0]))
print("Final obj func val: {}".format(svm.obj_function()))
print("No of func evals for all models: {}".format([sol["iterations"] for sol in solutions]))
print("KKT violation for all models: {}".format(all_kkt_viol))
#print("Status: {} Optimal".format("NOT" if kkt_viol > 0 else ""))
print("Total CPU time for all training: {}".format(total_time))

cm = confusion_matrix(y_test368, y_pred)
print("Test Confusion Matrix: \n{}".format(cm))

df_cm = pd.DataFrame(cm, [3,6,8], [3,6,8])
sn.heatmap(df_cm, annot=True)
plt.show()