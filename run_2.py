import sys, os
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path[0:-1])
from utils import *
from svm_code import *


params.gamma = 3
params.C = 2.5
params.kernel_method = "poly"
svm = SVM(params)
opt_time, kkt_viol, iter = svm.fit_decomposition(x_train38, y_train38, q=100)
(opt_time, kkt_viol)

print("C: {}".format(svm.C))
print("Gamma: {}".format(svm.gamma))
print("q: 100")
print("Train accuracy: {}".format(np.sum(svm.predict(x_train38) == y_train38) / y_train38.shape[0]))
y_pred = svm.predict(x_test38)
print("Test accuracy: {}".format(np.sum(y_pred == y_test38) / y_test38.shape[0]))
print("Final obj func val: {}".format(svm.obj_function()[0]))
print("No of iterations: {}".format(iter))
print("KKT Violation: {}".format(kkt_viol))
print("Status: {} Optimal".format("NOT" if kkt_viol > 0 else ""))
print("CPU Time: {}".format(opt_time))

cm = confusion_matrix(y_test38, y_pred)
print("Test Confusion Matrix: \n{}".format(cm))

df_cm = pd.DataFrame(cm, [3,8], [3,8])
sn.heatmap(df_cm, annot=True)
plt.show()