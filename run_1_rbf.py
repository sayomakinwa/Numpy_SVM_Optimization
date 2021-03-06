import sys, os
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path[0:-1])
from utils import *
from model.svm_code import *


if __name__ ==  "__main__":
	params.gamma = 0.001
	params.C = 3
	params.kernel_method = "rbf"
	svm = SVM(params)
	sol, opt_time, kkt_viol = svm.fit(x_train38, y_train38)

	print("C: {}".format(svm.C))
	print("Gamma: {}".format(svm.gamma))
	print("Train accuracy: {}".format(np.sum(svm.predict(x_train38) == y_train38) / y_train38.shape[0]))
	y_pred = svm.predict(x_test38)
	print("Test accuracy: {}".format(np.sum(y_pred == y_test38) / y_test38.shape[0]))
	print("Final obj func val: {}".format(svm.obj_function()[0,0]))
	print("No of func eval: {}".format(sol["iterations"]))
	print("KKT Violation: {}".format(kkt_viol))
	print("Status: {} Optimal".format("NOT" if kkt_viol > 0 else ""))
	print("CPU Time: {}".format(opt_time))

	cm = confusion_matrix(y_test38, y_pred)
	print("Test Confusion Matrix: \n{}".format(cm))

	df_cm = pd.DataFrame(cm, [3,8], [3,8])
	sn.heatmap(df_cm, annot=True)
	plt.show()