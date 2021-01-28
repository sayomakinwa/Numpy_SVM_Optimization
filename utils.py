from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
import os, sys, gzip, itertools, random, time


SEED = 1858908
random.seed(SEED)
np.random.seed(SEED)


def load_mnist(path, kind='train'):
    """
    @author: Diego 
    """
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    """
    We are only interested in the items with label 2, 4 and 6.
    Only a subset of 1000 samples per class will be used.
    """
    indexLabel3 = np.where((labels==3))
    xLabel3 =  images[indexLabel3][:1000,:].astype('float64')
    yLabel3 = labels[indexLabel3][:1000].astype('float64')

    indexLabel8 = np.where((labels==8))
    xLabel8 =  images[indexLabel8][:1000,:].astype('float64')
    yLabel8 = labels[indexLabel8][:1000].astype('float64')

    indexLabel6 = np.where((labels==6))
    xLabel6 =  images[indexLabel6][:1000,:].astype('float64')
    yLabel6 = labels[indexLabel6][:1000].astype('float64')

    # =====================================================================

    x_label_368 = np.vstack((xLabel3, xLabel6, xLabel8))
    y_label_368 = np.concatenate((yLabel3, yLabel6, yLabel8))
    
    # converting labels of classes 3 and 8 into +1 and -1
    yLabel3 = yLabel3 / 3.0
    yLabel8 = yLabel8 / -8.0

    x_label_38 = np.vstack((xLabel3, xLabel8))
    y_label_38 = np.concatenate((yLabel3, yLabel8))

    scaler = StandardScaler()
    scaler.fit(x_label_38)
    x_label_38 = scaler.transform(x_label_38)

    scaler = StandardScaler()
    scaler.fit(x_label_368)
    x_label_368 = scaler.transform(x_label_368)

    x_train38, x_test38, y_train38, y_test38 = train_test_split(x_label_38, y_label_38, test_size=0.2, random_state=SEED)
    x_train368, x_test368, y_train368, y_test368 = train_test_split(x_label_368, y_label_368, test_size=0.2, random_state=SEED)

    return (x_train38, x_test38, y_train38, y_test38), (x_train368, x_test368, y_train368, y_test368)



def grid_search(params, model_class, params_attr, params_vals, kfold_split, x_train, y_train):
    """
    Runs a grid saearch over the supplied parameters
    Args:
        params: object of the Params class. It is expected to already hold at least values that are not to be searched 
        params_attr: list of attribute names to be searched. e.g. ["hidden_size", "rho"]
        params_vals: list of lists of values for params_attr, each sublist holding the values to be searched for each attribute. e.g. [[2, 3, 4, 5], [1e-3, 1e-4, 1e-5]] 
        x_train: the input features of the train dataset
        y_train: the output of the train dataset
    Returns:
        best_result: a dictionary of best result from the search
    """
    best_result = {
        "val_acc": 0,
        "params": params
    }
    combos_len = 1
    for val in params_vals:
        combos_len *= len(val)
    
    # all the possible combinations of the values of all the parameters
    combos = itertools.product(*params_vals) 
    
    kf = KFold(n_splits=kfold_split, shuffle=True, random_state=SEED)
    result_list = list()
    
    print("Searching parameters...")
    for i, combo in enumerate(combos):
        for idx, val in enumerate(params_attr):
            # each combination is a tupule, so we loop to set the params object attribute values
            setattr(params, val, combo[idx])

        sol, opt_time, kkt_viol = [], [], []
        train_acc, val_acc = [], []

        print("{}/{}. Running K-Fold training with parameters: {}:{}".format(i+1, combos_len, params_attr, combo))
        for train_idx, val_idx in kf.split(x_train):
            k_x_train, k_x_val = x_train[train_idx], x_train[val_idx]
            k_y_train, k_y_val = y_train[train_idx], y_train[val_idx]
            
            # create a fresh model
            model = model_class(params)

            start = time.time()
            result = model.fit(k_x_train, k_y_train, verbose=False)
            time_taken = time.time() - start

            
            train_acc.append( np.sum(model.predict(k_x_train) == k_y_train) / k_y_train.shape[0] )
            val_acc.append( np.sum(model.predict(k_x_val) == k_y_val) / k_y_val.shape[0] )

            sol.append(result[0])
            opt_time.append(result[1])
            kkt_viol.append(result[2])

            #compute_time.append(time_taken)

        print("    Val Acc: {:.4f}".format(np.mean(val_acc)))
        print("=============================================\n")        

        if np.mean(val_acc) > best_result["val_acc"]:
            best_result = {
                "params": params_attr,
                "combo": combo,
                "train_acc": np.mean(train_acc),
                "val_acc": np.mean(val_acc),
                
                "status": result[0]["status"],
                "compute_time": np.mean(opt_time),
                
                "kkt_viol": np.mean(kkt_viol)
            }
        
        result_list.append({
            "params": params_attr,
                "combo": combo,
                "train_acc": np.mean(train_acc),
                "val_acc": np.mean(val_acc),
                
                "status": result[0]["status"],
                "compute_time": np.mean(opt_time),
                
                "kkt_viol": np.mean(kkt_viol)
        })
    
    print("Done searching!")
    print("Best result: \n {}".format(best_result))
    
    return result_list


class Params(object):
    # gamma = 0.01
    gamma = 3
    C = 2.5
    # kernel_method = "rbf"
    kernel_method = "poly"

params = Params()


dir_path = os.path.dirname(os.path.realpath(__file__))
path = dir_path+"/data/"
task1, bonus_task_data = load_mnist(path)
x_train38, x_test38, y_train38, y_test38 = task1