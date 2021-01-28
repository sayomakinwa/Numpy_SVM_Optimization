from cvxopt import matrix, solvers
import time
import numpy as np

class SVM_Multi(object):
    """
    This class is implements SVM
    """
    def __init__(self, params):

        self.gamma = params.gamma
        self.C = params.C
        self.kernel_method = params.kernel_method # rbf | linear | poly
        self.alpha_val = None
        self.b = None

        self.params = params


    def obj_function(self):
        obj_vals = []
        for idx, class_i in enumerate(self.model_alphas):
            alpha_val = self.model_alphas[class_i]
            #x_train = self.model_x_train[class_i]
            y_train = self.model_y_train[class_i]
            b = self.model_b[class_i]
            Q = self.model_Q[class_i]

            obj_vals.append( (0.5 * (alpha_val.T @ Q @ alpha_val) - (-np.ones((len(alpha_val), 1))).T @ alpha_val)[0,0])
        
        return obj_vals


    def kernel(self, x1, x2):
        if self.kernel_method == "rbf":
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2)**2)
        elif  self.kernel_method == "poly":
            return ((x1 @ x2) + 1) ** self.gamma


    def compute_Q(self, X, y):
        P = y.shape[0]
        Q = np.zeros((P, P))
        for i in range(P):
            for j in range(i, P): # Since the matrix is symmetric, we don't compute the whole thing
                Q[i, j] = y[i] * y[j] * self.kernel(X[i], X[j])
                Q[j, i] = np.copy(Q[i, j]) 
        return Q


    def compute_b(self, X, y):
        b = 0
        b_count = 0
        for h in range(self.alpha_val.shape[0]):
            if self.alpha_val[h] < self.C:
                sub_sum = 0
                for i in range(self.alpha_val.shape[0]):
                    sub_sum += self.alpha_val[i] * y[i] * self.kernel(X[i], X[h])
                
                b += y[h] - sub_sum
                b_count += 1

        b = b/b_count
        return b


    def KKT_violation(self, Q, alpha, y, C):
        e = np.ones((y.shape[0], 1))
        f_grad = (Q @ alpha) - e 
        
        m_R_set, M_S_set = [], []

        for i in range(y.shape[0]):
            if alpha[i] < C:
                if y[i] > 0:
                    m_R_set.append( -(f_grad[i] * y[i]) )
                else:
                    M_S_set.append( -(f_grad[i] * y[i]) )
            elif alpha[i] > 0:
                if y[i] < 0:
                    m_R_set.append( -(f_grad[i] * y[i]) )
                else:
                    M_S_set.append( -(f_grad[i] * y[i]) )
        
        return np.max(m_R_set) - np.min(M_S_set)


    def fit(self, X, y, verbose=False):

        self.x_train, self.y_train= X, y

        # Preparing the input to the cvxopt solvers.qp function
        if verbose:
            print("Setting up the optimization problem: Initializing Q, P, A, b, G and h...")
        n_samples = X.shape[0]
        
        self.Q = self.compute_Q(X, y)
        Q = matrix(self.Q, tc="d")  # shape is (n, n)

        P = matrix(-np.ones((n_samples, 1)), tc="d")  # -e shape is (n, 1)
        
        # equality constraints
        A = matrix(y.reshape((1, -1)), tc="d")  # shape is (1, n)
        b = matrix([0.0], tc="d")  # shape is (1, 1)
        
        # 2 inequality constraints for each sample; -lambda <= 0 and lambda <= C
        G = matrix( np.concatenate( (-np.eye(n_samples), np.eye(n_samples)) ), tc="d" )  # shape is (n*2, n)
        h = matrix( np.concatenate( ( np.zeros((n_samples, 1)), np.zeros((n_samples, 1)) + self.C ) ), tc="d" )  # shape is (2*n, 1), since there are two constraints
        
        solvers.options['show_progress'] = False
        
        # Starting the optimization function...
        if verbose:
            print("Starting optimization...")
        start_time = time.time()
        sol = solvers.qp(Q, P, G, h, A, b)
        total_time = time.time() - start_time
        if verbose:
            print("Optimization complete...")
            print("Computing b and KKT violation...")

        self.alpha_val = np.array(sol['x'])
        self.b = self.compute_b(X, y)

        kkt_viol = self.KKT_violation(Q, self.alpha_val, y, self.C)

        return sol, np.round(total_time, 2), kkt_viol


    def fit_one_v_all(self, X, y, y_classes=[3, 6, 8], verbose=True):
        self.model_alphas, self.model_b, self.model_y_train, self.model_Q = {}, {}, {}, {}
        
        total_time = 0.0
        #train each class
        solutions = []
        all_kkt_viol = []
        for class_i in y_classes:
            if verbose:
                print("Training for class {}".format(class_i))
            
            y_train_i = (((y != class_i) * -2) + 1)
            sol, time_spent, kkt_viol = self.fit(X, y_train_i)
            
            self.model_alphas[class_i] = np.copy(self.alpha_val) 
            self.model_b[class_i] = np.copy(self.b)
            #self.model_x_train[class_i] = np.copy(self.b)
            self.model_y_train[class_i] = np.copy(y_train_i)
            self.model_Q[class_i] = np.copy(self.Q)

            solutions.append(sol)
            total_time += time_spent
            all_kkt_viol.append(kkt_viol)
        
        return solutions, total_time, all_kkt_viol
    

    def predict_one_v_all(self, x_test):
        print("Running predictions for the three models. It'll be a moment...")
        y_pred = np.zeros((x_test.shape[0], len(self.model_alphas)))
        #class_y_pred = {}
        classes = np.zeros(len(self.model_alphas))
        for idx, class_i in enumerate(self.model_alphas):
            classes[idx] = class_i
            #class_y_pred[class_i] = 
            
            alpha_val = self.model_alphas[class_i]
            #x_train = self.model_x_train[class_i]
            y_train = self.model_y_train[class_i]
            b = self.model_b[class_i]

            for j in range(x_test.shape[0]):
                sub_sum = 0
                for i in range(self.x_train.shape[0]):
                    sub_sum += alpha_val[i] * y_train[i] * self.kernel(self.x_train[i], x_test[j])
                
                #y_pred[j] = np.sign(sub_sum + self.b)
                y_pred[j, idx] = sub_sum + b
        
        y_pred_ = [classes[idx] for idx in y_pred.argmax(1)]

        return y_pred_

