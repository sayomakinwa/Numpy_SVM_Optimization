from cvxopt import matrix, solvers
import time
import numpy as np

class SVM(object):
    """
    This class implements SVM
    """
    def __init__(self, params):

        self.gamma = params.gamma
        self.C = params.C
        self.kernel_method = params.kernel_method # rbf | linear | poly
        self.alpha_val = None
        self.b = None

        self.params = params 


    def obj_function(self):
        return (0.5 * (self.alpha_val.T @ self.Q @ self.alpha_val) - (-np.ones((len(self.alpha_val), 1))).T @ self.alpha_val)


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


    def predict(self, x_test):
        y_pred = np.zeros((x_test.shape[0]))

        for j in range(x_test.shape[0]):
            sub_sum = 0
            for i in range(self.x_train.shape[0]):
                sub_sum += self.alpha_val[i] * self.y_train[i] * self.kernel(self.x_train[i], x_test[j])
            
            y_pred[j] = np.sign(sub_sum + self.b)

        return y_pred


    def KKT_violation(self, Q, alpha, y, C):
        e = np.ones((y.shape[0], 1))
        f_grad = (Q @ alpha) - e 
        
        m_R_set, M_S_set = [], []

        for i in range(y.shape[0]):
            if alpha[i] < C:
                if y[i] > 0:
                    m_R_set.append( -f_grad[i] * y[i] )
                else:
                    M_S_set.append( -f_grad[i] * y[i] )
            elif alpha[i] > 0:
                if y[i] < 0:
                    m_R_set.append( -f_grad[i] * y[i] )
                else:
                    M_S_set.append( -f_grad[i] * y[i] )
        
        return np.max(m_R_set) - np.min(M_S_set)


    def fit(self, X, y, verbose=True):

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


    def fit_decomposition(self, X, y, q=100, verbose=True):
        self.x_train, self.y_train = X, y

        # Preparing the input to the cvxopt solvers.qp function
        n_samples = X.shape[0]
        
        if verbose:
            print("Setting up the optimization problem...")
        
        self.Q = self.compute_Q(X, y)
        Q = matrix(self.Q, tc="d")  # shape is (n, n)
        
        # Starting the optimization function...
        if verbose:
            print("Starting optimization...")
        start_time = time.time()
        ############################################################################
        alpha_k = np.zeros(y.shape[0])

        f_grad = -np.ones(y.shape[0])
        f_grad_y = -f_grad * y

        R_id_set, S_id_set = [], []

        for i in range(y.shape[0]):
            if alpha_k[i] < self.C:
                if y[i] > 0:
                    R_id_set.append(i)
                else:
                    S_id_set.append(i)
            elif alpha_k[i] > 0:
                if y[i] < 0:
                    R_id_set.append(i)
                else:
                    S_id_set.append(i)
        
        m_R = np.max(f_grad_y[R_id_set])
        M_S = np.min(f_grad_y[S_id_set])

        iter = 0
        solutions = []
        #eps = 0.01
        eps = 0.0015
        while (m_R - M_S > eps ):
            #print(m_R - M_S)
            # Selecting working set using the rule of SVM-light
            k_end = int(q/2)

            k_idx1 = np.array(R_id_set)[ (-f_grad_y[R_id_set]).argsort()[0:k_end] ].tolist()
            k_idx2 = np.array(S_id_set)[ f_grad_y[S_id_set].argsort()[0:k_end] ].tolist()

            working_set = k_idx1 + k_idx2
            k_idx = working_set

            d = np.array([ 
                y[k_idx1], # i from R 
                -y[k_idx2] # j from S
            ]).reshape((len(working_set), 1))
            
            # Exact Linesearch algorithm
            # t_star
            denom = d.T @ np.array(Q[working_set, working_set]) @ d
            
            t_star = 0.0
            if denom > 0:
                t_star = (-( f_grad[working_set].reshape((len(working_set), 1)).T @ d ) / denom)[0,0]
            elif denom == 0:
                t_star = np.inf

            # t_star_fea
            a, b = [], []
            
            for idx, dki in enumerate(d.flatten()):
                if dki > 0:
                    a.append( (self.C - alpha_k[working_set][idx]) / d.flatten()[idx] )
                if dki < 0:
                    b.append( alpha_k[working_set][idx] / np.abs(d.flatten()[idx]) )
            
            t_star_fea = np.min( [np.min(a+[np.inf]), np.min(b+[np.inf])] )
            
            t_k = t_star if t_star <= t_star_fea else t_star_fea

            # alpha k+1
            alpha_k_1 = alpha_k[working_set] + t_k*d.flatten()

            # Updating f_grad
            f_grad[working_set] += ( np.array(Q[working_set, working_set]) @ ( alpha_k_1 - alpha_k[working_set] ).reshape((len(working_set), 1)) ).flatten()
            f_grad_y[working_set] = -(f_grad[working_set] * y[working_set])
            
            alpha_k[k_idx] = np.copy(alpha_k_1.flatten())

            # Recalculate m and M
            R_id_set, S_id_set = [], []
            for i in range(y.shape[0]):
                if alpha_k[i] < self.C:
                    if y[i] > 0:
                        R_id_set.append(i)
                    else:
                        S_id_set.append(i)
                elif alpha_k[i] > 0:
                    if y[i] < 0:
                        R_id_set.append(i)
                    else:
                        S_id_set.append(i)
            
            m_R = np.max(f_grad_y[R_id_set])
            M_S = np.min(f_grad_y[S_id_set])

            iter += 1
            
        ############################################################################
        total_time = time.time() - start_time
        if verbose:
            print("Optimization complete...")
            print("Computing b and KKT violation...")


        self.alpha_val = alpha_k
        self.b = self.compute_b(X, y)

        #kkt_viol = self.KKT_violation(Q, self.alpha_val, y, self.C)
        
        return np.round(total_time, 2), m_R - M_S, iter


    def fit_MVP(self, X, y, verbose=True):
        self.x_train, self.y_train = X, y

        # Preparing the input to the cvxopt solvers.qp function
        if verbose:
            print("Setting up the optimization problem: Initializing Q...")
        n_samples = X.shape[0]
        
        self.Q = self.compute_Q(X, y)
        Q = matrix(self.Q, tc="d")  # shape is (n, n)
        
        # Starting the optimization function...
        if verbose:
            print("Starting optimization...")
        start_time = time.time()
        ############################################################################
        alpha_k = np.zeros(y.shape[0])

        f_grad = -np.ones(y.shape[0]) # -e
        f_grad_y = -(f_grad * y)

        R_id_set, S_id_set = [], []

        for i in range(y.shape[0]):
            if alpha_k[i] < self.C:
                if y[i] > 0:
                    R_id_set.append(i)
                else:
                    S_id_set.append(i)
            elif alpha_k[i] > 0:
                if y[i] < 0:
                    R_id_set.append(i)
                else:
                    S_id_set.append(i)
        
        m_R = np.max(f_grad_y[R_id_set])
        M_S = np.min(f_grad_y[S_id_set])

        iter = 0
        eps = 1e-10
        while (m_R - M_S > eps):
            #print(m_R - M_S)
            # Selecting working set
            working_set = [
                R_id_set[np.argmax(f_grad_y[R_id_set])], # i MVP from R 
                S_id_set[np.argmin(f_grad_y[S_id_set])]  # j MVP from S
            ]
            
            d = np.array([ 
                y[working_set[0]], # i from R 
                -y[working_set[1]] # j from S
            ]).reshape((2, 1))
            
            # Exact Linesearch algorithm
            # t_star
            denom = d.T @ np.array(Q[working_set, working_set]) @ d
            
            t_star = 0.0
            if denom > 0:
                t_star = (-( f_grad[working_set].reshape((2, 1)).T @ d ) / denom)[0,0]
            elif denom == 0:
                t_star = np.inf

            # t_star_fea
            a, b = [], []
            
            for idx, dki in enumerate(d.flatten()):
                if dki > 0:
                    a.append( (self.C - alpha_k[working_set][idx]) / d.flatten()[idx] )
                if dki < 0:
                    b.append( alpha_k[working_set][idx] / np.abs(d.flatten()[idx]) )
            
            t_star_fea = np.min( [np.min(a+[np.inf]), np.min(b+[np.inf])] )
            
            t_k = t_star if t_star <= t_star_fea else t_star_fea

            # alpha k+1
            alpha_k_1 = alpha_k[working_set] + t_k*d.flatten()
            
            # Updating f_grad
            f_grad[working_set] += ( np.array(Q[working_set, working_set]) @ ( alpha_k_1 - alpha_k[working_set] ).reshape((2, 1)) ).flatten()
            f_grad_y[working_set] = -(f_grad[working_set] * y[working_set])
            
            alpha_k[working_set] = alpha_k_1.flatten()

            # Re-calculate m and M
            R_id_set, S_id_set = [], []
            for i in range(y.shape[0]):
                if alpha_k[i] < self.C:
                    if y[i] > 0:
                        R_id_set.append(i)
                    else:
                        S_id_set.append(i)
                elif alpha_k[i] > 0:
                    if y[i] < 0:
                        R_id_set.append(i)
                    else:
                        S_id_set.append(i)
            
            m_R = np.max(f_grad_y[R_id_set])
            M_S = np.min(f_grad_y[S_id_set])
            iter += 1
            
        ############################################################################
        total_time = time.time() - start_time
        if verbose:
            print("Optimization complete...")
            print("Computing b and KKT violation...")

        self.alpha_val = alpha_k
        self.b = self.compute_b(X, y)

        #kkt_viol = self.KKT_violation(Q, self.alpha_val, y, self.C)
        
        return np.round(total_time, 2), m_R - M_S, iter

