import numpy as np
import time
from sklearn.decomposition import PCA

def compute_weights(C, weight_type='entropy'):
#     print('\nWeight type: {}'.format(weight_type))
    epsilon = 1e-6
    z = np.sum(np.abs(C)**2,axis=-1)**(1./2) 
    if weight_type.lower() == 'inverse':
#         print('   Update weights')
        weights = (z + epsilon)**-1
    elif weight_type.lower() == 'entropy':
#         print('   Update weights')
        z = np.abs(z + epsilon*np.random.randn(z.shape[0]))
        zSum = np.sum(z)
        zLog = np.log2(z)
        zLogSum   = np.sum(np.multiply(z, zLog))
        weights = - np.divide(zLog, zSum) + np.divide(zLogSum, zSum**2)
#         weights += 1e-6*np.random.randn(weights.shape[0])
#     print('   Done. Return weights {}.'.format(np.linalg.norm(weights)))
    return weights

class WeightedLassoADMM():
    def __init__(self, data, regularization_params=[2, 2], norm_type=1, weights=None, verbose=False, printing_step=5, 
                 TOLERANCE=[10**-8, -1], num_iters=5000, affine=False, initial_sol=None):
        self.data = data
        self.norm_type = norm_type
        self.verbose = verbose
        self.printing_step = printing_step
        self.TOLERANCE = TOLERANCE
        self.num_iters = num_iters
        self.affine = affine     
        self.regularization_params = regularization_params
        self.num_rows, self.num_columns = self.data.shape
        self.residuals = []
        if initial_sol is None:
            self.initial_sol = np.zeros([self.num_columns, self.num_columns])
        else:
            self.initial_sol = initial_sol
        if weights is None:
            self.weights = np.ones(self.num_rows)
        else:
            self.weights = weights
            
    def compute_lambda(self):
#         print ('Computing lambda...')

        _lambda = []
        T = np.zeros(self.num_columns)


        if not self.affine:

            T = np.linalg.norm(np.dot(self.data.T, self.data), axis=1)

        else:
            #affine transformation
            y_mean = np.mean(self.data, axis=1)

            tmp_mat = np.outer(y_mean, np.ones(self.num_columns)) - self.data

            T = np.linalg.norm(np.dot(self.data.T, tmp_mat),axis=1)

        _lambda = np.amax(T)

        return _lambda
    
    def compute_mean_error(self, Z, X):

        mean_error = np.sum(np.abs(Z-X)) / (np.shape(X)[0] * np.shape(X)[1])

        return mean_error
    
    def weighted_shrinkage_L1Lq(self, C, _lambda):
        # TODO: weighted shrinkage for self.norm_type==1 and self.norm_type='inf'
        num_rows, num_cols = C.shape
        row_sparse_C = []
        weights = np.expand_dims(self.weights, axis=1)
        if self.norm_type == 1:
            # soft thresholding            
            row_sparse_C = np.abs(C) - _lambda*np.matlib.repmat(weights, 1, num_cols)
            ind = row_sparse_C < 0
            row_sparse_C[ind] = 0
            row_sparse_C = np.multiply(row_sparse_C, np.sign(C))
        elif self.norm_type == 2:
            r = np.zeros([num_rows, 1])
            for i_row in range(0, num_rows):
                shrinkage_threshold = np.linalg.norm(C[i_row, :]) - _lambda*weights[i_row]
                r[i_row] = 0 if shrinkage_threshold < 0 else shrinkage_threshold
            row_sparse_C = np.multiply(np.matlib.repmat(np.divide(r, (r + _lambda*weights )), 1, num_cols), C)
        elif self.norm_type == 'inf':
            # TODO: write it
            print('To be implemented for inf norm')
        # elif self.norm_type == 2:
        #     print ''
        # elif self.norm_type == inf:
        #     print ''

        return row_sparse_C
    
    def run_weighted_lasso_admm(self):

        '''
        This function represents the Augumented Lagrangian Multipliers method for Lasso problem
        The lagrangian form of the Lasso can be expressed as following:

        MIN{ 1/2||Y-AZ||_2^2 + lambda||X||_1} s.t Z-X=0

        When applied to this problem, the ADMM updates take the form

        Z^t+1 = (AtA + rhoI)^-1(Aty + rho^t - mu^t)
        X^t+1 = Shrinkage_lambda/rho(Z(t+1) + mu(t)/rho)
        mu(t+1) = mu(t) + rho(Z(t+1) - Z(t+1))

        The algorithm involves a 'ridge regression' update for Z, a soft-thresholding (shrinkage) step for X and
        then a simple linear update for mu

        NB: Actually, this ADMM version contains several variations such as the using of two penalty parameters instead
        of just one of them (mu1, mu2)
        '''


#         print ('ADMM processing...')
        
        alpha1 = alpha2 = 0
        if (len(self.regularization_params) == 1):
            alpha1 = self.regularization_params[0]
            alpha2 = self.regularization_params[0]
        elif (len(self.regularization_params) == 2):
            alpha1 = self.regularization_params[0]
            alpha2 = self.regularization_params[1]

        #thresholds parameters for stopping criteria
        if (len(self.TOLERANCE) == 1):
            auxiliary_constraint_tolerance = self.TOLERANCE[0]
            affine_constraint_tolerance = self.TOLERANCE[0]
        elif (len(self.thr) == 2):
            auxiliary_constraint_tolerance = self.TOLERANCE[0]
            auxiliary_constraint_tolerance = self.TOLERANCE[1]

        # entry condition
        auxiliary_constraint_residual = 10 * auxiliary_constraint_tolerance    # || X - Z ||
        affine_constraint_residual = 10 * auxiliary_constraint_tolerance       # || 1^T X - 1^T ||

        start_time = time.time()

        # setting penalty parameters for the ALM
        mu1p = alpha1 * 1/self.compute_lambda()
#         print("-Compute Lambda- Time = %s seconds" % (time.time() - start_time))
        mu2p = alpha2 * 1

        mu1 = mu1p
        mu2 = mu2p

        i_iter = 1

        start_time = time.time()


        # defining penalty parameters e constraint to minimize, lambda and C matrix respectively
        X = self.initial_sol
#         print('\nInitial solution norm: {}'.format(np.linalg.norm(X)))
        
        # Lagrangian multiplier for "X - Z = 0" constraint
        LAMBDA2 = np.zeros([self.num_columns, self.num_columns])   
        
        P = self.data.T.dot(self.data)
        OP1 = np.multiply(P, mu1)

        if self.affine == True:

            # INITIALIZATION
            
            # Lagrangian multiplier for affine contraint (1^T)X = 1^T
            LAMBDA3 = np.zeros(self.num_columns).T 

            A = np.multiply(mu1,P) +  np.multiply(mu2, np.eye(self.num_columns, dtype=int)) +  np.multiply(mu2, np.ones([self.num_columns,self.num_columns]) )

            OP3 = np.multiply(mu2, np.ones([self.num_columns, self.num_columns]))

            while ( (auxiliary_constraint_residual > auxiliary_constraint_tolerance or affine_constraint_residual > auxiliary_constraint_tolerance) and i_iter < self.num_iters):
                # updating Z
                OP2 = np.multiply(X - np.divide(LAMBDA2,mu2), mu2)
                OP4 = np.matlib.repmat(LAMBDA3, self.num_columns, 1)
                
                Z = np.linalg.solve(A, OP1 + OP2 + OP3 + OP4)
                
                # updating C
                X = self.weighted_shrinkage_L1Lq(Z + np.divide(LAMBDA2,mu2), 1/mu2)
                # updating Lagrange multipliers
                LAMBDA2 = LAMBDA2 + np.multiply(mu2,Z - X)
                LAMBDA3 = LAMBDA3 + np.multiply(mu2, np.ones([1,self.num_columns]) - np.sum(Z,axis=0))

                auxiliary_constraint_residual = self.compute_mean_error(Z, X)
                affine_constraint_residual = self.compute_mean_error(np.sum(Z,axis=0), np.ones([1, self.num_columns]))

                residual = self.compute_mean_error(self.data, np.dot(self.data, X))
                self.residuals.append(residual)
                # reporting errors
                if (self.verbose and  (i_iter % self.printing_step == 0)):
                    print('\r\tADMM iteration %d, ||Z - C|| = %2.5e, ||1 - C^T 1|| = %2.5e' % (i_iter, auxiliary_constraint_residual, affine_constraint_residual), end="")
                i_iter += 1

            Err = [auxiliary_constraint_residual, affine_constraint_residual]

#             if(self.verbose):
#                 print ('\nTerminating ADMM at iteration %5.0f, \n ||Z - C|| = %2.5e, ||1 - C^T 1|| = %2.5e. \n' % (i_iter, auxiliary_constraint_residual,affine_constraint_residual))


        else:
#             print('CPU not affine')

            #A = np.linalg.inv(OP1 +  np.multiply(mu2, np.eye(self.num_columns, dtype=int)))
            A = OP1 +  np.multiply(mu2, np.eye(self.num_columns, dtype=int))

            while ( auxiliary_constraint_residual > auxiliary_constraint_tolerance and i_iter < self.num_iters):

                # updating Z
                OP2 = np.multiply(mu2, X) - LAMBDA2
                #Z = A.dot(OP1 + OP2)
                Z = np.linalg.solve(A, OP1 + OP2)
                
                # updating X
                X = Z + np.divide(LAMBDA2, mu2)
                X = self.shrinkL1Lq(X, 1/mu2)                
                
                # updating Lagrange multipliers
                LAMBDA2 = LAMBDA2 + np.multiply(mu2,Z - X)

                # computing errors
                auxiliary_constraint_residual = self.compute_mean_error(Z, X)

                # reporting errors
                if (self.verbose and  (i_iter % self.printing_step == 0)):                   
                    print('\t\rIteration %5.0f, ||Z - X|| = %2.5e' % (i_iter, auxiliary_constraint_residual), end="")
                i_iter += 1

            Err = [auxiliary_constraint_residual, affine_constraint_residual]
            if(self.verbose):
                print ('\n\tTerminating ADMM at iteration %5.0f, \n ||Z - X|| = %2.5e' % (i_iter, auxiliary_constraint_residual))

#         print("-ADMM- Time = %s seconds" % (time.time() - start_time))

        return X, Err  
    
    
class REM():

    def __init__(self, data, initial_sol=None, alpha=10, norm_type=1, weights=None,
                verbose=False, printing_step=5, TOLERANCE=[10**-8,-1], num_admm_iters=5000,
                affine=False,
                normalize=True,
                PCA=False, num_pca_components=10, GPU=False):

        self.data = data
        self.alpha = alpha
        self.norm_type=norm_type
        self.verbose = verbose
        self.printing_step = printing_step
        self.TOLERANCE = TOLERANCE
        self.num_admm_iters = num_admm_iters
        self.affine = affine
        self.normalize = normalize
        self.PCA = PCA
        self.num_pca_components = num_pca_components

        self.num_rows = data.shape[0]
        self.num_columns = data.shape[1]        
        if weights is None:
            self.weights = np.ones(self.num_rows)
        else:
            self.weights = weights
        self.initial_sol = initial_sol            
        
    def remove_near_duplicate_representatives(self, coarse_representative_indices, refining_threshold):

        '''
        This function takes the data matrix and the indices of the representatives and removes the representatives
        that are too close to each other

        :param coarse_representative_indices: indices of the representatives
        :param refining_threshold: threshold for pruning the representatives, typically in [0.9,0.99]
        :return: representatives indices
        '''
        from sklearn.preprocessing import normalize

        Ys = self.data[:, coarse_representative_indices]
        Ys = normalize(Ys, axis=0, norm='l2')
        num_coarse_representatives = Ys.shape[1]
        d = np.zeros([num_coarse_representatives, num_coarse_representatives])

        # Computes a the distance matrix for all selected columns by the algorithm
        for i in range(0, num_coarse_representatives-1):
            for j in range(i+1, num_coarse_representatives):
                d[i,j] = np.linalg.norm(Ys[:,i] - Ys[:,j])

        d = d + d.T # define symmetric matrix

        dsorti = np.argsort(d,axis=0)[::-1]
        dsort = np.flipud(np.sort(d,axis=0))

        pind = np.arange(0, num_coarse_representatives)
        for i in range(0, num_coarse_representatives):
            if np.any(pind==i) == True:
                cum = 0
                t = -1
                while cum <= (refining_threshold * np.sum(dsort[:,i])):
                    t += 1
                    cum += dsort[t, i]

                pind = np.setdiff1d(pind, np.setdiff1d( dsorti[t:,i], np.arange(0,i+1), assume_unique=True), assume_unique=True)

        refined_indices = coarse_representative_indices[pind]

        return refined_indices



    def find_reprentatives(self, C, thr, norm):
        
#         print ('Finding most representative objects')
        def find_significant_rows(C, thr, norm_type):
            '''
            This function takes the coefficient matrix with few nonzero rows and computes the indices of the nonzero rows
            :param X: NxN coefficient matrix
            :param thr: threshold for selecting the nonzero rows of X, typically in [0.9,0.99]
            :param norm_type: value of norm used in the L1/Lq minimization program in {1,2,inf}
            :return: the representatives indices on the basis of the ascending norm of the row of X (larger is the norm of
            a generic row most representative it is)
            '''
            N = C.shape[0]

            r = np.zeros([1,N])

            for i in range(0, N):

                r[:,i] = np.linalg.norm(C[i,:], norm_type)

            nrmInd = np.argsort(r)[0][::-1] #descending order
            nrm = r[0,nrmInd]
            nrmSum = 0

            j = []
            for j in range(0,N):
                nrmSum = nrmSum + nrm[j]
                if ((nrmSum/np.sum(nrm)) > thr):
                    break

            cssInd = nrmInd[0:j+1]
            return cssInd
        
        return find_significant_rows(C, thr, norm)

    def rem(self):

        '''
        '''        
        coarse_threshold = 0.85
        refining_threshold = 0.95

        # Normalization by subtracting mean from sample
        if self.normalize == True:
            self.data = self.data - np.matlib.repmat(np.mean(self.data, axis=1), self.num_columns,1).T


        if (self.PCA == True):
#             print ('Performing PCA...')
            pca = PCA(n_components = self.num_pca_components)
            self.data = pca.fit_transform(self.data).T
#             print('   Data shape after PCA: {}'.format(self.data.shape))
            self.num_columns = self.data.shape[0]
            self.num_rows = self.data.shape[0]
            self.num_columns = self.data.shape[1]
#             self.weighted_lasso_solver.data = self.data
#             self.weighted_lasso_solver.num_columns = self.num_columns
#             self.weighted_lasso_solver.num_rows = self.num_rows
#             print('   After PCA, row number: {}, column number: {}'.format(self.num_rows, self.num_columns))
        
        admm_regularization_params = [self.alpha, self.alpha]
        weighted_lasso_solver = WeightedLassoADMM(data=self.data, regularization_params=admm_regularization_params, norm_type=self.norm_type, 
                                                       weights=self.weights, verbose=self.verbose, 
                                                       printing_step=self.printing_step, TOLERANCE=self.TOLERANCE,
                                                        num_iters=self.num_admm_iters, affine=self.affine, 
                                                       initial_sol=self.initial_sol)
        self.X,_ = weighted_lasso_solver.run_weighted_lasso_admm()

        self.coarse_representative_indices = self.find_reprentatives(self.X, coarse_threshold, self.norm_type)

        self.refined_representative_indices = self.remove_near_duplicate_representatives(
            self.coarse_representative_indices, refining_threshold)
        self.residuals = weighted_lasso_solver.residuals

        return self.coarse_representative_indices, self.refined_representative_indices, self.X       
    