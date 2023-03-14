import numpy as np, cvxpy as cp
from numpy.linalg import *
from tqdm import tqdm
from utils import *


''' Solvers for PCD Registration prooblem '''

class LinearRelaxationSolver:

    R0 = np.eye(3)
    
    def solve(self, X, Y):
        '''
        Return 
            - R0: Optimal rotation (that's iteratively optimized)
            - t : Optimal translation
        '''
        assert X.shape[0] == 3, f"X should have shape (3,N), but instead got {X.shape}"

        meanX, meanY = np.mean(X, axis=1), np.mean(Y, axis=1)
        t = meanY - meanX       
        X = (X.T + t).T

        for _ in tqdm(range(200)):
            A = -np.concatenate([self.R0 @ skew_sym(x) for x in X.T])    # (N*3, 3)
            B = np.concatenate([y- self.R0@x for x,y in zip(X.T, Y.T)])  # (N*3, )

            w = cp.Variable(3)
            prob = cp.Problem(
                cp.Minimize(cp.norm((A@w - B), p='fro')), 
                [(cp.norm2(w))<=1e-2]
            )

            try:
                prob.solve()
                w = w.value
                if w is None: break
            except Exception: break
            
            R = vec_to_rot(self.R0, w)
            self.R0 = R

        t = meanY - (self.R0 @ meanX)

        return self.R0, t


class LeastSquareSolver:
    # Ref: https://web.stanford.edu/class/cs273/refs/umeyama.pdf
    
    def solve(self, X, Y):
        ''' 
        Return:
            - R: Optimal rotation 
            - t: Optimal translation
        '''
        assert X.shape[0] == 3, f"X should have shape (3,N), but instead got {X.shape}"
        meanX, meanY = np.mean(X, axis=1), np.mean(Y, axis=1)
        X = (X.T - meanX).T   # (3, N)
        Y = (Y.T - meanY).T   # (3, N)

        U, _, Vh = svd(Y @ X.T)
        R = U @ Vh
        if det(R) == -1:
            Vh[-1,:] = -Vh[-1,:]
            R = U @ Vh
        t = meanY - (R @ meanX)
        return R, t



class ConvexRelaxationSolver:
    # Ref: https://arxiv.org/pdf/1401.3700.pdf

    def solve(self, X, Y):
        
        R = cp.Variable((3,3))
        t = cp.Variable(3)

        constraints = [ 1+R[0][0]+R[1][1]+R[2][2] >= 0, R[2][1]-R[1][2] >= 0, R[0][2]-R[2][0] >= 0, R[1][0]-R[0][1] >= 0, \
                        1+R[0][0]-R[1][1]-R[2][2] >= 0, R[1][0]+R[0][1] >= 0, R[0][2]+R[2][0] >= 0, \
                        1-R[0][0]+R[1][1]-R[2][2] >= 0, R[2][1]+R[1][2] >=0, \
                        1-R[0][0]-R[1][1]+R[2][2] >= 0 ]
        prob = cp.Problem(
            cp.Minimize(cp.norm((R@X.T+t - Y.T), p='fro')), 
            constraints
        )

        prob.solve()
        R = R.value
        t = t.value

        return R, t