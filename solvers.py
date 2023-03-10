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
        '''
        assert X.shape[0] == 3, f"X should have shape (3,N), but instead got {X.shape}"
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
        return self.R0


class LeastSquareSolver:
    # Ref: https://web.stanford.edu/class/cs273/refs/umeyama.pdf
    
    def solve(self, X, Y):
        ''' 
        Return:
            - R: Optimal rotation 
            - t: Optimal translation
        '''
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
    
    # def solve(self, src, tgt):
    #     src = src.T
    #     tgt = tgt.T
    #     # 1. Produce mean-subtracted P, Q matrices from src, tgt
    #     src_mean, tgt_mean = np.mean(src, axis=0), np.mean(tgt, axis=0)     # (3,)
    #     src = src - src_mean   # (n, 3)
    #     tgt = tgt - tgt_mean   # (n, 3)
    #     P, Q = src.T, tgt.T    # (3, n)

    #     # 2. Perform singular value decomposition
    #     M = Q @ P.T    # (3, 3) = (3, n) * (n, 3)
    #     U, S, Vh = np.linalg.svd(M)

    #     # 3. Compute rotation from SVD result
    #     R_ = U @ Vh
    #     if np.linalg.det(R_) == -1:
    #         Vh[-1, :] = -Vh[-1, :]
    #         R_ = U @ Vh

    #     # 4. Compute translation
    #     t_ = tgt_mean - (R_ @ src_mean)

    #     # 5. Update initial R, t
    #     # R = R_ @ R
    #     # t = R_ @ t  + t_

    #     return R_, t_
        



class ConvexRelaxationSolver:
    pass