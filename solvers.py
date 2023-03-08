import numpy as np, cvxpy as cp
from tqdm import tqdm
from utils import *


''' Solvers for PCD Registration prooblem '''

class LinearRelaxationSolver:

    R0 = np.eye(3)
    
    def solve(self, X, Y):
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



class SVDSolver:
    pass


class ConvexRelaxationSolver:
    pass