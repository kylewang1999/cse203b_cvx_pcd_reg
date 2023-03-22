import copy, numpy as np, cvxpy as cp
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

        for _ in (range(50)):
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

            if (np.linalg.norm(self.R0-R) < 1e-6):
                break
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

    def solve(self, X, Y, max_iters=50):
        
        r = cp.Variable((3,3))
        t = cp.Variable(3)

        C = cp.bmat([
            [1 + r[0][0] + r[1][1] + r[2][2], r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]],
            [r[2][1] - r[1][2], 1 + r[0][0] - r[1][1] - r[2][2], r[1][0] + r[0][1], r[0][2] + r[2][0]],
            [r[0][2] - r[2][0], r[1][0] + r[0][1], 1 - r[0][0] + r[1][1] - r[2][2], r[2][1] + r[1][2]],
            [r[1][0] - r[0][1], r[0][2] + r[2][0], r[2][1] + r[1][2], 1 - r[0][0] - r[1][1] + r[2][2]]
        ])
        constraints = [C >> 0]

        prob = cp.Problem(
            cp.Minimize(cp.norm(( r @ X + cp.vstack([t for _ in range(X.shape[1])]).T - Y), p='fro')), 
            # cp.Minimize(cp.norm(( X.T @ r + t - Y.T), p='fro')), 
            constraints
        )

        opt = prob.solve(solver='SCS', max_iters=max_iters, verbose=False)
        r = r.value
        t = t.value

        if np.linalg.norm(r@r.T-np.eye(3)) > 1e-3:
            u,s,vh = np.linalg.svd(r)
            r = u @ vh

        return r, t




def icp(solver, src, tgt, iters=10):
    '''(self-implemented) Iterative closest point to align src and tgt point clouds
    Input:
        - solver: used for each iteration. Choose from the above 3
        - src, tgt: Open3d point clouds (3, N)
    '''
    if src.shape[0] != 3: src = src.T
    if tgt.shape[0] != 3: tgt = tgt.T

    ''' Initialized `R, t, trans_pred` '''
    # trans_pred = init_transformation(src, tgt,init_trans=init_trans, translate_only=True)  

    R, t = np.eye(3), np.zeros(3)

    for _ in tqdm(range(iters)):

        src_ = ((R @ copy.deepcopy(src)).T + t).T
        corr = find_nn_corr(src_, tgt)                  # Find correspondence

        R_, t_ = solver.solve(src_, tgt[:, corr])       # Align

        if (np.linalg.norm(R_-R) < 1e-6):
            break

        R = R_ @ R              # Update `R, t, trans_pred`
        t = R_ @ t + t_
        compare_pcd([tgt.T, src_.T])

    return R, t