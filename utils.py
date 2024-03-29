import numpy as np, open3d as o3d, matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


''' -------- Experiment Helper -------- '''
def sample_pcd(X, num_samples=None):
    assert X.shape[0] == 3, f"X should have shape (3, N), but instead got {X.shape}"
    if num_samples is None: return X
    
    idx = np.random.randint(low = 0, high = X.shape[1], size = num_samples)
    return X[:,idx]

def find_nn_corr(src, tgt):
    ''' Given two input point clouds, find nearest-neighbor correspondence (from source to target) 
    Input:
        - src: Source point cloud (n*3), either array or open3d pcd
        - tgt: Target point cloud (n*3), either array or open3d pcd
    Output:
        - idxs: Array indices corresponds to src points, 
            array elements corresponds to nn in tgt points (n, np.array)
    '''

    ''' Way1: Sklearn'''
    if src.shape[1] != 3: src = src.T
    if tgt.shape[1] != 3: tgt = tgt.T
    
    if not isinstance(src, np.ndarray):
        src = np.asarray(src.points)    # (16384*3)
        tgt = np.asarray(tgt.points)
    
    neighbors = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(tgt)
    dists, idxs = neighbors.kneighbors(src)  # (16384*1), (16384*1)
    return idxs.flatten()


def pcd_to_o3d(pcd):
    ''' Convert np array (n,3) to open3d pcd'''
    points = o3d.utility.Vector3dVector(pcd.reshape([-1, 3]))
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = points
    return points_o3d


def pcd_to_o3d_rgb(pcd, rgb):
    ''' Convert np array (n,3) to open3d pcd'''
    points = o3d.utility.Vector3dVector(pcd.reshape([-1, 3]))
    colors = o3d.utility.Vector3dVector(
        rgb.reshape([-1, 3]))  # TODO: What's this
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = points
    pcd_o3d.colors = colors
    return pcd_o3d



''' -------- Arithmetic --------'''
def skew_sym(x):    
    ''' Given a vector x, apply skew-symmetric operator '''
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])    

def vec_to_rot(R0, w):
    ''' Given a initial (3,3) rotation R0 and an (3,) angle w from R0
    compute new rotation matrix'''
    return R0 @ (np.eye(3) + skew_sym(w))

def mse(X, Y): return np.mean((X-Y)**2)

def sse(X,Y): return np.sum((X-Y)**2)



''' --------- Transformation ---------'''

def get_rand_rotation():
    ''' Get a random (3,3) rotation matrix as np.array
    ref: http://www.open3d.org/docs/latest/tutorial/Basic/transformation.html#Rotation
    '''
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    angles = np.random.uniform(-np.pi/2, np.pi/2, size=3)
    return mesh.get_rotation_matrix_from_xyz(angles)

def get_rand_translation(distance=1):
    ''' Get a random (3,) translation vector as np.array ''' 
    t = np.random.uniform(low=-1, high=1, size=3)
    return (t / np.linalg.norm(t)) * distance

def get_rand_transform():
    ''' Get a (4, 4) general transformation matrix in SE(3)'''
    transformation = np.eye(4)
    transformation[:3,:3] = get_rand_rotation()
    transformation[3,:3] = get_rand_translation()
    return transformation
    
def gaussian_corruption(pcd, std=0.03):
    noise = np.random.normal(0, std, size = pcd.shape)
    return pcd + noise




''' --------- Visualization --------- '''

def plot_pcd(pcd, ax_lim=None, path=None): 
    ''' Plot np.array (n,3)
        - ax_lim: 2d List (3,2). Min, max limit for x,y,z axis.
    '''
    if not isinstance(pcd, np.ndarray):
        pcd = np.array(pcd.points)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_proj_type('persp')
    if ax_lim is None:
        ax.set_xlim3d([-3,3]), ax.set_ylim3d([-3,3]), ax.set_zlim3d([-3,3])
    else:
        ax.set_xlim3d(ax_lim[0]); ax.set_ylim3d(ax_lim[1]); ax.set_zlim3d(ax_lim[2])

    ax.scatter(pcd[:, 0], pcd[:, 2], pcd[:, 1], marker='.', alpha=1.0, edgecolors='none')

    plt.show()
    if path is not None: plt.savefig(f"./imgs/{path}.png")
    plt.clf()


def compare_pcd(pcds, labels=None, path=None):
    if labels is None: 
        labels = ['Original', 'Corrupted', 'Recovered']

    dpi = 80
    fig = plt.figure(figsize=(1440/dpi, 720/dpi), dpi=dpi)
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim3d([-3,3]), ax.set_ylim3d([-3,3]), ax.set_zlim3d([-3,3])
    ax.set_proj_type('persp')

    for points, label in zip(pcds, labels):
        ax.scatter(points[:, 0], points[:, 2], points[:, 1],
                marker='.', alpha=0.5, edgecolors='none', label=label)

    plt.legend()
    

    plt.show()
    if path is not None:
        if path is not None: plt.savefig(f"./imgs/{path}.png")
    plt.clf()
        