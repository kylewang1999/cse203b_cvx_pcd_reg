import numpy as np, open3d as o3d, matplotlib.pyplot as plt



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




''' --------- Transformation ---------'''

def get_rand_rotation():
    ''' Get a random (3,3) rotation matrix as np.array
    ref: http://www.open3d.org/docs/latest/tutorial/Basic/transformation.html#Rotation
    '''
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    angles = np.random.uniform(-np.pi/2, np.pi/2, size=3)
    return mesh.get_rotation_matrix_from_xyz(angles)

def get_rand_translation():
    ''' Get a random (3,) translation vector as np.array ''' 
    print()
    return np.random.uniform(size=3)

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


def compare_pcd(points1, points2, label1='original', label2='corrupted', path=None):
    if not isinstance(points1, np.ndarray):
        points1 = np.asarray(points1.points)    # (16384*3)
        points2 = np.asarray(points2.points)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim3d([-3,3]), ax.set_ylim3d([-3,3]), ax.set_zlim3d([-3,3])
    ax.set_proj_type('persp')

    ax.scatter(points1[:, 0], points1[:, 2], points1[:, 1],
               marker='.', alpha=0.5, edgecolors='none', label=label1)
    ax.scatter(points2[:, 0], points2[:, 2], points2[:, 1],
               marker='.', alpha=0.2, edgecolors='none',label=label2)
    plt.legend()
    

    plt.show()
    if path is not None:
        if path is not None: plt.savefig(f"./imgs/{path}.png")
        