import scipy.io
import numpy as np 
from scipy.sparse.linalg import eigsh
import os

def SO3_EIG(pred_path , gt_path, log_dir):
    mat = scipy.io.loadmat(gt_path)
    ncams_c = mat['ncams_c'][0][0]
    R_gt_c = mat['R_gt_c']
    A = np.ones((ncams_c , ncams_c))
    G = scipy.io.loadmat(pred_path)['data']

    D = np.kron(np.diag(1.0/np.sum(A , 1)) , np.eye(3))
    M  = eigsh(np.matmul(D , G) ,  k=3)[1]
    M[: ,[0 , 2]] = M[: , [2 , 0]]

    R = np.zeros((3, 3, ncams_c))

    for i in range(ncams_c):
        U , s , V = np.linalg.svd(M[3*i :3*i + 3 , :])
        R[: , : , i] = np.matmul(U  , V.T)
        if (np.linalg.det(R[: , : , i]) < 0):
            R[: , : , i]-= np.matmul(U  , V.T)
    
    data = {'data':R}
    scipy.io.savemat(os.path.join(log_dir ,"R_our.mat"), data)



        

