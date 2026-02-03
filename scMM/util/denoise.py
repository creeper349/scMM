import numpy as np
from scipy.ndimage import median_filter
from joblib import Parallel, delayed
from tqdm import tqdm

def r1_decomposition(X:np.ndarray, tol:float=1e-6, max_iter:int = 100, dtype = np.float64):
    """
    Generate two vectors to approximate the input matrix X by their dot product.
    i.e. Find a, b to minimize ||X - a b^T||_F
    
    :param X: input matrix
    :type X: np.ndarray
    :param tol: delta ||a|| and ||b|| for early stopping
    :type tol: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :param dtype: data type for computation
    :type dtype: data type
    """
    a, b = np.ones((X.shape[0], 1), dtype = dtype), np.ones((X.shape[1], 1), dtype = dtype)
    for _ in range(max_iter):
        a_new = X @ b / (b.T @ b)
        b_new = X.T @ a_new / (a_new.T @ a_new)
        if np.linalg.norm(a_new - a) < tol and np.linalg.norm(b_new - b) < tol:
            break
        a, b = a_new, b_new
    return a, b

def _filter(data:np.ndarray, size:int = 10, filter = median_filter, **filter_kwargs):
    return filter(data, size = (size, 1), **filter_kwargs)

def _optimize_single_channel(s:np.ndarray, b:np.ndarray, lam=0.5, sigma_min=1e-3, tau=2.0, max_iter=50, tol=1e-4, dtype=np.float64):

    r = s - b
    sigma = np.maximum(np.std(r), sigma_min)
    c = np.zeros_like(r, dtype=dtype)
    for _ in range(max_iter):
        c_tilde = r - lam * sigma
        c_new = np.maximum(c_tilde - tau * sigma, 0.0)
        residual = s - b - c_new
        sum_c = np.sum(c_new)
        t = len(r)
        term = lam * sum_c
        sigma_new = (term + np.sqrt(term ** 2 + 4 * t * np.sum(residual ** 2))) / (2 * t)
        sigma_new = max(sigma_new, sigma_min)

        diff = np.linalg.norm(c_new - c) / (np.linalg.norm(c) + 1e-12)
        if diff < tol:
            c = c_new
            sigma = sigma_new
            break

        c = c_new
        sigma = sigma_new

    return c, sigma

def peak_recon(S:np.ndarray, B:np.ndarray, 
               lam:float = 0.5, 
               sigma_min:float = 1e-3, 
               tau:float = 2.0,
               max_iter:int = 50, 
               n_jobs=-1,
               dtype=np.float64):
    """
    Function to reconstruct CyESI signal by separating gaussian baseline and sparse peaks.
    
    :param S: Signal matrix
    :type S: np.ndarray
    :param B: Precomputed baseline matrix
    :type B: np.ndarray
    :param lam: Hyperparameter for L1 regularization to control cell signal sparsity
    :type lam: float
    :param sigma_min: Minimum variance for baseline noise
    :type sigma_min: float
    :param tau: Hyperparameter of soft thresholding to control peak sparsity
    :type tau: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :param n_jobs: number of parallel jobs to run
    :type n_jobs: int
    :param dtype: data type for computation
    :type dtype: data type
    :return: Reconstructed peak matrix and estimated noise standard deviations
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    T, M = S.shape
    results = Parallel(n_jobs = n_jobs)(
        delayed(_optimize_single_channel)(S[:, m], B[:, m], lam, sigma_min, tau, max_iter, dtype=dtype)
        for m in tqdm(range(M), desc="Parallel reconstruction")
    )

    C = np.column_stack([r[0] for r in results]).astype(dtype)
    sigma = np.array([r[1] for r in results], dtype=dtype)
    return C, sigma