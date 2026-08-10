import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import median_filter
from tqdm import tqdm


def r1_decomposition(X: np.ndarray, tol: float = 1e-6, max_iter: int = 100, dtype=np.float64):
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
    X = np.asarray(X, dtype=dtype)
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X must be a non-empty 2D array")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    if tol <= 0 or max_iter < 1:
        raise ValueError("tol must be positive and max_iter must be at least 1")
    if not np.any(X):
        return np.zeros((X.shape[0], 1), dtype=dtype), np.zeros((X.shape[1], 1), dtype=dtype)

    a = np.ones((X.shape[0], 1), dtype=dtype)
    b = np.ones((X.shape[1], 1), dtype=dtype)
    for _ in range(max_iter):
        b_denom = (b.T @ b).item()
        if b_denom <= np.finfo(dtype).eps:
            break
        a_new = X @ b / b_denom
        a_denom = (a_new.T @ a_new).item()
        if a_denom <= np.finfo(dtype).eps:
            return np.zeros_like(a), np.zeros_like(b)
        b_new = X.T @ a_new / a_denom
        if np.linalg.norm(a_new - a) < tol and np.linalg.norm(b_new - b) < tol:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    return a, b


def _filter(data: np.ndarray, size: int = 10, filter=median_filter, **filter_kwargs):
    return filter(data, size=(size, 1), **filter_kwargs)


def _optimize_single_channel(
    s: np.ndarray,
    b: np.ndarray,
    lam=0.5,
    sigma_min=1e-3,
    tau=2.0,
    max_iter=50,
    tol=1e-4,
    dtype=np.float64,
):

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
        sigma_new = (term + np.sqrt(term**2 + 4 * t * np.sum(residual**2))) / (2 * t)
        sigma_new = max(sigma_new, sigma_min)

        diff = np.linalg.norm(c_new - c) / (np.linalg.norm(c) + 1e-12)
        if diff < tol:
            c = c_new
            sigma = sigma_new
            break

        c = c_new
        sigma = sigma_new

    return c, sigma


def peak_recon(
    S: np.ndarray,
    B: np.ndarray,
    lam: float = 0.5,
    sigma_min: float = 1e-3,
    tau: float = 2.0,
    max_iter: int = 50,
    n_jobs=-1,
    dtype=np.float64,
):
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

    S = np.asarray(S, dtype=dtype)
    B = np.asarray(B, dtype=dtype)
    if S.ndim != 2 or S.shape != B.shape or 0 in S.shape:
        raise ValueError("S and B must be non-empty 2D arrays with the same shape")
    if sigma_min <= 0 or max_iter < 1:
        raise ValueError("sigma_min must be positive and max_iter must be at least 1")
    _T, M = S.shape
    results = Parallel(n_jobs=n_jobs)(
        delayed(_optimize_single_channel)(
            S[:, m], B[:, m], lam, sigma_min, tau, max_iter, dtype=dtype
        )
        for m in tqdm(range(M), desc="Parallel reconstruction")
    )

    C = np.column_stack([r[0] for r in results]).astype(dtype)
    sigma = np.array([r[1] for r in results], dtype=dtype)
    return C, sigma
