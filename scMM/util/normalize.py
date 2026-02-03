import numpy as np

NORM_REGISTRY = {}

def register_norm(name):
    def wrapper(func):
        NORM_REGISTRY[name] = func
        return func
    return wrapper

def _check_array(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("Input data must be a 2D array (n_samples, n_features)")
    return X

@register_norm("total")
def norm_total(X, params):
    
    X = _check_array(X)
    scale = params.get("scale", 1.0)

    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0

    return X / row_sum * scale

@register_norm("quantile")
def norm_quantile(X, params):
    
    X = _check_array(X)

    sorted_X = np.sort(X, axis=0)
    mean_quantiles = sorted_X.mean(axis=1)

    ranks = np.argsort(np.argsort(X, axis=0), axis=0)
    Xn = np.zeros_like(X)

    for j in range(X.shape[1]):
        Xn[:, j] = mean_quantiles[ranks[:, j]]

    return Xn

@register_norm("pqn")
def norm_pqn(X, params):
    """
    Anal. Chem. 2006, 78, 13, 4281-4290
    """
    X = _check_array(X)

    reference = params.get("reference", "median")

    if reference == "median":
        ref = np.median(X, axis=0)
    elif reference == "mean":
        ref = np.mean(X, axis=0)
    else:
        ref = np.asarray(reference, dtype=float)

    ref[ref == 0] = 1.0

    quotients = X / ref
    scale = np.median(quotients, axis=1, keepdims=True)
    scale[scale == 0] = 1.0

    return X / scale


@register_norm("zscore")
def norm_zscore(X, params):

    X = _check_array(X)

    axis = params.get("axis", 0)

    mean = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True)

    std[std == 0] = 1.0
    return (X - mean) / std

@register_norm("log")
def norm_log(X, params):
    
    X = _check_array(X)

    base = params.get("base", 2)
    pseudo = params.get("pseudo", 1e-6)

    Xp = X + pseudo

    if base == 2:
        return np.log2(Xp)
    elif base == 10:
        return np.log10(Xp)
    else:
        return np.log(Xp) / np.log(base)

@register_norm("minmax")
def norm_minmax(X, params):
    
    X = _check_array(X)

    axis = params.get("axis", 0)

    xmin = X.min(axis=axis, keepdims=True)
    xmax = X.max(axis=axis, keepdims=True)

    denom = xmax - xmin
    denom[denom == 0] = 1.0

    return (X - xmin) / denom

def _run_normalization(X, method, params):
    if method not in list(NORM_REGISTRY.keys()):
        raise ValueError(
            f"Unknown normalization method: {method}\n"
            f"Available: {list(NORM_REGISTRY.keys())}"
        )
    return NORM_REGISTRY[method](X, params)

def normalize(
    X,
    method="total",
    norm_kwargs=None,
    return_params=False
):
    """
    Parameters
    ----------
    X : array-like (n_samples, n_features)
    method : str
        total | quantile | pqn | zscore | log | minmax
    norm_kwargs : dict
        Parameters for normalization function.
    return_params : bool
        When true, normalization parameters will be returned in a dict.
    """

    norm_kwargs = norm_kwargs or {}

    Xn = _run_normalization(X, method, norm_kwargs)

    if return_params:
        return {
            "X_norm": Xn,
            "method": method,
            "norm_params": norm_kwargs
        }

    return Xn